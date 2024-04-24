# Copyright 2024 The jax_tpu_embedding Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for jax tpu embedding support."""

import functools
from typing import Callable, Dict, Tuple

from absl.testing import parameterized
from flax import linen as nn
from flax.core import scope as flax_scope
from flax.training.train_state import TrainState
import jax
from jax.experimental import pjit
import jax.extend as jex
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding import input_utils
from jax_tpu_embedding import tpu_embedding as jte
from jax_tpu_embedding import tpu_embedding_utils as jte_utils
from jax_tpu_embedding.tests import test_utils
import numpy as np
import optax
import tensorflow as tf

Array = jax.Array
OptState = optax.OptState

_LEARNING_RATE = 0.1


class JaxTpuEmbeddingTestBase(parameterized.TestCase):

  class JaxDense(nn.Module):
    """Jax single dense layer."""
    num_dim: int

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(features=self.num_dim)(x)
      x = nn.relu(x)
      return x

  def setUp(self):
    super(JaxTpuEmbeddingTestBase, self).setUp()
    self.input_ds = test_utils.create_dummy_dataset(
        batch_size=test_utils.PER_CORE_BATCH_SIZE * jax.local_device_count())
    self.embedding_optimizer = tf.tpu.experimental.embedding.Adagrad(
        learning_rate=_LEARNING_RATE)
    self.learning_rate = _LEARNING_RATE
    self.feature_config_fn = functools.partial(
        test_utils.create_feature_config,
        batch_size=test_utils.PER_CORE_BATCH_SIZE)

    self.n_devices = jax.local_device_count()

    self.model = self.JaxDense(num_dim=12)

    # Expected loss.
    self.expected_loss = [2.2778971, 1.8492181]

  def init_params(self):
    sample_input = jnp.ones((1, self.model.num_dim))
    init_params = self.model.init(jax.random.PRNGKey(123), sample_input)
    return init_params


class JaxJaxTpuEmbeddingTest(JaxTpuEmbeddingTestBase):
  """Test Jax JaxTpuEmbedding APIs."""

  def setUp(self):
    """Initialize tpu embedding system."""
    jte_utils.init_tpu_system()
    super().setUp()

    self.split_fn = lambda xs: {'host': xs[0], 'device': xs[1]}

  def tearDown(self):
    """Tear down for Jax and TPUEmbedding."""
    super().tearDown()

    # Reset PjRt client.
    jex.backend.clear_backends()

    # Clear up tpu embedding
    jte_utils.shutdown_tpu_system()

  def _create_train_state(self) -> TrainState:
    """Initializes model parameters and optimizer state.

    Returns:
      A flax.training.TrainState with loaded model parameters and optax.
    """
    params = self.init_params()
    tx = optax.adagrad(learning_rate=self.learning_rate)
    return TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

  def _create_step_fn(
      self,
      train_state: TrainState,
      embedding_layer: jte.TPUEmbedding,
      is_training: bool = False,
      use_pjit: bool = False
  ) -> Callable[[Array, TrainState], Tuple[Array, TrainState]]:
    """Create train/eval step function.

    Args:
      train_state: A flax.training.TrainState defines model and optax.
      embedding_layer: A tpu embedding layer using JaxTpuEmbedding devices.
      is_training: If this step function is a training or eval step.
      use_pjit: If true, creating step_fn for pjit, otherwise for pmap.

    Returns:
      A callable function to run a train or eval step.
    """

    def loss_fn(inputs: Dict[str, Array]) -> Array:
      concat_features = jnp.concatenate(
          inputs['embedding_actv'] + (inputs['dense_feats'],), axis=-1)
      logits = train_state.apply_fn(inputs['params'], concat_features)
      loss = jnp.mean(logits)
      return loss

    def step_fn(
        dense_features: Array,
        train_state: TrainState) -> Tuple[Array, TrainState]:
      embedding_activations = embedding_layer.dequeue()

      # Group embedding_activations and model's params in a dict of inputs, so
      # that `jax.value_and_grad` returns gradients for both dense model and
      # corresponding embedding table.
      inputs = {
          'embedding_actv': embedding_activations,
          'dense_feats': dense_features,
          'params': train_state.params,
      }
      if is_training:
        loss, grads = jax.value_and_grad(loss_fn)(inputs)
        embedding_grads, params_grads = grads['embedding_actv'], grads['params']
        # Combine the gradient across all devices (by taking their mean).
        if not use_pjit:
          params_grads = jax.lax.pmean(params_grads, axis_name='x')
        train_state = train_state.apply_gradients(grads=params_grads)
        embedding_layer.apply_gradients(embedding_grads)
      else:
        loss = loss_fn(inputs)
      return loss, train_state

    return step_fn

  # TODO(b/298825635): Enable `True` for `is_training` after the fix.
  @parameterized.product(
      is_training=[False],
      use_shape_inference=[True, False],
  )
  def test_pmap(
      self, is_training, use_shape_inference
  ):
    """Test data parallelism with pmap."""
    # Create and initialize tpu embedding layer.
    embedding_layer = jte.TPUEmbedding(
        feature_configs=self.feature_config_fn(
            use_shape_inference=use_shape_inference),
        optimizer=self.embedding_optimizer)

    if not use_shape_inference:
      embedding_layer.initialize_tpu_embedding()
      embedding_layer.load_embedding_tables()

    # Define callable for embedding enqueue, dequeue and apply gradients.
    enqueue_fn = functools.partial(
        embedding_layer.enqueue, is_training=is_training)

    # set init params as stateless initialized weights.
    train_state = self._create_train_state()
    step_fn = self._create_step_fn(
        train_state=train_state,
        embedding_layer=embedding_layer,
        is_training=is_training)

    # Replicated train state on devices.
    train_state = jax.device_put_replicated(train_state, jax.local_devices())

    # Build input pipeline that prefetch for host and devices inputs.
    device_input_fn = input_utils.make_pmap_array_fn()
    iterator = input_utils.split_and_prefetch_to_host_and_devices(
        iterator=iter(self.input_ds),
        split_fn=self.split_fn,
        host_input_fn=input_utils.enqueue_prefetch(enqueue_fn),
        device_input_fn=device_input_fn)

    # Start one step training loop.
    feature = next(iterator)

    pmap_step_fn = jax.pmap(step_fn, axis_name='x')
    loss, _ = pmap_step_fn(feature['device']['dense'], train_state)
    self.assertIsInstance(loss, Array)
    self.assertLen(loss.addressable_shards, self.n_devices)
    self.assertListEqual(list(loss), self.expected_loss)

  # TODO(b/298825635): Enable `True` for `is_training` after the fix.
  @parameterized.product(
      is_training=[False],
      use_shape_inference=[True, False],
  )
  def test_pjit(
      self, is_training, use_shape_inference
  ):
    """Test model parallelism with pjit."""
    # Create and initialize tpu embedding layer.
    embedding_layer = jte.TPUEmbedding(
        feature_configs=self.feature_config_fn(
            use_shape_inference=use_shape_inference),
        optimizer=self.embedding_optimizer,
        cores_per_replica=jax.local_device_count())

    if not use_shape_inference:
      embedding_layer.initialize_tpu_embedding()
      embedding_layer.load_embedding_tables()

    # Define callable for embedding enqueue.
    enqueue_fn = functools.partial(
        embedding_layer.enqueue, is_training=is_training)

    train_state = self._create_train_state()

    params_pspecs = {
        'params': {
            'Dense_0': {
                'kernel': P('x', None),
                'bias': P('x',),
            },
        },
    }

    train_state_pspecs = TrainState(  # pytype: disable=wrong-arg-types  # dataclass_transform
        step=P(), apply_fn=train_state.apply_fn,
        params=params_pspecs,
        tx=train_state.tx,
        opt_state=(
            optax.ScaleByRssState(
                sum_of_squares=params_pspecs), optax.EmptyState()))

    step_fn = self._create_step_fn(
        train_state=train_state,
        embedding_layer=embedding_layer,
        is_training=is_training,
        use_pjit=True)

    global_mesh = test_utils.create_global_mesh((2,), ('x',))

    # Build input pipeline that prefetch for host and devices inputs.
    device_input_fn = input_utils.make_pjit_array_fn(global_mesh, (P('x',)))
    iterator = input_utils.split_and_prefetch_to_host_and_devices(
        iterator=iter(self.input_ds),
        split_fn=self.split_fn,
        host_input_fn=input_utils.enqueue_prefetch(enqueue_fn),
        device_input_fn=device_input_fn)

    # Train the model for one step.
    feature = next(iterator)

    with global_mesh:
      train_state = pjit.pjit(
          lambda x: x,
          in_shardings=None,
          out_shardings=train_state_pspecs,
          keep_unused=True)(train_state)

      loss, _ = pjit.pjit(
          step_fn,
          in_shardings=(P('x', None), train_state_pspecs),
          out_shardings=(None, train_state_pspecs),
          keep_unused=True)(
              feature['device']['dense'], train_state)
      self.assertIsInstance(loss, Array)
      self.assertLen(loss.addressable_shards, self.n_devices)

      # Since loss is replicated and compute by reduction mean, it should be
      # equal to self.expect_loss average.
      expect_loss = np.mean(self.expected_loss)
      self.assertAlmostEqual(
          loss.addressable_data(0), expect_loss)
      self.assertAlmostEqual(loss.addressable_data(1), expect_loss)


class TFJaxTpuEmbeddingTest(JaxTpuEmbeddingTestBase, tf.test.TestCase):
  """Test TF JaxTpuEmbedding API gives same expected results."""

  def setUp(self):
    # create tpu strategy.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.tpu.experimental.initialize_tpu_system(resolver)
    self._strategy = tf.distribute.TPUStrategy(resolver)

    # As base class would starts jax, it needs to start TF TPU first..
    super().setUp()

  def tearDown(self):
    # tear down for dtensor.
    jex.backend.clear_backends()
    tf.tpu.experimental.shutdown_tpu_system()
    super().tearDown()

  def _dense_model(self, dim_size):
    """TF dense layer."""
    inputs = tf.keras.Input(shape=(dim_size,))
    dense_layer = tf.keras.layers.Dense(dim_size, activation=tf.nn.relu)
    outputs = dense_layer(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

  @parameterized.named_parameters(('_train', True), ('_eval', False))
  def test_tf_jax_tpu_embedding(self, is_training):
    params = self.init_params()
    kernel_weights = params['params']['Dense_0']['kernel']
    bias_weights = params['params']['Dense_0']['bias']

    strategy = self._strategy
    with strategy.scope():
      tf_model = self._dense_model(dim_size=self.model.num_dim)
      tf_model.set_weights([kernel_weights, bias_weights])
      embedding_layer = tf.tpu.experimental.embedding.TPUEmbedding(
          feature_config=self.feature_config_fn(),
          optimizer=self.embedding_optimizer)
      model_optimizer = tf.keras.optimizers.Adagrad(
          learning_rate=self.learning_rate)

    dist_ds = strategy.experimental_distribute_dataset(
        self.input_ds,
        options=tf.distribute.InputOptions(experimental_fetch_to_device=False))
    dist_iter = iter(dist_ds)

    @tf.function
    def step_fn(embedding_features, dense_features, is_training=True):

      def eval_step(features):
        embedding_activations = embedding_layer.dequeue()
        concat_features = tf.concat(
            embedding_activations + (features,), axis=1)
        logits = tf_model(concat_features)
        loss = tf.reduce_mean(logits)
        return loss

      def train_step(features):
        with tf.GradientTape() as tape:
          embedding_activations = embedding_layer.dequeue()
          tape.watch(embedding_activations)
          concat_features = tf.concat(
              embedding_activations + (features,), axis=1)
          logits = tf_model(concat_features)
          loss = tf.reduce_mean(logits)

        model_gradients, embedding_gradients = tape.gradient(
            loss, [tf_model.trainable_variables, embedding_activations])
        embedding_layer.apply_gradients(embedding_gradients)
        model_optimizer.apply_gradients(
            list(zip(model_gradients, tf_model.trainable_variables)))
        return loss

      embedding_layer.enqueue(embedding_features, training=is_training)
      if is_training:
        loss = strategy.run(train_step, args=(dense_features,))
      else:
        loss = strategy.run(eval_step, args=(dense_features,))
      return loss

    sparse_features, dense_features = next(dist_iter)
    loss = step_fn(sparse_features, dense_features['dense'],
                   is_training=is_training)
    self.assertIsInstance(loss, tf.distribute.DistributedValues)

    loss = list(strategy.experimental_local_results(loss))
    self.assertLen(loss, self.n_devices)
    self.assertListEqual(loss, self.expected_loss)


if __name__ == '__main__':
  tf.test.main()
