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

"""Example for jax_tpu_embedding on singlehost with pjit.

Demonstrates how to use jax_tpu_embedding with pjit
"""

import functools
import math
from typing import Dict, Sequence, Tuple, Union

from absl import app
from absl import flags
from flax.core import scope as flax_scope
import flax.linen as nn
from flax.training import common_utils
from flax.training.train_state import TrainState
import jax
from jax.experimental import pjit
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax_tpu_embedding import input_utils
from jax_tpu_embedding import tpu_embedding as jte
from jax_tpu_embedding import tpu_embedding_utils
import numpy as np
import optax
import tensorflow as tf


jax.distributed.initialize()
tpu_embedding_utils.init_tpu_system()
Array = Union[jnp.ndarray, jnp.DeviceArray]  # pytype: disable=module-attr
Initializer = jax.nn.initializers.Initializer

NUM_TARGET_IDS = 5
NUM_WATCHES = 10
IS_REPLICATED = flags.DEFINE_boolean('is-replicated', True, 'whether to '
                                     'replicate the dense layers of the model'
                                     'accross devices or shard them. Defaults '
                                     'to True.')

configs = dict(
    global_batch_size=16,
    embedding_dimension=64,
    hidden_layer_dimension=32,
    num_hidden_layers=1,
    vocab_size=16,
    num_classes=4,
    learning_rate=1.0,
    dropout=0.5,
    num_targets=NUM_TARGET_IDS,
    is_training=True)


class MLPLayers(nn.Module):
  """Create mlp layers."""
  hidden_dim: int
  num_hidden_layers: int
  dropout: float
  num_classes: int
  kernel_init: Initializer = nn.initializers.glorot_uniform()
  bias_init: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self, x: Array, is_training: bool = False) -> Array:
    for _ in range(self.num_hidden_layers):
      x = nn.Dense(
          features=self.hidden_dim,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init)(
              x)
      x = nn.relu(x)

    if is_training:
      x = nn.Dropout(rate=self.dropout)(x, deterministic=False)
    x = nn.Dense(features=self.num_classes, bias_init=self.bias_init)(x)
    return x


def compute_one_hot_targets(targets: Array, num_classes: int,
                            on_value: float) -> Array:
  """Compute one hot encoded targets.

  Args:
    targets: An array of target value.
    num_classes: number of classes to one-hot encoding.
    on_value: Value to fill to non-zero locations.
  Returns:
    An array of one-hot encoded targets.
  """
  one_hot_targets = common_utils.onehot(targets, num_classes, on_value=on_value)
  one_hot_targets = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=1),
                                           one_hot_targets)
  return one_hot_targets


@jax.vmap
def categorical_cross_entropy_loss(logits: Array, one_hot_targets: Array):
  return -jnp.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)


def dummy_dataset(global_batch_size: int,
                  vocab_size: int,
                  num_classes: int,
                  seed: int = 123
                  ):
  """Create dummy sample inputs."""
  rng_state = np.random.RandomState(seed=seed)

  def _create_feature():
    watches = rng_state.randint(low=0, high=vocab_size,
                                size=NUM_WATCHES * global_batch_size)
    watches = tf.sparse.from_dense(watches.reshape(
        [global_batch_size, NUM_WATCHES]))
    targets = rng_state.randint(low=0, high=num_classes,
                                size=NUM_TARGET_IDS * global_batch_size)
    targets = tf.convert_to_tensor(
        targets.reshape([global_batch_size, NUM_TARGET_IDS]))
    return ({
        'watches': tf.sparse.reset_shape(
            watches, new_shape=[global_batch_size, vocab_size]),
    }, {
        'watches_target': tf.cast(targets, dtype=tf.float32),
    })
  ds = tf.data.Dataset.from_tensors(_create_feature())
  ds = ds.repeat()
  return ds


def build_embedding_configs(batch_size_per_device: int,
                            embedding_dimension: int,
                            vocab_size: int):
  """Create feature configurations for embedding layer.

  Args:
    batch_size_per_device: batch size of inputs to equeue.
    embedding_dimension: dimension size of embedding table.
    vocab_size: vocabulary size of embedding table.
  Returns:
    A dictionary of feature configurations.
  """
  feature_configs = {
      'watches': tf.tpu.experimental.embedding.FeatureConfig(
          table=tf.tpu.experimental.embedding.TableConfig(
              vocabulary_size=vocab_size,
              dim=embedding_dimension,
              initializer=tf.initializers.TruncatedNormal(
                  mean=0.0, stddev=1 / math.sqrt(embedding_dimension)),
              combiner='mean'),
          output_shape=[batch_size_per_device])
  }
  return feature_configs


def create_global_mesh(mesh_shape: Tuple[int, ...],
                       axis_names: Sequence[jax.pxla.MeshAxisName]) -> Mesh:  # pytype: disable=module-attr
  size = np.prod(mesh_shape)
  if len(jax.devices()) < size:
    raise ValueError(f'Test requires {size} global devices.')
  devices = sorted(jax.devices(), key=lambda d: d.id)
  mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
  global_mesh = Mesh(mesh_devices, axis_names)
  return global_mesh


def build_step(embedding_layer: jte.TPUEmbedding,
               train_state: TrainState,
               config_flags: Dict[str, Union[int, float]],
               is_training: bool,
               use_pjit: bool):
  """Build train or eval step using tpu embedding."""

  def forward(inputs):
    embedding_activations = inputs['embedding_actv']
    params = inputs['params']
    logits = train_state.apply_fn(params, embedding_activations['watches'])
    one_hot_targets = compute_one_hot_targets(
        inputs['watches_targets'],
        num_classes=config_flags['num_classes'],
        on_value=1.0 / config_flags['num_targets'])
    loss = categorical_cross_entropy_loss(logits, one_hot_targets)
    loss = jnp.sum(loss, axis=0) * (1.0 / config_flags['global_batch_size'])
    return loss

  def step_fn(train_state, watches_targets):
    embedding_activation = embedding_layer.dequeue()
    inputs = {
        'embedding_actv': embedding_activation,
        'params': train_state.params,
        'watches_targets': watches_targets,
    }
    if is_training:
      loss, grads = jax.value_and_grad(forward)(inputs)
      embedding_grads, params_grads = grads['embedding_actv'], grads['params']
      if not use_pjit:
        params_grads = jax.lax.pmean(params_grads, axis_name='devices')
        loss = jax.lax.pmean(loss, axis_name='devices')
      train_state = train_state.apply_gradients(grads=params_grads)
      embedding_layer.apply_gradients(embedding_grads)
    else:
      loss = forward(inputs)
    return loss, train_state

  return step_fn


def main(_):
  batch_size_per_device = configs['global_batch_size'] // jax.device_count()

  feature_configs = build_embedding_configs(
      batch_size_per_device=batch_size_per_device,
      embedding_dimension=configs['embedding_dimension'],
      vocab_size=configs['vocab_size'],
      )

  embedding_optimizer = tf.tpu.experimental.embedding.SGD(
      learning_rate=configs['learning_rate'])
  tpu_embedding_layer = jte.TPUEmbedding(
      feature_configs=feature_configs, optimizer=embedding_optimizer,
      # Pjit user must set `cores_per_replica` to enable TPUEmbedding SPMD
      cores_per_replica=jax.local_device_count())

  # Must call initialize_tpu_embedding to configure TPUEmbedding
  tpu_embedding_layer.initialize_tpu_embedding()

  # Call load_embedding_tables to initialize embedding tables.
  tpu_embedding_layer.load_embedding_tables()

  num_devices = jax.device_count()
  mesh_axis_names = ('x',)

  global_mesh = create_global_mesh((num_devices,), mesh_axis_names)

  partition_spec = PartitionSpec(mesh_axis_names)

  device_input_fn = input_utils.make_pjit_array_fn(global_mesh, partition_spec)

  ds = dummy_dataset(global_batch_size=configs['global_batch_size'],
                     vocab_size=configs['vocab_size'],
                     num_classes=configs['num_classes'])

  dummy_iter = input_utils.split_and_prefetch_to_host_and_devices(
      iterator=iter(ds),
      split_fn=lambda xs: {'host': xs[0], 'device': xs[1]},
      host_input_fn=input_utils.enqueue_prefetch(
          enqueue_fn=functools.partial(
              tpu_embedding_layer.enqueue, is_training=configs['is_training'])),
      device_input_fn=device_input_fn,
      buffer_size=2)

  # Create TrainState
  mlp_model = MLPLayers(
      hidden_dim=configs['hidden_layer_dimension'],
      num_hidden_layers=configs['num_hidden_layers'],
      dropout=configs['dropout'],
      num_classes=configs['num_classes'])

  init_params = mlp_model.init(
      jax.random.PRNGKey(123),
      jnp.ones((batch_size_per_device, configs['embedding_dimension'])))
  tx = optax.adagrad(learning_rate=configs['learning_rate'])

  train_state = TrainState.create(apply_fn=mlp_model.apply, params=init_params,
                                  tx=tx)

  train_step_fn = build_step(
      embedding_layer=tpu_embedding_layer,
      train_state=train_state,
      config_flags=configs,
      is_training=configs['is_training'],
      use_pjit=True)

  num_steps = 10
  is_replicated = IS_REPLICATED.value
  with global_mesh:
    if is_replicated:
      print('Running with Replicated dense layers')
      axis_resources = None
    else:
      print('Running with Sharded dense layers')
      axis_resources = get_sharded_axis_resources(train_state)

    new_train_state = pjit.pjit(
        lambda x: x,
        in_shardings=None,
        out_shardings=axis_resources,
        keep_unused=True,
    )(train_state)

    for step in range(num_steps):
      inputs = next(dummy_iter)
      loss, new_train_state = pjit.pjit(
          train_step_fn,
          in_shardings=(
              axis_resources,
              PartitionSpec(
                  'x',
              ),
          ),
          out_shardings=(None, axis_resources),
          keep_unused=True,
      )(new_train_state, inputs['device']['watches_target'])
      print('train_step = ', step, 'loss = ', loss)


def get_sharded_axis_resources(train_state):
  """Get sharding parameters."""
  params_resources = flax_scope.FrozenDict({
      'params': {
          'Dense_0': {
              'kernel': PartitionSpec('x', None),
              'bias': PartitionSpec('x',),
          },
          'Dense_1': {
              'kernel': PartitionSpec('x', None),
              'bias': PartitionSpec('x',),
          },
      },
  })

  return TrainState(  # pytype: disable=wrong-arg-types
      step=PartitionSpec(), apply_fn=train_state.apply_fn,
      params=params_resources,
      tx=train_state.tx,
      opt_state=(
          optax.ScaleByRssState(sum_of_squares=params_resources),
          optax.EmptyState()
          )
  )


if __name__ == '__main__':
  app.run(main)
