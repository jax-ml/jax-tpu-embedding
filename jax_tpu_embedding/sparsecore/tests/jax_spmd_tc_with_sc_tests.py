# Copyright 2024 The JAX SC Authors.
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
from collections.abc import Callable
import functools
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
import einops
import flax.linen as nn
import jax
# pylint: disable-next=g-importing-member
from jax.lax import with_sharding_constraint
import jax.numpy as jnp
# pylint: disable-next=g-importing-member
from jax.sharding import Mesh
# pylint: disable-next=g-importing-member
from jax.sharding import NamedSharding
# pylint: disable-next=g-importing-member
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax


_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 128, 'Vocabulary size.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 8, 'Batch size.')

_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')

_SEQ_LEN = flags.DEFINE_integer(
    'sequence_length', 16, 'Sequence length of context words.'
)

_NUM_TABLES = flags.DEFINE_integer(
    'num_tables', 1, 'Number of tables to create.'
)

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1000, 'Number of steps to train for.'
)


_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 8, 'Embedding size.')


class ShakespeareSpmdModel(nn.Module):
  vocab_size: int
  hidden_size: int
  embedding_size: int
  mesh: Mesh

  dense_init: Callable[[Any, Any, Any], Any] = nn.initializers.lecun_normal()

  @nn.compact
  def __call__(self, emb_activations):
    # emb_activations is of shape (batch_size, seq_len, num_tables,
    # embedding_size)

    # fully replicated input
    x = with_sharding_constraint(
        emb_activations, NamedSharding(self.mesh, P(None, None, None, None))
    )

    # Concat sequence embeddings (Retain batch dimension)
    x = jnp.reshape(x, (x.shape[0], -1))

    # This dense is performed on TC (model sharded along first dimension).
    x = nn.Dense(
        self.hidden_size,
        kernel_init=nn.with_partitioning(
            self.dense_init, ('device', None), self.mesh
        ),
    )(x)
    # The output of this dense is of shape (batch_size, hidden_size)

    # This dense is performed on TC (model sharded along first dimension).
    x = nn.Dense(
        self.vocab_size,
        kernel_init=nn.with_partitioning(
            self.dense_init, ('device', None), self.mesh
        ),
    )(x)
    # The output of this dense is of shape (batch_size, vocab_size)

    return x


class ShakespeareTest(absltest.TestCase):
  """Input processing and convergence of basic model using JAX SPMD libraries."""

  # TODO: b/356880228 - refactor setUp and split in to functions possibly using
  # decorators?
  def setUp(self):
    super().setUp()
    self.devices = jax.devices()
    self.mesh = Mesh(np.array(self.devices), axis_names=['device'])
    self.num_sc_per_device = utils.num_sparsecores_per_device(self.devices[0])

    self.shakespeare_table_spec = embedding_spec.TableSpec(
        vocabulary_size=_VOCAB_SIZE.value,
        embedding_dim=_EMBEDDING_SIZE.value,
        initializer=lambda: jnp.zeros(
            (_VOCAB_SIZE.value, _EMBEDDING_SIZE.value),
            dtype=jnp.float32,
        ),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='shakespeare_table',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.shakespeare_feature = embedding_spec.FeatureSpec(
        table_spec=self.shakespeare_table_spec,
        input_shape=(_BATCH_SIZE.value * _SEQ_LEN.value * _NUM_TABLES.value, 1),
        output_shape=(
            _BATCH_SIZE.value * _SEQ_LEN.value * _NUM_TABLES.value,
            self.shakespeare_table_spec.embedding_dim,
        ),
        name='shakespeare_feature',
    )

    word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
    feature_batches, self.label_batches = shakespeare_data.word_id_batches(
        word_ids,
        _NUM_STEPS.value,
        _BATCH_SIZE.value,
        _SEQ_LEN.value,
        _NUM_TABLES.value,
    )
    self.feature_batches = feature_batches['words_0']

    emb_table = np.zeros([_VOCAB_SIZE.value, _EMBEDDING_SIZE.value])
    emb_table_sharded = einops.rearrange(
        emb_table,
        '(v c s) f -> c (s v) f',
        c=len(self.devices),
        s=self.num_sc_per_device,
    )

    self.embedding_variables = {}
    self.embedding_variables[self.shakespeare_table_spec.name] = [
        jax.device_put(
            emb_table_sharded[i],
            device=device,
        )
        for i, device in enumerate(self.devices)
    ]
    sharding = NamedSharding(self.mesh, P('device', None))
    self.embedding_variables[self.shakespeare_table_spec.name] = (
        embedding.EmbeddingVariables(
            table=jax.make_array_from_single_device_arrays(
                shape=(_VOCAB_SIZE.value, _EMBEDDING_SIZE.value),
                sharding=sharding,
                arrays=self.embedding_variables[
                    self.shakespeare_table_spec.name
                ],
            ),
            slot=(),
        )
    )
    # Construct the model.
    self.model = ShakespeareSpmdModel(
        vocab_size=_VOCAB_SIZE.value,
        hidden_size=8,
        embedding_size=_EMBEDDING_SIZE.value,
        mesh=self.mesh,
    )

    # Initialize the model.
    init_emb_activations = jnp.zeros((
        _BATCH_SIZE.value,
        _SEQ_LEN.value,
        _NUM_TABLES.value,
        _EMBEDDING_SIZE.value,
    ))
    self.params = self.model.init(jax.random.key(42), init_emb_activations)
    logging.info(
        'params: %s', jax.tree_util.tree_map(lambda x: x.shape, self.params)
    )

    # Create optimizer.
    self.tx = optax.adam(learning_rate=_LEARNING_RATE.value)
    self.opt_state = self.tx.init(self.params)

    def loss_fn(params, emb_acts, labels):
      logits = self.model.apply(params, emb_acts)
      xentropy = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
      return jnp.mean(xentropy)

    self.loss_grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

    embedding.prepare_feature_specs_for_training(
        self.shakespeare_feature,
        global_device_count=self.mesh.size,
        num_sc_per_device=self.num_sc_per_device,
    )
    sharded_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        global_device_count=self.mesh.size,
        feature_specs=(self.shakespeare_feature,),
        sharding_strategy='MOD',
    )
    self.sparse_matmul = jax.shard_map(
        sharded_matmul,
        mesh=self.mesh,
        in_specs=(P('device'), P('device', None)),
        out_specs=P('device'),
        check_vma=False,
    )

    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=(self.shakespeare_feature,),
        sharding_strategy='MOD',
    )
    self.sparse_grad_update = jax.shard_map(
        sharded_grad_update,
        mesh=self.mesh,
        in_specs=(P('device'), P('device'), P('device', None)),
        out_specs=P('device', None),
        check_vma=False,
    )

  def test_spmd_shakespeare_model_convergence(self):

    step = 0
    losses = []

    def train_step(
        params,
        opt_state,
        preprocessed_inputs,
        embedding_variables,
        labels,
    ):
      # SC forward pass
      activations = self.sparse_matmul(preprocessed_inputs, embedding_variables)
      activations = jnp.reshape(
          activations[0],
          (
              _BATCH_SIZE.value,
              _SEQ_LEN.value,
              _NUM_TABLES.value,
              _EMBEDDING_SIZE.value,
          ),
      )

      # TC forward/backward pass.
      loss_val, grads = self.loss_grad_fn(params, activations, labels)
      updates, new_opt_state = self.tx.update(grads[0], opt_state)
      params = optax.apply_updates(params, updates)

      # SC backward pass
      gradient_updates = jnp.reshape(grads[1], (-1, _EMBEDDING_SIZE.value))
      new_embedding_variables = self.sparse_grad_update(
          (gradient_updates,),  # Should be same structure as features.
          preprocessed_inputs,
          embedding_variables,
      )

      return params, new_opt_state, loss_val, new_embedding_variables

    for features, labels in zip(self.feature_batches, self.label_batches):
      features = np.reshape(features, (-1, 1))

      # SC input processing
      preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
          {self.shakespeare_feature.name: features},
          {
              self.shakespeare_feature.name: np.ones_like(
                  features, dtype=jnp.float32
              )
          },
          {self.shakespeare_feature.name: self.shakespeare_feature},
          local_device_count=self.mesh.local_mesh.size,
          global_device_count=self.mesh.size,
          num_sc_per_device=self.num_sc_per_device,
          sharding_strategy='MOD',
          batch_number=step,
      )
      self.params, self.opt_state, loss_val, self.embedding_variables = jax.jit(
          train_step
      )(
          self.params,
          self.opt_state,
          preprocessed_inputs,
          self.embedding_variables,
          labels,
      )

      if step % 10 == 0:
        logging.info('Loss step %s: %s', step, loss_val)
        losses.append(loss_val)

      step += 1

    self.assertLess(losses[-1], 0.001)


if __name__ == '__main__':
  absltest.main()
