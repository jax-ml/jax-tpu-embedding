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
import functools
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import flax_model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.flax import embed_optimizer
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax


shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested

FLAGS = flags.FLAGS

np.set_printoptions(threshold=np.inf)

_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 4096, 'Vocabulary size.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 8, 'Batch size.')

_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')

_SEQ_LEN = flags.DEFINE_integer(
    'sequence_length', 16, 'Sequence length of context words.'
)

_NUM_TABLES = flags.DEFINE_integer(
    'num_tables', 1, 'Number of tables to create.'
)

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 100, 'Number of steps to train for.'
)

_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    1,
    'Number of separate epochs.',
)

_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 16, 'Embedding size.')
_EMBEDDING_INIT = flags.DEFINE_enum(
    'embedding_init', 'normal', ['normal', 'row_id'], 'Embedding initializer.'
)


################################################################################
# Define the shakespeare test.
################################################################################
class ShakespeareTest(absltest.TestCase):
  """Input processing and convergence of basic model using JAX SC libraries."""

  def test_shakespeare_model_loss_convergence(self):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, 'x')

    word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
    feature_batches, label_batches = shakespeare_data.word_id_batches(
        word_ids,
        _NUM_STEPS.value,
        _BATCH_SIZE.value,
        _SEQ_LEN.value,
        _NUM_TABLES.value,
    )
    feature_batches = feature_batches['words_0']

    # Define feature/table specs.
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=_VOCAB_SIZE.value,
        embedding_dim=_EMBEDDING_SIZE.value,
        initializer=jax.nn.initializers.ones,
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=1),
        combiner='sum',
        name='shakespeare',
        max_ids_per_partition=64,
        max_unique_ids_per_partition=64,
    )

    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(_BATCH_SIZE.value * _SEQ_LEN.value, 1),
        output_shape=(
            _BATCH_SIZE.value * _SEQ_LEN.value,
            _EMBEDDING_SIZE.value,
        ),
        name='shakespeare_feature',
    )

    feature_specs = nn.FrozenDict({'shakespeare_feature': feature_spec})
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )

    # Construct the model.
    model = shakespeare_model.Model(
        feature_specs=feature_specs,
        global_batch_size=_BATCH_SIZE.value,
        vocab_size=_VOCAB_SIZE.value,
        seq_len=_SEQ_LEN.value,
        embedding_size=_EMBEDDING_SIZE.value,
        mesh=mesh,
        sharding_axis='x',
    )

    # Initialize the model.
    def process_inputs(batch_number, feature_batch):
      features = np.reshape(feature_batch, (-1, 1))
      feature_weights = np.ones(features.shape, dtype=np.float32)

      # Pack the features into a tree structure.
      feature_structure = jax.tree.structure(feature_specs)
      features = jax.tree_util.tree_unflatten(feature_structure, [features])
      feature_weights = jax.tree_util.tree_unflatten(
          feature_structure, [feature_weights]
      )

      return embed.EmbeddingLookupInput(
          *embedding.preprocess_sparse_dense_matmul_input(
              features,
              feature_weights,
              feature_specs,
              mesh.local_mesh.size,
              mesh.size,
              num_sc_per_device=num_sc_per_device,
              sharding_strategy='MOD',
              batch_number=batch_number,
          )[0]
      )

    first_model_input = process_inputs(-1, feature_batches[0])
    params = model.init(jax.random.key(42), first_model_input)

    # Create optimizer.
    tx = embed_optimizer.create_optimizer_for_sc_model(
        params,
        optax.sgd(learning_rate=_LEARNING_RATE.value),
    )
    opt_state = tx.init(params)

    # Define the forward pass function.
    @functools.partial(
        jax.jit,
        out_shardings=(
            nn.meta.get_sharding(params, mesh),
            None,
            None,
        ),
        donate_argnums=(0),
    )
    def train_step(
        params: Any,
        embedding_lookup_inputs: embed.EmbeddingLookupInput,
        labels: jax.Array,
        opt_state,
    ):
      def forward_pass(params, embedding_lookups, labels):
        logits = model.apply(params, embedding_lookups)
        xentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        return jnp.mean(xentropy), logits

      # Run model forward/backward pass.
      train_step_fn = jax.value_and_grad(
          forward_pass, has_aux=True, allow_int=True
      )

      (loss_val, unused_logits), grads = train_step_fn(
          params, embedding_lookup_inputs, labels
      )

      updates, opt_state = tx.update(grads, opt_state)
      params = embed_optimizer.apply_updates_for_sc_model(params, updates)

      return params, opt_state, loss_val

    step = 0
    for features, labels in zip(feature_batches, label_batches):
      # ------------------------------------------------------------------------
      # Step 1: SC input processing.
      # ------------------------------------------------------------------------
      processed_input_tensor = process_inputs(step, features)

      # ------------------------------------------------------------------------
      # Step 2: run model.
      # ------------------------------------------------------------------------
      params, opt_state, loss_val = train_step(
          params, processed_input_tensor, labels, opt_state
      )

      logging.info(
          'Step %s: loss=%s, params is %s',
          step,
          loss_val,
          jax.tree.map(jnp.sum, params),
      )

      step += 1


if __name__ == '__main__':
  absltest.main()
