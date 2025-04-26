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
import os

from absl import flags
from absl import logging
from absl.testing import absltest
import einops
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax
import orbax.checkpoint as ocp


FLAGS = flags.FLAGS

np.set_printoptions(threshold=np.inf)

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

_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    1,
    'Number of separate epochs.',
)

_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 8, 'Embedding size.')
_EMBEDDING_INIT = flags.DEFINE_enum(
    'embedding_init', 'normal', ['normal', 'row_id'], 'Embedding initializer.'
)


class ShakespeareTest(absltest.TestCase):
  """Input processing and convergence of basic model using JAX SC libraries."""

  def test_shakespeare_model_loss_convergence(self):
    chkpt_dir = os.path.join(FLAGS.test_tmpdir, 'shakespeare_test')
    chkpt_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(chkpt_dir),
        item_names=('params', 'opt_state', 'embedding'),
    )

    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(devices, 'x')
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])

    word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
    feature_batches, label_batches = shakespeare_data.word_id_batches(
        word_ids,
        _NUM_STEPS.value,
        _BATCH_SIZE.value,
        _SEQ_LEN.value,
        _NUM_TABLES.value,
    )
    feature_batches = feature_batches['words_0']

    # Construct the model.
    model = shakespeare_model.Model(
        global_batch_size=_BATCH_SIZE.value,
        vocab_size=_VOCAB_SIZE.value,
        seq_len=_SEQ_LEN.value,
        embedding_size=_EMBEDDING_SIZE.value,
    )

    # Initialize the model.
    init_emb_activations = {
        model.feature_name: jnp.zeros(
            (_BATCH_SIZE.value, _SEQ_LEN.value, 1, _EMBEDDING_SIZE.value)
        )
    }
    params = model.init(jax.random.key(42), init_emb_activations)
    feature_specs = model.create_feature_specs()

    # Create optimizer.
    tx = optax.adam(learning_rate=_LEARNING_RATE.value)
    opt_state = tx.init(params)

    # Prepare embedding tables.
    emb_table = np.zeros([_VOCAB_SIZE.value, _EMBEDDING_SIZE.value])
    emb_table_sharded = einops.rearrange(
        emb_table,
        '(v c s) f -> c (s v) f',
        c=len(devices),
        s=num_sc_per_device,
    )

    embedding_variables = {}
    embedding_variables[model.table_name] = [
        jax.device_put(
            emb_table_sharded[i],
            device=device,
        )
        for i, device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, P('x', None))
    embedding_variables[model.table_name] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOCAB_SIZE.value, _EMBEDDING_SIZE.value),
            sharding=sharding,
            arrays=embedding_variables[model.table_name],
        )
    ])

    # Define the forward pass function.
    loss_grad_fn = jax.value_and_grad(
        functools.partial(shakespeare_model.loss, model),
        argnums=(
            0,
            1,
        ),
        has_aux=True,
    )
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=mesh.size,
        num_sc_per_device=num_sc_per_device,
    )
    config = embedding.SparseDenseMatmulConfig(
        feature_specs=feature_specs,
        global_device_count=mesh.size,
        num_sc_per_device=num_sc_per_device,
        local_device_count=mesh.local_mesh.size,
        sharding_strategy='MOD',
    )
    sharded_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        config=config,
    )
    sparse_matmul = shard_map.shard_map(
        sharded_matmul,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(mesh.axis_names[0]),
        check_rep=False,
    )
    sparse_matmul = jax.jit(sparse_matmul)

    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        config=config,
    )
    sparse_grad_update = shard_map.shard_map(
        sharded_grad_update,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(mesh.axis_names[0], None),
        check_rep=False,
    )
    sparse_grad_update = jax.jit(sparse_grad_update)
    step = 0
    losses = []
    for features, labels in zip(feature_batches, label_batches):
      # --------------------------------------------------------------------------
      # Step 1: SC input processing.
      # --------------------------------------------------------------------------
      features = np.reshape(features, (-1, 1))
      feature_weights = np.ones(features.shape, dtype=np.float32)
      # Pack the features into a tree structure.
      feature_structure = jax.tree.structure(feature_specs)
      features = jax.tree_util.tree_unflatten(feature_structure, [features])
      feature_weights = jax.tree_util.tree_unflatten(
          feature_structure, [feature_weights]
      )

      preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
          features, feature_weights, config
      )

      # --------------------------------------------------------------------------
      # Step 2: SC forward pass.
      # --------------------------------------------------------------------------
      activations = sparse_matmul(preprocessed_inputs, embedding_variables)
      # Activations returned by matmul have the same structure as input feature
      # specs which is `frozendict` in this case. Make it a regular dict.
      activations = dict(activations)
      activations = jax.tree_util.tree_map(
          lambda x: jnp.reshape(x, (_BATCH_SIZE.value, -1)), activations
      )
      # ------------------------------------------------------------------------
      # Step 3: TC forward/backward pass.
      # ------------------------------------------------------------------------
      (loss_val, unused_logits), grads = loss_grad_fn(
          params, activations, labels
      )

      updates, opt_state = tx.update(grads[0], opt_state)
      params = optax.apply_updates(params, updates)

      # ------------------------------------------------------------------------
      # Step 4: SC backward pass.
      # ------------------------------------------------------------------------

      gradient_updates = {
          model.feature_name: jnp.reshape(
              grads[1][model.feature_name], (-1, model.embedding_size)
          )
      }
      embedding_variables = sparse_grad_update(
          gradient_updates, preprocessed_inputs, embedding_variables
      )

      if step % 10 == 0:
        logging.info('Loss step %s: %s', step, loss_val)
        losses.append(loss_val)
        chkpt_manager.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(params),
                opt_state=ocp.args.StandardSave(opt_state),
                embedding=ocp.args.StandardSave(
                    {'shakespeare_table': embedding_variables[model.table_name]}
                ),
            ),
        )
        chkpt_manager.wait_until_finished()
      step += 1

    # We want the last loss (at the last recorded step) to be less than 0.001.
    self.assertLess(losses[-1], 0.001)
    restored = chkpt_manager.restore(80)
    logging.info('Restored opt_state: %s', restored.get('opt_state'))
    logging.info('Restored params: %s', restored.get('params'))
    logging.info('Restored embedding: %s', restored.get('embedding'))


if __name__ == '__main__':
  absltest.main()
