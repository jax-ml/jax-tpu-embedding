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
import jax.numpy as jnp
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.nn import embedding
import numpy as np
import optax
import orbax.checkpoint as ocp


np.set_printoptions(threshold=np.inf)
FLAGS = flags.FLAGS


class ShakespeareTest(absltest.TestCase):
  """Input processing and convergence of basic model using JAX SC libraries."""

  def test_shakespeare_model_loss_convergence(self):
    FLAGS.vocab_size = 128
    FLAGS.batch_size = 8
    FLAGS.num_steps = 1000
    config = shakespeare_config.get_config()
    chkpt_dir = os.path.join(FLAGS.test_tmpdir, 'shakespeare_test')
    chkpt_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(chkpt_dir),
        item_names=('params', 'opt_state', 'embedding'),
    )

    mesh = jax.sharding.Mesh(config.global_devices, config.sharding_axis)

    feature_batches, label_batches = shakespeare_config.get_batches(config)

    # Construct the model.
    model = shakespeare_model.Model(
        global_batch_size=config.global_batch_size,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        embedding_size=config.embedding_size,
        feature_name=config.feature_name,
    )

    # Initialize the model.
    init_emb_activations = {
        config.feature_name: jnp.zeros(
            (config.global_batch_size, config.seq_len, 1, config.embedding_size)
        )
    }
    params = model.init(jax.random.key(42), init_emb_activations)
    feature_specs = shakespeare_config.create_feature_specs(config)

    # Create optimizer.
    tx = optax.adam(learning_rate=config.learning_rate)
    opt_state = tx.init(params)

    # Prepare embedding tables.
    emb_table = np.zeros([config.vocab_size, config.embedding_size])
    emb_table_sharded = einops.rearrange(
        emb_table,
        '(v c s) f -> c (s v) f',
        c=config.num_global_devices,
        s=config.num_sc_per_device,
    )

    embedding_variables = {}
    embedding_variables[config.table_name] = [
        jax.device_put(
            emb_table_sharded[i],
            device=device,
        )
        for i, device in enumerate(config.global_devices)
    ]
    sharding = NamedSharding(mesh, P(config.sharding_axis, None))
    table_arr = jax.make_array_from_single_device_arrays(
        shape=(config.vocab_size, config.embedding_size),
        sharding=sharding,
        arrays=embedding_variables[config.table_name],
    )
    embedding_variables[config.table_name] = embedding.EmbeddingVariables(
        table=table_arr,
        slot=(jnp.zeros_like(table_arr), jnp.zeros_like(table_arr)),
    )

    # Define the forward pass function.
    loss_grad_fn = jax.value_and_grad(
        functools.partial(shakespeare_model.loss, model),
        argnums=(
            0,
            1,
        ),
        has_aux=True,
    )
    sharded_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        global_device_count=config.num_global_devices,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    sparse_matmul = jax.shard_map(
        sharded_matmul,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(config.sharding_axis),
        check_vma=False,
    )
    sparse_matmul = jax.jit(sparse_matmul)

    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    sparse_grad_update = jax.shard_map(
        sharded_grad_update,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(config.sharding_axis, None),
        check_vma=False,
    )
    sparse_grad_update = jax.jit(sparse_grad_update)
    step = 0
    losses = []
    for features, labels in zip(feature_batches, label_batches):
      # --------------------------------------------------------------------------
      # Step 1: SC input processing.
      # --------------------------------------------------------------------------
      features = shakespeare_config.local_slice(config, features)
      labels = shakespeare_config.local_slice(config, labels)
      preprocessed_inputs, _ = shakespeare_config.process_inputs(
          config, feature_specs, step, features, data_sharding=None
      )

      # --------------------------------------------------------------------------
      # Step 2: SC forward pass.
      # --------------------------------------------------------------------------
      activations = sparse_matmul(preprocessed_inputs, embedding_variables)
      # Activations returned by matmul have the same structure as input feature
      # specs which is `frozendict` in this case. Make it a regular dict.
      activations = dict(activations)
      activations = jax.tree_util.tree_map(
          lambda x: jnp.reshape(x, (config.global_batch_size, -1)), activations
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
          config.feature_name: jnp.reshape(
              grads[1][config.feature_name], (-1, config.embedding_size)
          )
      }
      embedding_variables = sparse_grad_update(
          gradient_updates,
          preprocessed_inputs,
          embedding_variables,
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
                    {config.table_name: embedding_variables[config.table_name]}
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
