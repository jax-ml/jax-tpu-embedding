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
"""An example Shakespeare model that uses the SparseCore embedding API."""

# pylint: disable=g-importing-member
import collections
from functools import partial
import os
from typing import Any, Mapping, Optional

from absl import app
from absl import flags
from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax
import orbax.checkpoint as ocp


np.set_printoptions(threshold=np.inf)
Nested = embedding.Nested


@flax.struct.dataclass
class TrainState:
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """

  params: Any
  opt_state: optax.OptState


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  # train_accuracy: metrics.Accuracy
  # learning_rate: metrics.LastValue.from_output("learning_rate")
  train_loss: metrics.Average.from_output('loss')
  train_loss_std: metrics.Std.from_output('loss')


_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'If set, checkpoints will be written to the directory.',
)

_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval', 500, 'Number of steps per checkpoint'
)

_CHECKPOINT_RESUME = flags.DEFINE_bool(
    'checkpoint_resume',
    True,
    'If set True and checkpoint_dir is specified, try to resume the training'
    ' from the latest checkpoint available in the checkpoint_dir.',
)

_CHECKPOINT_MAX_TO_KEEP = flags.DEFINE_integer(
    'checkpoint_max_to_keep',
    5,
    'Number of checkpoints to keep.',
)

_FDO_DIR = flags.DEFINE_string(
    'fdo_dir',
    '/tmp',
    'If set, FDO dumps will be written to the directory.',
)

info = logging.info
vlog1 = partial(logging.vlog, 1)


def create_train_state(
    config: shakespeare_config.Config, rng: jax.Array
) -> tuple[
    nn.Module,
    optax.GradientTransformation,
    TrainState,
    embedding.Nested[embedding_spec.FeatureSpec],
]:
  """Create and initialize the model.

  Args:
    config: The configuration.
    rng: JAX PRNG Key.

  Returns:
    The model, optimizer,  initial train state, table specs, and feature specs.
  """
  model = shakespeare_model.Model(
      global_batch_size=config.global_batch_size,
      vocab_size=config.vocab_size,
      seq_len=config.seq_len,
      embedding_size=config.embedding_size,
      feature_name=config.feature_name,
  )

  # Global embedding activations. We can change this to local arrays and use
  # make_array_from_single_device_arrays as above.
  init_emb_activations = {
      config.feature_name: jnp.zeros((
          config.global_batch_size,
          config.seq_len,
          1,
          config.embedding_size,
      ))
  }

  params = model.init(rng, init_emb_activations)
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.adam(learning_rate=config.learning_rate)
  feature_specs = shakespeare_config.create_feature_specs(config)

  return (
      model,
      optimizer,
      TrainState(
          params=params,
          opt_state=optimizer.init(params),
      ),
      feature_specs,
  )


def _try_restore_latest_checkpoint(
    chkpt_mgr: ocp.CheckpointManager, init_train_state: TrainState
) -> tuple[Optional[int], Optional[Mapping[str, Any]]]:
  """Try restoring the latest checkpoint from the checkpoint directory."""
  info(
      'Try restoring latest checkpoint from %s',
      chkpt_mgr.directory,
  )
  latest_step = chkpt_mgr.latest_step()
  if latest_step is None:
    info('No previous steps found in checkpoint.')
    return None, None

  info('Found latest_step = %s', latest_step)
  # load and resume from the latest step
  # Note that the following metadata need to be passed in during restore only
  # when the topology changes between saved checkpoint and restore.
  # chkpt_metadata = chkpt_mgr.item_metadata(latest_step)

  # Provide a sample target to make sure Orbax able to restore the
  # custom TrainState object instead of raw PyTree
  restore_target = {
      'train_state': ocp.args.PyTreeRestore(init_train_state),
      'emb_variables': ocp.args.PyTreeRestore(),
      'stacking_proto': ocp.args.ProtoRestore(
          embedding_spec_pb2.EmbeddingSpecProto
      ),
  }

  # restore from the latest steps
  restored = chkpt_mgr.restore(
      latest_step,
      args=ocp.args.Composite(**restore_target),
  )

  info(
      'Successfully restored from last checkpoint (latest_step = %s)',
      latest_step,
  )

  return latest_step, restored


def run_model():
  """Runs the model including input processing and training."""
  config = shakespeare_config.get_config()

  pd = P(config.sharding_axis)  # Device sharding.
  pe = P(config.sharding_axis, None)  # PartitionSpec for embedding tables.

  global_mesh = Mesh(
      np.array(config.global_devices), axis_names=[config.sharding_axis]
  )
  data_sharding = NamedSharding(global_mesh, pd)
  embedding_sharding = NamedSharding(global_mesh, pe)

  chkpt_mgr = None

  # Note 1: InputProcessing is currently global so all the input batch features
  # and labels are global as well.
  #
  # Note 2: The input processor expects 2-d lookups, so we scale-up the batch
  # size and reshape the results.

  # Initialize the model.
  model, optimizer, train_state, feature_specs = create_train_state(
      config, jax.random.key(42)
  )

  feature_batches, label_batches = shakespeare_config.get_batches(config)

  emb_variables = None
  latest_step = None
  if _CHECKPOINT_DIR.value:
    chkpt_mgr = ocp.CheckpointManager(
        directory=_CHECKPOINT_DIR.value,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=_CHECKPOINT_MAX_TO_KEEP.value,
            enable_background_delete=True,
            save_interval_steps=_CHECKPOINT_INTERVAL.value,
        ),
    )

    if _CHECKPOINT_RESUME.value:
      latest_step, restored = _try_restore_latest_checkpoint(
          chkpt_mgr, train_state
      )
      if latest_step is not None:
        # restore successfully, use the restored train_state and embedding
        train_state = restored['train_state']
        emb_variables = {}
        for k, v in restored['emb_variables'].items():
          emb_variables[k] = embedding.EmbeddingVariables(
              table=v['table'],
              slot=v['slot'],
          )

  if emb_variables is None:
    table_specs = {
        f.table_spec.name: f.table_spec for f in jax.tree.leaves(feature_specs)
    }
    emb_variables = embedding.init_embedding_variables(
        jax.random.key(13),
        table_specs,
        embedding_sharding,
        config.num_sc_per_device,
    )
  emb_var_outsharding = utils.embedding_table_format(
      embedding_sharding.mesh, embedding_sharding.spec
  )

  @partial(
      jax.jit,
      static_argnums=(0, 1, 2, 3),
      out_shardings=(
          None,
          None,
          emb_var_outsharding,
      ),
      donate_argnums=(6),
  )
  def train_step_fn(
      mesh: jax.sharding.Mesh,
      model: nn.Module,
      optimizer,
      feature_specs,
      train_state: TrainState,
      preprocessed_inputs,
      emb_variables,
      labels,
  ) -> tuple[TrainState, TrainMetrics, Nested[jax.Array]]:
    """Performs a single training step at the chip level."""

    # Sparse forward pass - embedding lookup.
    with jax.named_scope('sc_forward_pass'):
      tpu_sparse_dense_matmul = partial(
          embedding.tpu_sparse_dense_matmul,
          global_device_count=config.num_global_devices,
          feature_specs=feature_specs,
          sharding_strategy='MOD',
      )
      tpu_sparse_dense_matmul = jax.shard_map(
          tpu_sparse_dense_matmul,
          mesh=mesh,
          in_specs=(pd, pe),
          out_specs=pd,
          check_vma=False,
      )
      emb_act = tpu_sparse_dense_matmul(
          preprocessed_inputs,
          emb_variables,
      )

    # Dense forward + backward pass.
    emb_act = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (model.global_batch_size, -1)), emb_act
    )
    with jax.named_scope('tc_pass'):
      loss_grad_fn = jax.value_and_grad(
          partial(shakespeare_model.loss, model), argnums=(0, 1), has_aux=True
      )

      (loss, logits), (dense_grad, emb_grad) = loss_grad_fn(
          train_state.params, emb_act, labels
      )

    with jax.named_scope('tc_update'):
      updates, new_opt_state = optimizer.update(
          dense_grad, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)

    emb_grad = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, model.embedding_size)), emb_grad
    )

    # Sparse backward pass - embedding update.
    with jax.named_scope('sc_backward_pass'):
      tpu_sparse_dense_matmul_grad = partial(
          embedding.tpu_sparse_dense_matmul_grad,
          feature_specs=feature_specs,
          sharding_strategy='MOD',
      )
      tpu_sparse_dense_matmul_grad = jax.shard_map(
          tpu_sparse_dense_matmul_grad,
          mesh=mesh,
          in_specs=(pd, pd, pe),
          out_specs=pe,
          check_vma=False,
      )
      emb_variables = tpu_sparse_dense_matmul_grad(
          emb_grad,
          preprocessed_inputs,
          emb_variables,
      )

    train_state = train_state.replace(
        params=new_params, opt_state=new_opt_state
    )

    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss,
        logits=logits,
        labels=labels,
    )

    return train_state, metrics_update, emb_variables

  # Distributed training.
  parameter_overview.log_parameter_overview(train_state.params)

  out_path = os.path.join(_FDO_DIR.value, 'fdo_dump')
  os.makedirs(out_path, exist_ok=True)
  logging.info('FDO storage path: %s', out_path)
  fdo_client = file_fdo_client.NPZFileFDOClient(out_path)

  train_metrics = None
  step = latest_step or -1
  for features, labels in zip(
      feature_batches[step + 1 :], label_batches[step + 1 :]
  ):
    step += 1

    shakespeare_config.step_header(step)

    # ----------------------------------------------------------------------
    # SC input processing.
    # ----------------------------------------------------------------------
    # These are currently global batches so each task needs to offset into
    # the data for it's local slice.
    labels = shakespeare_config.device_slice(config, labels, data_sharding)

    # Each input preprocessing processes the current process's slice of the
    # global batch.
    features = shakespeare_config.local_slice(config, features)
    preprocessed_inputs, stats = shakespeare_config.process_inputs(
        config,
        feature_specs,
        step,
        features,
        data_sharding,
    )

    fdo_client.record(stats)

    # ----------------------------------------------------------------------
    # Combined: SC forward, TC, SC backward
    # ----------------------------------------------------------------------
    train_state, metrics_update, emb_variables = train_step_fn(
        global_mesh,
        model,
        optimizer,
        feature_specs,
        train_state,
        preprocessed_inputs,
        emb_variables,
        labels,
    )

    train_metrics = (
        metrics_update
        if train_metrics is None
        else train_metrics.merge(metrics_update)
    )

    if (step + 1) % config.log_frequency == 0:
      m = train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(train_state.params)
      fdo_client.publish()

    if (step + 1) % config.loss_reset_frequency == 0:
      train_metrics = None
      loaded_stats = fdo_client.load()
      jax.experimental.multihost_utils.sync_global_devices('FDO_load_barrier')

      embedding.update_preprocessing_parameters(
          feature_specs, loaded_stats, config.num_sc_per_device
      )
    if chkpt_mgr:
      chkpt_mgr.save(
          step,
          args=ocp.args.Composite(**{
              'train_state': ocp.args.PyTreeSave(train_state),
              'emb_variables': ocp.args.PyTreeSave(emb_variables),
              'stacking_proto': ocp.args.ProtoSave(
                  embedding.create_proto_from_feature_specs(
                      feature_specs,
                      global_device_count=config.num_global_devices,
                      num_sparsecore_per_device=config.num_sc_per_device,
                  )
              ),
          }),
      )

  if chkpt_mgr:
    if chkpt_mgr.latest_step() != step:
      # Make sure the latest step is saved before exiting.
      chkpt_mgr.save(  # pytype: disable=attribute-error
          step,
          args=ocp.args.Composite(**{
              'train_state': ocp.args.PyTreeSave(train_state),
              'emb_variables': ocp.args.PyTreeSave(emb_variables),
              'stacking_proto': ocp.args.ProtoSave(
                  embedding.create_proto_from_feature_specs(
                      feature_specs,
                      global_device_count=config.num_global_devices,
                      num_sparsecore_per_device=config.num_sc_per_device,
                  )
              ),
          }),
          force=True,
      )

    # Close the checkpoint manager and wait for background save or deletion.
    chkpt_mgr.close()


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_model()


if __name__ == '__main__':
  app.run(main)
