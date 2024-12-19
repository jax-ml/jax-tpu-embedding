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
import dataclasses
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
from jax.experimental.layout import DeviceLocalLayout as DLL
from jax.experimental.layout import Layout
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax
import orbax.checkpoint as ocp
import tree

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


_NUM_SCS_PER_DEVICE = 4

_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 1024, 'Vocabulary size.')

_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'Global batch size.'
)

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

_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency', 10, 'Frequency to log metrics.'
)
_LOSS_RESET_FREQUENCY = flags.DEFINE_integer(
    'loss_window', 10, 'Number of steps to average loss over.'
)

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
    rng: jax.Array,
    global_device_count: int,
    global_batch_size: int,
    vocab_size: int,
    seq_len: int,
    embedding_size: int,
) -> tuple[
    nn.Module,
    optax.GradientTransformation,
    TrainState,
    embedding.Nested[embedding_spec.FeatureSpec],
]:
  """Create and initialize the model.

  Args:
    rng: JAX PRNG Key.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    global_batch_size: global batch size.
    vocab_size: embedding vocabulary size.
    seq_len: sequence length.
    embedding_size: embedding dimension.

  Returns:
    The model, optimizer,  initial train state, table specs, and feature specs.
  """
  model = shakespeare_model.Model(
      global_batch_size=global_batch_size,
      vocab_size=vocab_size,
      seq_len=seq_len,
      embedding_size=embedding_size,
  )

  # Global embedding activations. We can change this to local arrays and use
  # make_array_from_single_device_arrays as above.
  init_emb_activations = {
      model.feature_name: jnp.zeros((
          global_batch_size,
          seq_len,
          1,
          embedding_size,
      ))
  }

  params = model.init(rng, init_emb_activations)
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.adam(learning_rate=_LEARNING_RATE.value)
  feature_specs = model.create_feature_specs()
  embedding.prepare_feature_specs_for_training(
      feature_specs, global_device_count=global_device_count
  )
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
  chkpt_metadata = chkpt_mgr.item_metadata(latest_step)

  # Provide a sample target to make sure Orbax able to restore the
  # custom TrainState object instead of raw PyTree
  restore_target = {
      'train_state': init_train_state,
      'emb_variables': chkpt_metadata['emb_variables'],
  }

  # restore from the latest steps
  restored = chkpt_mgr.restore(
      latest_step,
      args=ocp.args.PyTreeRestore(
          item=restore_target,
      ),
  )

  info(
      'Successfully restored from last checkpoint (latest_step = %s)',
      latest_step,
  )

  return latest_step, restored


def run_model():
  """Runs the model including input processing and training."""
  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  num_processes = jax.process_count()
  process_id = jax.process_index()
  info(
      'num devices: local = %s, global = %s',
      num_local_devices,
      num_global_devices,
  )
  info('process_id = %s, num_processes = %s', process_id, num_processes)
  pd = P('device')  # Device sharding.
  pe = P('device', None)  # PartitionSpec for embedding tables.
  local_devices = jax.local_devices()
  global_devices = jax.devices()
  info('local_devices [len=%s] = %s', len(local_devices), local_devices)
  info('global_devices [len=%s] = %s', len(global_devices), global_devices)
  global_mesh = Mesh(np.array(global_devices), axis_names=['device'])
  global_sharding = NamedSharding(global_mesh, pd)
  global_emb_sharding = NamedSharding(global_mesh, pe)

  chkpt_mgr = None

  # Note 1: InputProcessing is currently global so all the input batch features
  # and labels are global as well.
  #
  # Note 2: The input processor expects 2-d lookups, so we scale-up the batch
  # size and reshape the results.

  # Initialize the model.
  model, optimizer, train_state, feature_specs = create_train_state(
      jax.random.key(42),
      num_global_devices,
      _GLOBAL_BATCH_SIZE.value,
      _VOCAB_SIZE.value,
      _SEQ_LEN.value,
      _EMBEDDING_SIZE.value,
  )

  local_batch_size = _GLOBAL_BATCH_SIZE.value // num_processes
  device_batch_size = _GLOBAL_BATCH_SIZE.value // num_global_devices
  info(
      'batch sizes: global=%s, local=%s, device=%s',
      _GLOBAL_BATCH_SIZE.value,
      local_batch_size,
      device_batch_size,
  )

  per_sc_vocab_size = _VOCAB_SIZE.value // _NUM_SCS_PER_DEVICE
  if per_sc_vocab_size < 8 or per_sc_vocab_size % 8 != 0:
    raise ValueError(
        'Vocabulary size must be a multiple of 8 per SC: VOCAB_SIZE ='
        f' {_VOCAB_SIZE.value}, num_scs = {_NUM_SCS_PER_DEVICE}'
    )

  word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
  vlog1('word_ids len = %s', len(word_ids))
  feature_batches, label_batches = shakespeare_data.word_id_batches(
      word_ids,
      _NUM_STEPS.value,
      _GLOBAL_BATCH_SIZE.value,
      _SEQ_LEN.value,
      _NUM_TABLES.value,
  )
  feature_batches = feature_batches['words_0']
  vlog1('feature_batches len = %s', len(feature_batches))
  vlog1('feature_batches[0] shape = %s', feature_batches[0].shape)
  vlog1('label_batches len = %s', len(label_batches))
  vlog1('label_batches[0] shape = %s', label_batches[0].shape)

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
        item_handlers=ocp.PyTreeCheckpointHandler(),
    )

    if _CHECKPOINT_RESUME.value:
      latest_step, restored = _try_restore_latest_checkpoint(
          chkpt_mgr, train_state
      )
      if latest_step is not None:
        # restore successfully, use the restored train_state and embedding
        train_state = restored['train_state']
        emb_variables = restored['emb_variables']

  if emb_variables is None:
    table_specs = {
        f.table_spec.name: f.table_spec for f in tree.flatten(feature_specs)
    }
    emb_variables = embedding.init_embedding_variables(
        jax.random.key(13), table_specs, global_emb_sharding
    )
  emb_var_outsharding = Layout(
      DLL(
          major_to_minor=(0, 1),
          _tiling=((8,),),
      ),
      global_emb_sharding,
  )
  out_path = os.path.join(_FDO_DIR.value, 'fdo_dump')
  os.makedirs(out_path, exist_ok=True)
  logging.info('FDO storage path: %s', out_path)
  fdo_client = file_fdo_client.NPZFileFDOClient(out_path)
  @partial(
      utils.jit_with_dump,
      static_argnums=(0, 1, 2, 3),
      out_shardings=(
          None,
          None,
          emb_var_outsharding,
      ),
      donate_argnums=(9),
  )
  def train_step_fn(
      mesh: jax.sharding.Mesh,
      model: nn.Module,
      optimizer,
      feature_specs,
      train_state: TrainState,
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      emb_variables,
      labels,
  ) -> tuple[TrainState, TrainMetrics, Nested[jax.Array]]:
    """Performs a single training step at the chip level."""

    # Sparse forward pass - embedding lookup.
    with jax.named_scope('sc_forward_pass'):
      tpu_sparse_dense_matmul = partial(
          embedding.tpu_sparse_dense_matmul,
          global_device_count=num_global_devices,
          feature_specs=feature_specs,
          sharding_strategy='MOD',
      )
      tpu_sparse_dense_matmul = shard_map(
          f=tpu_sparse_dense_matmul,
          mesh=mesh,
          in_specs=(pd, pd, pd, pd, pe),
          out_specs=pd,
          check_rep=False,
      )
      emb_act = tpu_sparse_dense_matmul(
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
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
      tpu_sparse_dense_matmul_grad = shard_map(
          f=tpu_sparse_dense_matmul_grad,
          mesh=mesh,
          in_specs=(pd, pd, pd, pd, pd, pe),
          out_specs=pe,
          check_rep=False,
      )
      emb_variables = tpu_sparse_dense_matmul_grad(
          emb_grad,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
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

  train_metrics = None
  step = latest_step or -1
  for features, labels in zip(
      feature_batches[step + 1 :], label_batches[step + 1 :]
  ):
    step += 1

    vlog1('*' * 70)
    vlog1('* STEP = %s', step)
    vlog1('*' * 70)

    # ----------------------------------------------------------------------
    # SC input processing.
    # ----------------------------------------------------------------------
    # These are currently global batches so each task needs to offset into
    # the data for it's local slice.
    labels = labels[
        process_id * local_batch_size : (process_id + 1) * local_batch_size
    ]
    labels = jax.make_array_from_process_local_data(global_sharding, labels)

    # Each input preprocessing processes the current process's slice of the
    # global batch.
    features = features[
        process_id * local_batch_size : (process_id + 1) * local_batch_size
    ]
    features = np.reshape(features, (-1, 1))
    feature_weights = np.ones(features.shape, dtype=np.float32)

    # Pack the features into a tree structure.
    feature_structure = jax.tree.structure(feature_specs)
    features = jax.tree_util.tree_unflatten(feature_structure, [features])
    feature_weights = jax.tree_util.tree_unflatten(
        feature_structure, [feature_weights]
    )

    # Preprocess the inputs and build Jax global views of the data.
    make_global_view = lambda x: jax.tree.map(
        lambda y: jax.make_array_from_process_local_data(global_sharding, y),
        x,
    )
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        stats,
    ) = map(
        make_global_view,
        embedding.preprocess_sparse_dense_matmul_input(
            features,
            feature_weights,
            feature_specs,
            local_device_count=global_mesh.local_mesh.size,
            global_device_count=global_mesh.size,
            num_sc_per_device=_NUM_SCS_PER_DEVICE,
            sharding_strategy='MOD',
        ),
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
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        emb_variables,
        labels,
    )

    with jax.spmd_mode('allow_all'):
      train_metrics = (
          metrics_update
          if train_metrics is None
          else train_metrics.merge(metrics_update)
      )

    if (step + 1) % _LOG_FREQUENCY.value == 0:
      with jax.spmd_mode('allow_all'):
        m = train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(train_state.params)
      fdo_client.publish()

    if (step + 1) % _LOSS_RESET_FREQUENCY.value == 0:
      train_metrics = None
      max_ids_per_partition, max_unique_ids_per_partition = fdo_client.load()

      def update_ids(f, max_ids_per_partition, max_unique_ids_per_partition):
        stack_name = f.table_spec.stacked_table_spec.stack_name
        logging.info('Maybe updating limits for table: %s', stack_name)
        max_id = max_ids_per_partition.get(
            stack_name,
            f.table_spec.stacked_table_spec.max_ids_per_partition,
        )
        max_unique_id = max_unique_ids_per_partition.get(
            stack_name,
            f.table_spec.stacked_table_spec.max_unique_ids_per_partition,
        )
        # Just use the new values. For shakespeare, the new values are
        # smaller than the old values and then they donot change at all
        # This will trigger recompilation once after the 10th step.
        logging.info(
            'Updating limits for table %s. Prev: (%s, %s) -> New: (%s, %s)',
            stack_name,
            f.table_spec.stacked_table_spec.max_ids_per_partition,
            f.table_spec.stacked_table_spec.max_unique_ids_per_partition,
            np.max(max_id),
            np.max(max_unique_id),
        )
        new_stacked_spec = dataclasses.replace(
            f.table_spec.stacked_table_spec,
            max_ids_per_partition=int(np.max(max_id)),
            max_unique_ids_per_partition=int(np.max(max_unique_id)),
        )
        new_table_spec = dataclasses.replace(
            f.table_spec, stacked_table_spec=new_stacked_spec
        )
        return dataclasses.replace(f, table_spec=new_table_spec)

      maybe_update_limits = partial(
          update_ids,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )
      feature_specs = jax.tree_util.tree_map(
          maybe_update_limits,
          feature_specs,
      )

    if chkpt_mgr:
      chkpt_mgr.save(
          step,
          args=ocp.args.PyTreeSave({
              'train_state': train_state,
              'emb_variables': emb_variables,
              'stacking_proto': embedding.create_proto_from_feature_specs(
                  feature_specs, global_device_count=num_global_devices
              ),
          }),
      )

  if chkpt_mgr:
    if chkpt_mgr.latest_step() != step:
      # Make sure the latest step is saved before exiting.
      chkpt_mgr.save(  # pytype: disable=attribute-error
          step,
          args=ocp.args.PyTreeSave({
              'train_state': train_state,
              'emb_variables': emb_variables,
              'stacking_proto': embedding.create_proto_from_feature_specs(
                  feature_specs, global_device_count=num_global_devices
              ),
          }),
          force=True,
      )

    # close the checkpoint manager and wait for background save or deletion
    chkpt_mgr.close()


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_model()


if __name__ == '__main__':
  app.run(main)