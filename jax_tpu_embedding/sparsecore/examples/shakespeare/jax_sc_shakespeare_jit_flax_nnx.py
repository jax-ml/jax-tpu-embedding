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
import os

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import flax_nnx_model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.fdo import csv_file_fdo_client
from jax_tpu_embedding.sparsecore.lib.flax.nnx import checkpoint_utils
from jax_tpu_embedding.sparsecore.lib.flax.nnx import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
import numpy as np
import optax
import orbax.checkpoint as ocp

Nested = embedding.Nested


_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir',
    None,
    'If set, checkpoints will be written to this directory.',
)
_INPUT_CHECKPOINT_DIR = flags.DEFINE_string(
    'input_checkpoint_dir',
    None,
    'If set, a checkpoint will be restored from this directory.',
)
_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'checkpoint_interval', 100, 'Number of steps per checkpoint'
)
_CHECKPOINT_MAX_TO_KEEP = flags.DEFINE_integer(
    'checkpoint_max_to_keep',
    2,
    'Number of checkpoints to keep.',
)
_CROSS_TOPOLOGY_CHECKPOINT_RESTORE = flags.DEFINE_string(
    'cross_topology_checkpoint_restore',
    None,
    'If set, a checkpoint restore will be attempted from the checkpoint in'
    ' this directory, across different topology.',
)
_FDO_DIR = flags.DEFINE_string(
    'fdo_dir',
    None,
    'If set, FDO stats will be written to this directory.',
)


def expand_directory_path(input_dir: str) -> str | None:
  """Expands the input directory if it refers to a test output directory.


  Args:
    input_dir: The input directory to expand.

  Returns:
    The expanded directory path.

  Raises:
    ValueError: If the TEST_UNDECLARED_OUTPUTS_DIR environment variable is not
      set when a relative path is provided.

  If the input directory is a relative path, it is expanded to an absolute path.
  If the input directory starts with TEST_UNDECLARED_OUTPUTS_DIR, it is
  expanded to the corresponding path in the test output directory, which is
  provided by the environment variable 'TEST_UNDECLARED_OUTPUTS_DIR'.
  """
  testdir_key = 'TEST_UNDECLARED_OUTPUTS_DIR'
  testdir_env_var = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')

  if input_dir is None:
    return None

  # Leave absolute paths alone.
  if input_dir.startswith('/'):
    return input_dir

  # Special handling for test output subdirectories.
  if input_dir.startswith(testdir_key + '/'):
    if not testdir_env_var:
      raise ValueError(f'{testdir_key} environment variable is not set.')
    expanded_dir = os.path.join(
        testdir_env_var, input_dir[len(testdir_key + '/') :]
    )
    os.makedirs(expanded_dir, exist_ok=True)
    return expanded_dir

  # Special handling for writing to the test output directory.
  if input_dir == testdir_key:
    if not testdir_env_var:
      raise ValueError(f'{testdir_key} environment variable is not set.')
    return testdir_env_var

  return os.path.abspath(input_dir)


def create_fdo_client() -> csv_file_fdo_client.CSVFileFDOClient | None:
  """Creates an FDO client if a directory is provided."""
  if _FDO_DIR.value is None:
    logging.info('No FDO enabled.')
    return None

  fdo_path = expand_directory_path(_FDO_DIR.value)
  logging.info('FDO storage path: %s', fdo_path)
  return csv_file_fdo_client.CSVFileFDOClient(fdo_path, retain_history=False)  # pyrefly: ignore[bad-argument-type]


def create_checkpoint_manager() -> ocp.CheckpointManager | None:
  """Creates a checkpoint manager if a directory is provided."""
  if _CHECKPOINT_DIR.value is None:
    logging.info('No checkpointing enabled.')
    return None
  cp_options = ocp.CheckpointManagerOptions(
      max_to_keep=_CHECKPOINT_MAX_TO_KEEP.value,
      save_interval_steps=_CHECKPOINT_INTERVAL.value,
  )
  cp_path = expand_directory_path(_CHECKPOINT_DIR.value)
  logging.info('Checkpoint path: %s', cp_path)
  return checkpoint_utils.create_checkpoint_manager(
      cp_path=cp_path,  # pyrefly: ignore[bad-argument-type]
      cp_options=cp_options,
  )


def run_model() -> None:
  """Runs the model including input processing and training."""

  if (
      _INPUT_CHECKPOINT_DIR.value is not None
      and _CROSS_TOPOLOGY_CHECKPOINT_RESTORE.value is not None
  ):
    raise ValueError(
        'Only one of --input_checkpoint or --cross_topology_checkpoint_restore'
        ' can be set.'
    )

  config = shakespeare_config.get_config()

  mesh = Mesh(
      np.array(config.global_devices), axis_names=[config.sharding_axis]
  )
  data_sharding = NamedSharding(mesh, PartitionSpec(config.sharding_axis))

  feature_specs = shakespeare_config.create_feature_specs(config)
  feature_batches, label_batches = shakespeare_config.get_batches(config)

  cp_manager = create_checkpoint_manager()
  fdo_client = create_fdo_client()

  model = shakespeare_model.Model(
      feature_specs=feature_specs,
      global_batch_size=config.global_batch_size,
      vocab_size=config.vocab_size,
      seq_len=config.seq_len,
      embedding_size=config.embedding_size,
      enable_minibatching=config.enable_minibatching,
      mesh=mesh,
      feature_name=config.feature_name,
      sharding_axis=config.sharding_axis,
  )

  dense_tx = optax.adam(learning_rate=config.learning_rate)
  optimizer = embed.PartitionedOptimizer(model, dense_tx)

  if _CROSS_TOPOLOGY_CHECKPOINT_RESTORE.value is not None:
    restore_path = expand_directory_path(
        _CROSS_TOPOLOGY_CHECKPOINT_RESTORE.value
    )
    if restore_path is not None and restore_path.endswith('.tgz'):
      restore_path = checkpoint_utils.decompress_checkpoint(restore_path)
    checkpoint_utils.restore_cross_topology_checkpoint(
        cross_topology_checkpoint_restore_path=restore_path,
        model=model,
        optimizer=optimizer,
        target_feature_specs=feature_specs,
        num_global_devices=config.num_global_devices,
        num_sc_per_device=config.num_sc_per_device,
    )
  elif _INPUT_CHECKPOINT_DIR.value is not None:
    input_path = expand_directory_path(_INPUT_CHECKPOINT_DIR.value)
    if input_path is not None and input_path.endswith('.tgz'):
      input_path = checkpoint_utils.decompress_checkpoint(input_path)
    checkpoint_utils.restore_checkpoint(
        input_checkpoint_path=input_path,
        step=None,  # Restore from the latest step.
        model=model,
        optimizer=optimizer,
    )

  model_sharding = embed.get_named_sharding(model, mesh)
  optimizer_sharding = embed.get_named_sharding(optimizer, mesh)

  @nnx.jit(
      in_shardings=(
          model_sharding,
          optimizer_sharding,
          data_sharding,
          data_sharding,
          None,  # step
      ),
      donate_argnames=[
          'model',
          'optimizer',
      ],
  )
  def train_step(
      model: nnx.Module,
      optimizer: nnx.Optimizer,
      embedding_lookup_inputs: embedding.PreprocessedInput,
      labels: jax.Array,
      step: jax.Array,
  ) -> jax.Array:
    """Executes a single training step for the model.

    Args:
      model: The Flax NNX model module to be trained.
      optimizer: The Flax NNX optimizer used to update the model parameters.
      embedding_lookup_inputs: Preprocessed embedding inputs containing the
        sparse core embedding lookups.
      labels: Target integer labels for the batch.
      step: The current training step.

    Returns:
      The scalar loss value for the training step.
    """

    def loss_fn(
        mdl: nnx.Module,
        inputs: embedding.PreprocessedInput,
        labels: jax.Array,
        step: jax.Array,
    ):
      logits = mdl(inputs, step=step)
      xentropy = optax.softmax_cross_entropy_with_integer_labels(
          logits=logits, labels=labels
      )
      return jnp.mean(xentropy), logits

    train_step_fn = nnx.value_and_grad(loss_fn, has_aux=True, allow_int=True)

    (loss_val, unused_logits), grads = train_step_fn(
        model, embedding_lookup_inputs, labels, step
    )

    optimizer.update(model, grads)

    return loss_val

  def training_loop(
      model: nnx.Module, start_step: int, feature_batches, label_batches
  ):
    step = start_step
    losses = []
    for features, labels in zip(feature_batches, label_batches):
      shakespeare_config.step_header(step)

      # These are currently global batches so each task needs to offset into
      # the data for it's local slice.
      labels = shakespeare_config.device_slice(config, labels, data_sharding)

      # Each input preprocessing processes the current process's slice of the
      # global batch.
      features = shakespeare_config.local_slice(config, features)

      model_inputs, stats = shakespeare_config.process_inputs(
          config,
          feature_specs,
          step,
          features,
          data_sharding,
      )

      if fdo_client is not None:
        fdo_client.record(stats)

      loss_val = train_step(
          model,
          optimizer,
          model_inputs,
          labels,
          jnp.array(step, dtype=jnp.int32),
      )
      losses.append(loss_val)

      if step % config.log_frequency == 0:
        logging.info('*** Step %s: Loss = %s', step, loss_val)

      if fdo_client is not None and (step + 1) % config.log_frequency == 0:
        fdo_client.publish()

      if (
          fdo_client is not None
          and (step + 1) % config.loss_reset_frequency == 0
      ):
        loaded_stats = fdo_client.load()
        jax.experimental.multihost_utils.sync_global_devices('FDO_load_barrier')
        embedding.update_preprocessing_parameters(
            feature_specs, loaded_stats, config.num_sc_per_device
        )

      step += 1
      if cp_manager is not None:
        checkpoint_utils.save_checkpoint(
            cp_manager=cp_manager,
            step=step,
            model=model,
            optimizer=optimizer,
            feature_specs=feature_specs,
            num_global_devices=config.num_global_devices,
            num_sc_per_device=config.num_sc_per_device,
        )
    if cp_manager is not None:
      cp_manager.wait_until_finished()
    return losses

  logging.info('=' * 80)
  logging.info('==')
  logging.info('== Training loop with %s steps', len(feature_batches))
  logging.info('==')
  logging.info('=' * 80)
  losses = training_loop(model, 0, feature_batches, label_batches)

  logging.info('=' * 80)
  logging.info('==')
  logging.info('== Final Losses: %s', losses[-1])
  logging.info('==')
  logging.info('=' * 80)


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_model()


if __name__ == '__main__':
  app.run(main)
