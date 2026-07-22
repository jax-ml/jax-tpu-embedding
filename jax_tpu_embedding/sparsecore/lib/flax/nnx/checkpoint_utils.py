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
"""Checkpointing utilities for SparseCore NNX models."""

import collections.abc
import os
import shutil
import time
from typing import Any

from absl import logging
from flax import nnx
import jax
from jax_tpu_embedding.sparsecore.lib.flax.nnx import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import orbax.checkpoint as ocp

Nested = embedding.Nested


def decompress_checkpoint(checkpoint_path: str) -> str:
  """Decompresses a checkpoint archive (.tgz) to TEST_UNDECLARED_OUTPUTS_DIR.

  Args:
    checkpoint_path: The path to the compressed checkpoint archive.

  Returns:
    The path to the decompressed checkpoint directory.

  Raises:
    ValueError: If TEST_UNDECLARED_OUTPUTS_DIR is not set.
  """
  testdir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
  if not testdir:
    raise ValueError(
        'TEST_UNDECLARED_OUTPUTS_DIR environment variable is not set.'
    )

  archive_name = os.path.basename(checkpoint_path)
  if archive_name.endswith('.tgz'):
    dir_name = archive_name[: -len('.tgz')]
  else:
    dir_name = archive_name

  output_dir = os.path.join(testdir, 'decompressed_checkpoints', dir_name)
  os.makedirs(output_dir, exist_ok=True)
  logging.info('Decompressing checkpoint %s to %s', checkpoint_path, output_dir)
  shutil.unpack_archive(checkpoint_path, output_dir)
  return output_dir


def create_checkpoint_manager(
    *,
    cp_path: str,
    cp_options: ocp.CheckpointManagerOptions,
    model_key: str = 'model',
    optimizer_key: str = 'optimizer',
    embedding_spec_key: str = 'embedding_spec',
) -> ocp.CheckpointManager:
  """Creates a checkpoint manager for the given checkpoint path.

  Args:
    cp_path: The path to the checkpoint directory.
    cp_options: The checkpoint manager options.
    model_key: The checkpoint item key name for the model state.
    optimizer_key: The checkpoint item key name for the optimizer state.
    embedding_spec_key: The checkpoint item key name for the embedding spec
      protobuf.

  Returns:
    The checkpoint manager.
  """
  cp_path = os.path.abspath(cp_path)
  handler_registry = ocp.DefaultCheckpointHandlerRegistry()
  std_handler = ocp.StandardCheckpointHandler()
  proto_handler = ocp.ProtoCheckpointHandler()

  handler_registry.add(model_key, ocp.args.StandardRestore, std_handler)
  handler_registry.add(optimizer_key, ocp.args.StandardRestore, std_handler)
  handler_registry.add(embedding_spec_key, ocp.args.ProtoRestore, proto_handler)
  handler_registry.add(model_key, ocp.args.StandardSave, std_handler)
  handler_registry.add(optimizer_key, ocp.args.StandardSave, std_handler)
  handler_registry.add(embedding_spec_key, ocp.args.ProtoSave, proto_handler)

  return ocp.CheckpointManager(
      directory=cp_path,
      options=cp_options,
      handler_registry=handler_registry,
  )


def save_checkpoint(
    *,
    cp_manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    feature_specs: Nested[embedding_spec.FeatureSpec],
    num_global_devices: int,
    num_sc_per_device: int,
    model_key: str = 'model',
    optimizer_key: str = 'optimizer',
    embedding_spec_key: str = 'embedding_spec',
) -> None:
  """Saves a checkpoint for a SparseCore NNX model.

  This isn't strictly necessary, but it serves as an example of writing a
  checkpoint that is compatible with the cross-topology checkpoint restore
  function below.

  Args:
    cp_manager: The Orbax CheckpointManager used to save the checkpoint.
    step: The current training step number.
    model: The NNX model module.
    optimizer: The NNX optimizer.
    feature_specs: A nested structure of FeatureSpecs defining the embedding
      configuration.
    num_global_devices: Total number of global devices (chips).
    num_sc_per_device: Number of SparseCores per device.
    model_key: Checkpoint item key name for the model state.
    optimizer_key: Checkpoint item key name for the optimizer state.
    embedding_spec_key: Checkpoint item key name for the embedding spec
      protobuf.

  Returns:
    None.
  """
  save_args = {
      model_key: ocp.args.StandardSave(nnx.state(model)),
      optimizer_key: ocp.args.StandardSave(nnx.state(optimizer, nnx.OptState)),
      embedding_spec_key: ocp.args.ProtoSave(
          embedding.create_proto_from_feature_specs(
              feature_specs,
              global_device_count=num_global_devices,
              num_sparsecore_per_device=num_sc_per_device,
          )
      ),
  }
  cp_manager.save(step, args=ocp.args.Composite(**save_args))


def restore_checkpoint(
    *,
    cp_manager: ocp.CheckpointManager | None = None,
    input_checkpoint_path: str | None = None,
    step: int | None = None,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    model_key: str = 'model',
    optimizer_key: str = 'optimizer',
    embedding_spec_key: str = 'embedding_spec',
) -> None:
  """Restores a checkpoint for a SparseCore NNX model.

  Args:
    cp_manager: The Orbax CheckpointManager used to restore the checkpoint. If
      both cp_manager and input_checkpoint_path are None, no restoration is
      performed.
    input_checkpoint_path: Path to the checkpoint directory to restore from. If
      cp_manager is not provided, a temporary CheckpointManager is created for
      this path.
    step: The checkpoint step to restore from. If None, restores from the latest
      step available in the checkpoint directory.
    model: The NNX model module.
    optimizer: The NNX optimizer whose state will be updated.
    model_key: Checkpoint item key name for the model state.
    optimizer_key: Checkpoint item key name for the optimizer state.
    embedding_spec_key: Checkpoint item key name for the embedding spec
      protobuf.

  Returns:
    None. Model and optimizer are updated in-place.
  """
  if cp_manager is None and input_checkpoint_path is None:
    return

  if cp_manager is None:
    assert input_checkpoint_path is not None
    logging.info('Restoring checkpoint from %s', input_checkpoint_path)
    cp_manager = create_checkpoint_manager(
        cp_path=input_checkpoint_path,
        cp_options=ocp.CheckpointManagerOptions(read_only=True),
        model_key=model_key,
        optimizer_key=optimizer_key,
        embedding_spec_key=embedding_spec_key,
    )
  else:
    logging.info('Restoring checkpoint from %s', cp_manager.directory)

  restore_step = _get_checkpoint_step(cp_manager, step)

  cp_metadata = cp_manager.item_metadata(restore_step)

  _verify_checkpoint_keys(
      cp_metadata, [model_key, optimizer_key, embedding_spec_key]
  )

  restore_args = {
      model_key: ocp.args.StandardRestore(nnx.state(model)),
      optimizer_key: ocp.args.StandardRestore(
          nnx.state(optimizer, nnx.OptState)
      ),
      embedding_spec_key: ocp.args.ProtoRestore(
          embedding_spec_pb2.EmbeddingSpecProto
      ),
  }
  restored_items = cp_manager.restore(
      restore_step,
      args=ocp.args.Composite(**restore_args),
  )
  embedding_spec_proto = restored_items[embedding_spec_key]
  logging.info('restored embedding_spec_proto: %s', embedding_spec_proto)

  nnx.update(model, restored_items[model_key])
  nnx.update(optimizer, restored_items[optimizer_key])

  embedding_layer_name = _get_embedding_layer_name(model)
  embedding_layer = getattr(model, embedding_layer_name)
  original_embedding_table = embedding_layer.embedding_table.get_value()
  new_embedding_table = {}
  for k, ev in original_embedding_table.items():
    sharding = ev.table.sharding
    new_table = _enforce_sharding_and_layout(ev.table, sharding)
    new_slot = jax.tree.map(
        lambda arr, s=sharding: _enforce_sharding_and_layout(arr, s), ev.slot
    )
    new_embedding_table[k] = embedding.EmbeddingVariables(
        table=new_table, slot=new_slot
    )
  embedding_layer.embedding_table.set_value(new_embedding_table)

  logging.info('Finished restoring checkpoint')
  return


def restore_cross_topology_checkpoint(
    *,
    cross_topology_checkpoint_restore_path: str | None,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    target_feature_specs: Nested[embedding_spec.FeatureSpec],
    num_global_devices: int,
    num_sc_per_device: int,
    model_key: str = 'model',
    optimizer_key: str = 'optimizer',
    embedding_spec_key: str = 'embedding_spec',
    embedding_layer_name: str | None = None,
    restore_step: int | None = None,
) -> None:
  """Restores a checkpoint from a different topology.

  Args:
    cross_topology_checkpoint_restore_path: Path to the checkpoint directory to
      restore from. If None, no restoration is performed.
    model: The NNX model module containing the embedding layer to be updated.
    optimizer: The NNX optimizer whose state will be updated.
    target_feature_specs: A nested structure of FeatureSpecs defining the target
      embedding configuration.
    num_global_devices: Total number of global devices (chips) in the target
      topology.
    num_sc_per_device: Number of SparseCores per device in the target topology.
    model_key: Checkpoint item key name for the model state.
    optimizer_key: Checkpoint item key name for the optimizer state.
    embedding_spec_key: Checkpoint item key name for the embedding spec
      protobuf.
    embedding_layer_name: The attribute name of the SparseCoreEmbed layer in the
      model. If None, the model is dynamically inspected to locate an attribute
      of type embed.SparseCoreEmbed.
    restore_step: The checkpoint step to restore from. If None, restores from
      the latest step available in the checkpoint directory.

  Returns:
    None. Model, optimizer, and embedding layer are updated in-place.
  """
  if cross_topology_checkpoint_restore_path is None:
    return
  logging.info(
      'Restoring cross-topology checkpoint from %s',
      cross_topology_checkpoint_restore_path,
  )
  cp_path = cross_topology_checkpoint_restore_path
  cp_manager = create_checkpoint_manager(
      cp_path=cp_path,
      cp_options=ocp.CheckpointManagerOptions(read_only=True),
      model_key=model_key,
      optimizer_key=optimizer_key,
      embedding_spec_key=embedding_spec_key,
  )

  restore_step = _get_checkpoint_step(cp_manager, restore_step)

  cp_metadata = cp_manager.item_metadata(restore_step)

  # Keys need to be consistent with the keys used during checkpoint saving.
  _verify_checkpoint_keys(
      cp_metadata, [model_key, optimizer_key, embedding_spec_key]
  )

  embedding_layer_name = _get_embedding_layer_name(model, embedding_layer_name)
  embedding_layer = getattr(model, embedding_layer_name)

  restore_args = {
      model_key: ocp.args.StandardRestore(
          _get_abstract_state(model, embedding_layer)
      ),
      optimizer_key: ocp.args.StandardRestore(
          _get_abstract_state(optimizer, embedding_layer)
      ),
      embedding_spec_key: ocp.args.ProtoRestore(
          embedding_spec_pb2.EmbeddingSpecProto
      ),
  }
  restored_items = cp_manager.restore(
      restore_step,
      args=ocp.args.Composite(**restore_args),
  )
  embedding_spec_proto = restored_items[embedding_spec_key]
  logging.info('restored embedding_spec_proto: %s', embedding_spec_proto)

  # Stash original embedding table variables which possess the correct physical
  # layout (0, 1)
  original_embedding_table = embedding_layer.embedding_table.get_value()

  # Update model and optimizer with restored dense parameters and optimizer
  # state
  nnx.update(model, restored_items[model_key])
  nnx.update(optimizer, restored_items[optimizer_key])

  # Note: 'embedding_table' is the name used in embed.SparseCoreEmbed.
  restored_embedding = restored_items[model_key][embedding_layer_name][
      'embedding_table'
  ]

  logging.info('Unstacking and unsharding tables...')
  start_time = time.time()
  stacked_tables_in = {k: ev.table for k, ev in restored_embedding.items()}
  tables = table_stacking.unstack_and_unshard_stacked_tables(
      stacked_tables_in,
      embedding_spec_proto,
      donate=False,
  )
  end_time = time.time()
  logging.info('Total unstack/unshard time taken: %s', end_time - start_time)

  if logging.vlog_is_on(1):
    _log_table_sizes(tables)

  table_specs = _get_table_specs(target_feature_specs)

  num_shards = num_global_devices * num_sc_per_device
  stacked_tables = table_stacking.stack_and_shard_tables(
      table_specs,
      tables,
      num_shards=num_shards,
  )

  # Unstack, unshard, and restack/reshard slot variables
  stacked_slots = _unstack_and_restack_slots(
      restored_embedding,
      embedding_spec_proto,
      table_specs,
      num_shards,
  )

  # Reconstruct the model's embedding table variables with the newly restacked
  new_embedding_table = {}
  # Iterate over the model's original embedding table dictionary to preserve
  # the exact keys and expected variable structures.
  for k, ev in original_embedding_table.items():
    sharding = ev.table.sharding
    # Reshape the 3D stacked table back to the expected 2D shape and enforce
    # the target TPU memory layout and sharding.
    stacked_table = stacked_tables[k]
    assert isinstance(stacked_table, jax.Array)
    new_table = _enforce_sharding_and_layout(
        stacked_table.reshape(ev.table.shape),
        sharding,
    )
    # Process each optimizer slot variable for this table/stack identically.
    new_slot = jax.tree.map(
        lambda slot_arr, slot_leaf, s=sharding: _enforce_sharding_and_layout(
            slot_arr.reshape(slot_leaf.shape),
            s,
        ),
        stacked_slots[k],
        ev.slot,
    )
    new_embedding_table[k] = embedding.EmbeddingVariables(
        table=new_table, slot=new_slot
    )
  # Update the NNX model's embedding layer with the fully restored variables.
  embedding_layer.embedding_table.set_value(new_embedding_table)

  logging.info('Finished restoring cross-topology checkpoint')
  return


def convert_cross_topology_checkpoint(
    *,
    input_checkpoint_path: str | None,
    output_checkpoint_path: str | None,
    num_global_devices: int,
    num_sc_per_device: int,
    target_batch_size: int | None = None,
    model_key: str = 'model',
    optimizer_key: str = 'optimizer',
    embedding_spec_key: str = 'embedding_spec',
    restore_step: int | None = None,
    save_step: int | None = None,
    target_feature_specs: Nested[embedding_spec.FeatureSpec] | None = None,
) -> None:
  """Converts a checkpoint from a different topology on CPU and saves a new checkpoint.

  Args:
    input_checkpoint_path: Path to the input checkpoint directory to restore
      from. If None, no conversion is performed.
    output_checkpoint_path: Path to the output checkpoint directory to save the
      converted checkpoint. If None, no conversion is performed.
    num_global_devices: Total number of global devices (chips) in the target
      topology.
    num_sc_per_device: Number of SparseCores per device in the target topology.
    target_batch_size: Optional target global batch size for recomputing the
      embedding specs.
    model_key: Checkpoint item key name for the model state.
    optimizer_key: Checkpoint item key name for the optimizer state.
    embedding_spec_key: Checkpoint item key name for the embedding spec
      protobuf.
    restore_step: The checkpoint step to restore from. If None, restores from
      the latest step available in the input checkpoint directory.
    save_step: The checkpoint step to save the converted checkpoint as. If None,
      defaults to the restore_step.
    target_feature_specs: Optional nested structure of FeatureSpecs defining the
      target embedding configuration. If provided, the checkpoint will be
      restacked according to the target stacking rules.

  Returns:
    None. A new checkpoint is saved to output_checkpoint_path.
  """
  if input_checkpoint_path is None or output_checkpoint_path is None:
    return
  logging.info(
      'Converting cross-topology checkpoint from %s to %s on CPU',
      input_checkpoint_path,
      output_checkpoint_path,
  )

  restore_cp_manager = create_checkpoint_manager(
      cp_path=input_checkpoint_path,
      cp_options=ocp.CheckpointManagerOptions(read_only=True),
      model_key=model_key,
      optimizer_key=optimizer_key,
      embedding_spec_key=embedding_spec_key,
  )

  restore_step = _get_checkpoint_step(restore_cp_manager, restore_step)

  if save_step is None:
    save_step = restore_step

  restore_cp_metadata = restore_cp_manager.item_metadata(restore_step)

  # Keys need to be consistent with the keys used during checkpoint saving.
  _verify_checkpoint_keys(
      restore_cp_metadata, [model_key, optimizer_key, embedding_spec_key]
  )

  cpu_device = jax.devices('cpu')[0]
  cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)

  restore_args = {
      model_key: ocp.args.StandardRestore(fallback_sharding=cpu_sharding),
      optimizer_key: ocp.args.StandardRestore(fallback_sharding=cpu_sharding),
      embedding_spec_key: ocp.args.ProtoRestore(
          item=embedding_spec_pb2.EmbeddingSpecProto
      ),
  }
  restored_items = restore_cp_manager.restore(
      restore_step, args=ocp.args.Composite(**restore_args)
  )
  embedding_spec_proto = restored_items[embedding_spec_key]
  logging.info('restored embedding_spec_proto: %s', embedding_spec_proto)

  model_state = restored_items[model_key]
  optimizer_state = restored_items[optimizer_key]
  restored_embedding = _find_embedding_table_in_dict(model_state)
  if restored_embedding is None:
    raise ValueError(
        "Could not find 'embedding_table' in the restored model state."
    )

  logging.info('Unstacking and unsharding tables on CPU...')
  start_time = time.time()

  if 'value' in restored_embedding:
    actual_tables_dict = restored_embedding['value']
  else:
    actual_tables_dict = restored_embedding

  stacked_tables_in = {k: ev['table'] for k, ev in actual_tables_dict.items()}
  tables = table_stacking.unstack_and_unshard_stacked_tables(
      stacked_tables_in,
      embedding_spec_proto,
      donate=False,
  )
  end_time = time.time()
  logging.info('Total unstack/unshard time taken: %s', end_time - start_time)

  if logging.vlog_is_on(1):
    _log_table_sizes(tables)

  if target_feature_specs is not None:
    logical_table_specs = _get_table_specs(target_feature_specs)
    target_proto = embedding.create_proto_from_feature_specs(
        target_feature_specs, num_global_devices, num_sc_per_device
    )
  else:
    target_proto, logical_table_specs = _recompute_target_specs(
        embedding_spec_proto,
        num_global_devices,
        num_sc_per_device,
        target_batch_size=target_batch_size,
    )

  num_shards = num_global_devices * num_sc_per_device
  stacked_tables = table_stacking.stack_and_shard_tables(
      logical_table_specs,
      tables,
      num_shards=num_shards,
  )

  stacked_slots = _unstack_and_restack_slots(
      actual_tables_dict,
      embedding_spec_proto,
      logical_table_specs,
      num_shards,
  )
  new_actual_tables_dict = {}
  # Inject the reshaped parameters back into the model pure arrays dictionary
  for st_name, stacked_table in stacked_tables.items():
    assert isinstance(stacked_table, jax.Array)
    reshaped_table = stacked_table.reshape((-1, stacked_table.shape[-1]))
    reshaped_slot = jax.tree.map(
        lambda sv: sv.reshape((-1, sv.shape[-1])),
        stacked_slots[st_name],
    )
    new_actual_tables_dict[st_name] = {
        'table': reshaped_table,
        'slot': reshaped_slot,
    }

  if 'value' in restored_embedding:
    restored_embedding['value'] = new_actual_tables_dict
  else:
    restored_embedding.clear()
    restored_embedding.update(new_actual_tables_dict)

  # Save the new checkpoint for the updated topology
  logging.info('Saving converted checkpoint to %s', output_checkpoint_path)

  # Set up CheckpointManager for saving
  out_cp_path = os.path.abspath(output_checkpoint_path)
  out_cp_manager = create_checkpoint_manager(
      cp_path=out_cp_path,
      cp_options=ocp.CheckpointManagerOptions(create=True),
      model_key=model_key,
      optimizer_key=optimizer_key,
      embedding_spec_key=embedding_spec_key,
  )

  # Using StandardSave without arguments ensures pure nested dictionaries are
  # saved cleanly.
  out_cp_manager.save(
      save_step,
      args=ocp.args.Composite(
          model=ocp.args.StandardSave(model_state),
          optimizer=ocp.args.StandardSave(optimizer_state),
          embedding_spec=ocp.args.ProtoSave(target_proto),
      ),
  )
  out_cp_manager.wait_until_finished()

  logging.info('Finished converting cross-topology checkpoint')
  return


def _enforce_sharding_and_layout(
    arr: jax.Array, base_sharding: jax.sharding.NamedSharding
) -> jax.Array:
  """Enforces the target physical memory layout and sharding on an array.

  Args:
    arr: The array to enforce sharding and layout on.
    base_sharding: The base sharding to enforce.

  Returns:
    The array with the enforced sharding and layout.

  SparseCore requires embedding tables and optimizer slot variables to possess
  a specific physical tile layout in TPU memory (e.g., minor/major dimension
  ordering). This helper uses `with_layout_constraint` inside a `shard_map`
  to repack the underlying device buffers to match the expected layout.
  """
  assert base_sharding.mesh is not None
  assert base_sharding.spec is not None
  fmt = utils.embedding_table_format(base_sharding.mesh, base_sharding.spec)
  sharding_axis = next((s for s in base_sharding.spec if s is not None), None)
  if not isinstance(fmt, jax.experimental.layout.Format):
    return jax.device_put(arr, fmt)

  repack = jax.jit(
      jax.shard_map(
          lambda x: jax.experimental.layout.with_layout_constraint(
              # x + 0.0 is a common JAX workaround to force the application of
              # layout constraints.
              x + 0.0,
              fmt.layout,
          ),
          mesh=base_sharding.mesh,
          in_specs=jax.sharding.PartitionSpec(sharding_axis, None),
          out_specs=base_sharding.spec,
      ),
      out_shardings=fmt,
  )
  return repack(arr)


def _verify_checkpoint_keys(
    cp_metadata: collections.abc.Mapping[str, Any],
    expected_keys: collections.abc.Collection[str],
) -> None:
  """Verifies that the checkpoint keys match the expected keys."""
  for key in expected_keys:
    if key not in cp_metadata:
      raise ValueError(f'No "{key}" found in checkpoint metadata.')


def _get_embedding_layer_name(
    model: nnx.Module,
    embedding_layer_name: str | None = None,
) -> str:
  """Returns the name of the SparseCoreEmbed layer in the model."""
  if embedding_layer_name is None:
    for attr_name in dir(model):
      if isinstance(getattr(model, attr_name, None), embed.SparseCoreEmbed):
        return attr_name
    raise ValueError(
        'Could not automatically locate an attribute of type '
        'embed.SparseCoreEmbed in model. Please specify '
        'embedding_layer_name explicitly.'
    )
  if not hasattr(model, embedding_layer_name):
    raise ValueError(
        f'Model does not have an attribute named {embedding_layer_name} of type'
        ' embed.SparseCoreEmbed.'
    )
  if not isinstance(
      getattr(model, embedding_layer_name), embed.SparseCoreEmbed
  ):
    raise ValueError(
        f'Attribute {embedding_layer_name} in model is not of type '
        'embed.SparseCoreEmbed.'
    )
  return embedding_layer_name


def _get_abstract_state(
    node: nnx.Module | nnx.Optimizer, embedding_layer: embed.SparseCoreEmbed
) -> nnx.State:
  """Returns the abstract state of the given NNX module or optimizer."""
  abs_state = jax.eval_shape(lambda: nnx.state(node))
  real_dict = dict(nnx.to_flat_state(nnx.state(node)))
  mesh = embedding_layer.mesh

  def fn(path, var_abs):
    var_real = real_dict[path]
    if isinstance(var_real, embed.EmbeddingVariablesParam):
      sharding = var_real.embedding_sharding
    else:
      sharding = nnx.get_named_sharding(var_real, mesh).get_raw_value()

    def attach(sds):
      if isinstance(sds, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(sds.shape, sds.dtype, sharding=sharding)
      return sds

    new_value = jax.tree.map(attach, var_abs.get_raw_value())
    return var_abs.replace(value=new_value)

  return nnx.map_state(fn, abs_state)


def _get_abstract_state_cpu(
    node: nnx.Module | nnx.Optimizer,
    cpu_sharding: jax.sharding.Sharding,
) -> nnx.State:
  abs_state = jax.eval_shape(lambda: nnx.state(node))

  def attach(sds):
    if isinstance(sds, jax.ShapeDtypeStruct):
      return jax.ShapeDtypeStruct(sds.shape, sds.dtype, sharding=cpu_sharding)
    return sds

  return jax.tree.map(attach, abs_state)


def _get_checkpoint_step(
    cp_manager: ocp.CheckpointManager, step: int | None
) -> int:
  """Returns the checkpoint step to use for the given checkpoint manager."""
  if step is None:
    step = cp_manager.latest_step()
    if step is None:
      raise ValueError(f'No checkpoints found in {cp_manager.directory}')
  elif step not in cp_manager.all_steps():
    raise ValueError(
        f'Step {step} not found in checkpoint directory {cp_manager.directory}'
    )
  return step


def _unstack_and_restack_slots(
    restored_embedding: collections.abc.Mapping[str, Any],
    embedding_spec_proto: embedding_spec_pb2.EmbeddingSpecProto,
    table_specs: dict[str, embedding_spec.TableSpec],
    num_shards: int,
) -> dict[str, Any]:
  """Unstacks, unshards, and restacks/reshards optimizer slot variables."""
  source_slots = {
      k: ev['slot'] if isinstance(ev, dict) else ev.slot
      for k, ev in restored_embedding.items()
  }
  if not source_slots:
    return {}

  first_slot = next(iter(source_slots.values()))
  if not jax.tree.leaves(first_slot):
    target_stack_names = {
        tspec.stacked_table_spec.stack_name
        if tspec.stacked_table_spec
        else tspec.name
        for tspec in table_specs.values()
    }
    return {st_name: first_slot for st_name in target_stack_names}

  target_stack_names = set()
  for tspec in table_specs.values():
    if tspec.stacked_table_spec:
      target_stack_names.add(tspec.stacked_table_spec.stack_name)
    else:
      target_stack_names.add(tspec.name)

  def _restack_leaf(
      *leaves_across_source_stacks: jax.Array,
  ) -> dict[str, Any]:
    source_stacked_slot_dict = dict(
        zip(source_slots.keys(), leaves_across_source_stacks, strict=True)
    )
    unstacked_slot_tables = table_stacking.unstack_and_unshard_stacked_tables(
        source_stacked_slot_dict,
        embedding_spec_proto,
        donate=False,
    )
    return table_stacking.stack_and_shard_tables(
        table_specs,
        unstacked_slot_tables,
        num_shards=num_shards,
        pad_value=0.0,
    )

  tree_of_target_slots = jax.tree.map(
      _restack_leaf,
      *source_slots.values(),
  )

  target_slots = {
      st_name: jax.tree.map(
          lambda target_stacked_map, s_name=st_name: target_stacked_map[s_name],
          tree_of_target_slots,
          is_leaf=lambda x, s_name=st_name: isinstance(x, dict) and s_name in x,
      )
      for st_name in target_stack_names
  }
  return target_slots


def _log_table_sizes(tables: collections.abc.Mapping[str, jax.Array]):
  """Logs the sizes of the tables in the given dictionary."""
  sizes = []
  for k, v in tables.items():
    bytes_size = np.prod(v.shape) * 4 / (1024**3)
    logging.info(
        'Feature Table: key: %s, shape: %s, sharding: %s, bytes: %s GB',
        k,
        v.shape,
        v.sharding,
        bytes_size,
    )
    sizes.append(bytes_size)
  logging.info('Max table size = %s GB', np.max(sizes))


def _get_table_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> dict[str, embedding_spec.TableSpec]:
  """Returns a dictionary of table specs from the given feature specs."""
  table_specs = {}
  for v in jax.tree_util.tree_leaves(feature_specs):
    table_specs[v.table_spec.name] = v.table_spec
  return table_specs


def _next_largest_multiple(value: int, multiple: int) -> int:
  return ((value + multiple - 1) // multiple) * multiple


def _reconstruct_feature_specs_from_proto(
    source_proto: embedding_spec_pb2.EmbeddingSpecProto,
    target_batch_size: int | None = None,
) -> dict[str, embedding_spec.FeatureSpec]:
  """Reconstructs unstacked FeatureSpecs from an EmbeddingSpecProto.

  Args:
    source_proto: The source embedding spec proto.
    target_batch_size: Optional target global batch size.

  Returns:
    A dictionary mapping feature names to reconstructed unstacked FeatureSpecs.
  """
  specs = {}
  for stack_proto in source_proto.stacked_table_specs:
    for t_proto in stack_proto.table_specs:
      if t_proto.HasField('optimizer'):
        opt = embedding.proto_to_optimizer_spec(t_proto.optimizer)
      else:
        opt = embedding_spec.SGDOptimizerSpec(learning_rate=0.0)
      combiner = t_proto.combiner if t_proto.combiner else 'mean'
      tspec = embedding_spec.TableSpec(
          vocabulary_size=t_proto.vocab_size,
          embedding_dim=t_proto.embedding_dim,
          initializer=jax.nn.initializers.constant(0.0),
          optimizer=opt,
          combiner=combiner,
          name=t_proto.table_name,
      )
      if t_proto.feature_specs:
        for f_proto in t_proto.feature_specs:
          in_shape = list(f_proto.input_shape)
          out_shape = list(f_proto.output_shape)
          if target_batch_size is not None and in_shape:
            in_shape[0] = target_batch_size
          if target_batch_size is not None and out_shape:
            out_shape[0] = target_batch_size
          fspec = embedding_spec.FeatureSpec(
              table_spec=tspec,
              input_shape=tuple(in_shape),
              output_shape=tuple(out_shape),
              name=f_proto.feature_name,
          )
          specs[f_proto.feature_name] = fspec
      else:
        bs = (
            target_batch_size
            if target_batch_size is not None
            else stack_proto.total_sample_count
        )
        fspec = embedding_spec.FeatureSpec(
            table_spec=tspec,
            input_shape=(bs, 1),
            output_shape=(bs, t_proto.embedding_dim),
            name=f'feature_{t_proto.table_name}',
        )
        specs[fspec.name] = fspec
  return specs


def _recompute_target_specs(
    source_proto: embedding_spec_pb2.EmbeddingSpecProto,
    num_global_devices: int,
    num_sc_per_device: int,
    target_batch_size: int | None = None,
) -> tuple[
    embedding_spec_pb2.EmbeddingSpecProto, dict[str, embedding_spec.TableSpec]
]:
  """Recomputes the target embedding spec proto and table specs.

  Args:
    source_proto: The source embedding spec proto.
    num_global_devices: The number of global devices.
    num_sc_per_device: The number of SparseCores per device.
    target_batch_size: Optional target global batch size.

  Returns:
    A tuple of the target embedding spec proto and a dictionary of table specs.
  """
  # If source_proto contains combiner and optimizer specs, perform automatic
  # stack regrouping via auto_stack_tables.
  has_full_metadata = False
  for s in source_proto.stacked_table_specs:
    if any(t.combiner and t.HasField('optimizer') for t in s.table_specs):
      has_full_metadata = True
      break

  if has_full_metadata:
    reconstructed_specs = _reconstruct_feature_specs_from_proto(
        source_proto, target_batch_size=target_batch_size
    )
    table_stacking.auto_stack_tables(
        reconstructed_specs,
        global_device_count=num_global_devices,
        num_sc_per_device=num_sc_per_device,
    )
    target_proto = embedding.create_proto_from_feature_specs(
        reconstructed_specs, num_global_devices, num_sc_per_device
    )
    logical_table_specs = _get_table_specs(reconstructed_specs)
    return target_proto, logical_table_specs

  num_shards = num_global_devices * num_sc_per_device
  target_proto = embedding_spec_pb2.EmbeddingSpecProto()
  target_proto.CopyFrom(source_proto)

  logical_table_specs = {}

  for stacked_spec_proto in target_proto.stacked_table_specs:
    stack_activation_mem_bytes = 0
    for t in stacked_spec_proto.table_specs:
      padded_dim = _next_largest_multiple(t.embedding_dim, 8)
      if t.feature_specs:
        table_sample_count = sum(
            int(np.prod(f.output_shape[:-1])) for f in t.feature_specs
        )
      else:
        table_sample_count = stacked_spec_proto.total_sample_count
      if (
          target_batch_size is not None
          and stacked_spec_proto.total_sample_count
      ):
        scale = target_batch_size / stacked_spec_proto.total_sample_count
        table_sample_count = int(table_sample_count * scale)

      table_sample_count_per_sc = table_sample_count // num_shards
      stack_activation_mem_bytes += padded_dim * table_sample_count_per_sc * 4

    activation_mem_bytes_limit = 2048 * 1024  # 2 MiB
    if (
        len(stacked_spec_proto.table_specs) > 1
        and stack_activation_mem_bytes > activation_mem_bytes_limit
    ):
      raise ValueError(
          f"Stack '{stacked_spec_proto.stack_name}' has per-SparseCore"
          f' activation memory of {stack_activation_mem_bytes} bytes,'
          f' exceeding the 2 MiB limit ({activation_mem_bytes_limit} bytes) for'
          f' target topology ({num_global_devices} devices,'
          f' {num_sc_per_device} SC/device). The target model will group tables'
          ' into different stacks. Please provide target_feature_specs to'
          ' convert_cross_topology_checkpoint.'
      )

    stack_embedding_dim = max([
        _next_largest_multiple(t.embedding_dim, 8)
        for t in stacked_spec_proto.table_specs
    ])
    stacked_spec_proto.stack_embedding_dim = stack_embedding_dim
    stacked_spec_proto.num_sparsecores = num_shards

    table_names = [t.table_name for t in stacked_spec_proto.table_specs]
    table_to_padded_vocab_size = {
        t.table_name: _next_largest_multiple(t.vocab_size, 8 * num_shards)
        for t in stacked_spec_proto.table_specs
    }
    table_to_setting = table_stacking.compute_table_to_setting_in_stack(
        stack_name=stacked_spec_proto.stack_name,
        table_names=table_names,
        padded_embedding_dim=stack_embedding_dim,
        table_to_padded_vocab_size=table_to_padded_vocab_size,
        num_shards=num_shards,
        rotation=num_sc_per_device,
    )

    stack_vocab_size = sum(table_to_padded_vocab_size.values())
    stacked_spec_proto.stack_vocab_size = stack_vocab_size

    for table_spec_proto in stacked_spec_proto.table_specs:
      setting = table_to_setting[table_spec_proto.table_name]
      table_spec_proto.padded_vocab_size = setting.padded_vocab_size
      table_spec_proto.padded_embedding_dim = setting.padded_embedding_dim
      table_spec_proto.row_offset_in_shard = setting.row_offset_in_shard
      table_spec_proto.shard_rotation = setting.shard_rotation

      tspec = embedding_spec.TableSpec(
          name=table_spec_proto.table_name,
          vocabulary_size=table_spec_proto.vocab_size,
          embedding_dim=table_spec_proto.embedding_dim,
          initializer=jax.nn.initializers.constant(0.0),
          optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.0),
          combiner='mean',
      )
      tspec.setting_in_stack = setting

      tspec.stacked_table_spec = embedding_spec.StackedTableSpec(
          stack_name=stacked_spec_proto.stack_name,
          stack_vocab_size=stack_vocab_size,
          stack_embedding_dim=stacked_spec_proto.stack_embedding_dim,
          optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.0),
          combiner='mean',
          total_sample_count=stacked_spec_proto.total_sample_count,
      )

      logical_table_specs[table_spec_proto.table_name] = tspec

  return target_proto, logical_table_specs


def _find_embedding_table_in_dict(state_dict: Any) -> dict[str, Any] | None:
  """Recursively searches for the embedding_table subdictionary."""
  if not isinstance(state_dict, dict):
    return None
  # 'embedding_table' is the embedding variable name in SparseCoreEmbed.
  if 'embedding_table' in state_dict and isinstance(
      state_dict['embedding_table'], dict
  ):
    return state_dict['embedding_table']
  for v in state_dict.values():
    res = _find_embedding_table_in_dict(v)
    if res is not None:
      return res
  return None
