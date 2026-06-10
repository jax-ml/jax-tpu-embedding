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

import os
import time

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
  if embedding_layer_name is None:
    for attr_name in dir(model):
      if isinstance(getattr(model, attr_name, None), embed.SparseCoreEmbed):
        embedding_layer_name = attr_name
        break
    if embedding_layer_name is None:
      raise ValueError(
          'Could not automatically locate an attribute of type '
          'embed.SparseCoreEmbed in model. Please specify '
          'embedding_layer_name explicitly.'
      )

  cp_path = os.path.abspath(cross_topology_checkpoint_restore_path)
  cp_options = ocp.CheckpointManagerOptions(read_only=True)
  handler_registry = ocp.DefaultCheckpointHandlerRegistry()
  std_handler = ocp.StandardCheckpointHandler()
  proto_handler = ocp.ProtoCheckpointHandler()

  handler_registry.add(model_key, ocp.args.StandardRestore, std_handler)
  handler_registry.add(optimizer_key, ocp.args.StandardRestore, std_handler)
  handler_registry.add(embedding_spec_key, ocp.args.ProtoRestore, proto_handler)

  cp_manager = ocp.CheckpointManager(
      directory=cp_path,
      options=cp_options,
      handler_registry=handler_registry,
  )

  if restore_step is None:
    restore_step = cp_manager.latest_step()
    if restore_step is None:
      raise ValueError(f'No checkpoints found in {cp_path}')
  elif restore_step not in cp_manager.all_steps():
    raise ValueError(
        f'Step {restore_step} not found in checkpoint directory {cp_path}'
    )

  cp_metadata = cp_manager.item_metadata(restore_step)

  # Keys need to be consistent with the keys used during checkpoint saving.
  expected_keys = [model_key, optimizer_key, embedding_spec_key]
  for key in expected_keys:
    if key not in cp_metadata:
      raise ValueError(f'No "{key}" found in checkpoint metadata.')

  embedding_layer = getattr(model, embedding_layer_name)

  def get_abstract_state(node: nnx.Module | nnx.Optimizer) -> nnx.State:
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

  restore_args = {
      model_key: ocp.args.StandardRestore(get_abstract_state(model)),
      optimizer_key: ocp.args.StandardRestore(get_abstract_state(optimizer)),
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
    logging.info('addressable data 1st row: %s', v.addressable_data(0)[0])
    sizes.append(bytes_size)
  logging.info('Max table size = %s GB', np.max(sizes))

  table_specs = {}
  for v in jax.tree_util.tree_leaves(target_feature_specs):
    table_specs[v.table_spec.name] = v.table_spec

  num_shards = num_global_devices * num_sc_per_device
  stacked_tables = table_stacking.stack_and_shard_tables(
      table_specs,
      tables,
      num_shards=num_shards,
  )

  # Unstack, unshard, and restack/reshard slot variables
  stacked_slots = {}
  for k, ev in restored_embedding.items():
    spec_proto_k = embedding_spec_pb2.EmbeddingSpecProto()
    for stacked_table_spec in embedding_spec_proto.stacked_table_specs:
      if stacked_table_spec.stack_name == k:
        spec_proto_k.stacked_table_specs.append(stacked_table_spec)
        break
    stacked_slots_k = []
    for i in range(len(ev.slot)):
      slots = table_stacking.unstack_and_unshard_stacked_tables(
          {k: ev.slot[i]},
          spec_proto_k,
          donate=False,
      )
      table_specs_k = {
          name: spec for name, spec in table_specs.items() if name in slots
      }
      stacked_slot_i = table_stacking.stack_and_shard_tables(
          table_specs_k,
          slots,
          num_shards=num_shards,
          pad_value=0.0,
      )
      stacked_slots_k.append(stacked_slot_i[k])
    stacked_slots[k] = tuple(stacked_slots_k)

  # Reconstruct the model's embedding table variables with the newly restacked
  # tables and optimizer slot variables, ensuring correct layout and sharding.
  def enforce_sharding_and_layout(arr, base_sharding):
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
    fmt = utils.embedding_table_format(base_sharding.mesh, base_sharding.spec)
    sharding_axis = next((s for s in base_sharding.spec if s is not None), None)
    repack = jax.jit(
        jax.shard_map(
            lambda x: jax.experimental.layout.with_layout_constraint(
                # x + 0.0 is a common JAX workaround to force the application of
                # layout constraints.
                x + 0.0,
                fmt.layout,  # pytype: disable=attribute-error
            ),
            mesh=base_sharding.mesh,
            in_specs=jax.sharding.PartitionSpec(sharding_axis, None),
            out_specs=base_sharding.spec,
        ),
        out_shardings=fmt,
    )
    return repack(arr)

  new_embedding_table = {}
  # Iterate over the model's original embedding table dictionary to preserve
  # the exact keys and expected variable structures.
  for k, ev in original_embedding_table.items():
    sharding = ev.table.sharding
    # Reshape the 3D stacked table back to the expected 2D shape and enforce
    # the target TPU memory layout and sharding.
    new_table = enforce_sharding_and_layout(
        stacked_tables[k].reshape(ev.table.shape),  # pytype: disable=attribute-error
        sharding,
    )
    # Process each optimizer slot variable for this table/stack identically.
    new_slot = tuple(
        enforce_sharding_and_layout(
            stacked_slots[k][i].reshape(ev.slot[i].shape),  # pytype: disable=attribute-error
            sharding,
        )
        for i in range(len(ev.slot))
    )
    new_embedding_table[k] = embedding.EmbeddingVariables(
        table=new_table, slot=new_slot
    )
  # Update the NNX model's embedding layer with the fully restored variables.
  embedding_layer.embedding_table.set_value(new_embedding_table)

  logging.info('Finished restoring cross-topology checkpoint')
  return
