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
"""Utilities for SparseCore embedding."""

import math
import typing

import einops
import jax
from jax.experimental import layout

Layout = layout.Layout

_ArrayType = typing.TypeVar('_ArrayType', bound=jax.typing.ArrayLike)

# The device kind names (keys) must align with the external names mapped in
# https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#versions.
# The SparseCore counts (values) should match the JAX TPU info in
# https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py.
NUM_SC_PER_DEVICE_MAP = {
    'TPU v5': 4,
    'TPU v5p': 4,  # Alias for 'TPU v5'.
    'TPU v6e': 2,  # Trillium.
    'TPU v6 lite': 2,  # Alias for 'TPU v6e'.
    'TPU7x': 2,  # Ironwood. Megacore is disabled.
}


class DeviceLike(typing.Protocol):
  @property
  def device_kind(self) -> str:
    ...


MeshLike: typing.TypeAlias = jax.sharding.Mesh | jax.sharding.AbstractMesh


def num_sparsecores_per_device(
    device: DeviceLike | None = None,
) -> int:
  """Determine the number of sparsecores available on a device.

  Args:
    device: JAX device to check.  If None, queries the first device in
      jax.devices().

  Returns:
    Number of sparsecores.

  Raises:
    ValueError: if the number of sparsecores cannot be determined.
  """
  target_device = device or jax.devices()[0]

  if not hasattr(target_device, 'device_kind'):
    raise ValueError(
        f'Cannot determine device kind for device: {target_device}'
    )

  device_kind = target_device.device_kind
  if device_kind not in NUM_SC_PER_DEVICE_MAP:
    raise ValueError(
        f'Unknown sparsecore count for device kind: {device_kind}. Known'
        f' device kinds: {NUM_SC_PER_DEVICE_MAP.keys()}'
    )

  return NUM_SC_PER_DEVICE_MAP[device_kind]


def embedding_table_format(
    mesh: MeshLike,
    partition_spec: jax.sharding.PartitionSpec,
) -> jax.sharding.Sharding | layout.Format:
  """Returns the layout format of the embedding table."""
  return embedding_table_format_with_sharding(
      jax.sharding.NamedSharding(mesh, partition_spec)
  )


def shard_emb_table(
    table: _ArrayType,
    *,
    num_devices: int,
    num_sc_per_device: int,
    sharding_strategy: str = 'MOD',
) -> _ArrayType:
  """Shards an embedding table for SparseCore using MOD sharding.

  Args:
    table: Unsharded embedding table of shape [vocab_size, emb_dim].
    num_devices: Number of chips/devices.
    num_sc_per_device: Number of SparseCores per device.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).

  Returns:
    Sharded table of shape [num_devices, num_sc_per_device * vocab_size_per_sc,
    emb_dim].
  """
  if sharding_strategy != 'MOD':
    raise ValueError('Currently only MOD sharding strategy is supported')
  return einops.rearrange(
      table, '(v c s) ... -> c (s v) ...', c=num_devices, s=num_sc_per_device
  )


def unshard_emb_table(
    sharded_table: _ArrayType,
    num_sc_per_device: int,
    sharding_strategy: str = 'MOD',
) -> _ArrayType:
  """Unshards embedding table from MOD sharding.

  Args:
    sharded_table: Sharded embedding table of shape [num_devices,
      num_sc_per_device * vocab_size_per_sc, emb_dim].
    num_sc_per_device: Number of SparseCores per device.
    sharding_strategy: Embedding table sharding strategy (only "MOD" supported).

  Returns:
    Unsharded table of shape [vocab_size, emb_dim].
  """
  if sharding_strategy != 'MOD':
    raise ValueError('Currently only MOD sharding strategy is supported')
  return einops.rearrange(
      sharded_table, 'c (s v) ... -> (v c s) ...', s=num_sc_per_device
  )


def embedding_table_format_with_sharding(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding | layout.Format:
  """Returns the layout format of the embedding table."""
  if hasattr(sharding, 'mesh') and isinstance(
      sharding.mesh, jax.sharding.AbstractMesh
  ):
    device_kind = getattr(sharding.mesh.abstract_device, 'device_kind', '')
  else:
    devices = list(sharding.device_set)
    device_kind = getattr(devices[0], 'device_kind', '') if devices else ''

  if device_kind == 'cpu':
    return sharding
  return layout.Format(
      Layout(
          major_to_minor=(0, 1),
          tiling=((8,),),
      ),
      sharding,
  )


def _round_up_to(value: float, alignment: int) -> int:
  """Rounds up a numerical value to the nearest multiple of alignment."""
  return math.ceil(value / alignment) * alignment


def estimate_preprocessing_parameters(
    *,
    global_batch_size: int,
    global_device_count: int,
    num_sc_per_device: int,
    valency: float,
    hot_shard_fraction: float = 0.20,
    headroom_factor: float = 1.25,
) -> dict[str, int]:
  """Estimates SparseCore preprocessing buffer size parameters.

  Uses closed-form formulas calibrated on real-world Zipfian/power-law
  distributions to estimate max IDs per partition and suggested COO buffer
  sizes.

  Args:
    global_batch_size: Total batch size across all devices.
    global_device_count: Number of devices (chips) in the mesh.
    num_sc_per_device: Number of SparseCores per device (e.g. 4 for v5e/v5p, 2
      for v6e).
    valency: Total average embedding lookups per sample across all stacked
      features.
    hot_shard_fraction: Assumed fraction of lookups concentrated on the hottest
      destination SparseCore shard (default: 0.20).
    headroom_factor: Safety headroom multiplier (default: 1.25).

  Returns:
    Dict containing estimated 'max_ids_per_partition',
    'max_unique_ids_per_partition', and
    'suggested_coo_buffer_size_per_device'.
  """
  if (
      global_batch_size <= 0
      or global_device_count <= 0
      or num_sc_per_device <= 0
      or valency <= 0
  ):
    raise ValueError(
        'global_batch_size, global_device_count, num_sc_per_device, and valency'
        ' must all be positive numbers.'
    )

  device_batch_size = global_batch_size // global_device_count
  sc_batch_size = device_batch_size // num_sc_per_device
  total_sparse_cores = global_device_count * num_sc_per_device
  device_alignment = 8 * num_sc_per_device
  sc_total_lookups = sc_batch_size * valency

  hot_shard_lookups = hot_shard_fraction * sc_total_lookups
  background_avg_lookups = (
      (1.0 - hot_shard_fraction) * sc_total_lookups / total_sparse_cores
  )
  tail_fluctuation = 2.0 * math.sqrt(sc_total_lookups / total_sparse_cores)
  expected_peak_lookups = (
      hot_shard_lookups + background_avg_lookups + tail_fluctuation
  )
  peak_partition_ids = headroom_factor * expected_peak_lookups

  max_ids_per_partition = _round_up_to(peak_partition_ids, 8)
  max_unique_ids_per_partition = _round_up_to(0.5 * peak_partition_ids, 8)

  min_device_buffer = num_sc_per_device * max_ids_per_partition
  estimated_device_lookups = (
      1.2 * device_batch_size * valency + 14 * total_sparse_cores
  )
  raw_device_buffer = max(min_device_buffer, estimated_device_lookups)
  suggested_coo_buffer_size_per_device = _round_up_to(
      raw_device_buffer, device_alignment
  )

  return {
      'max_ids_per_partition': max_ids_per_partition,
      'max_unique_ids_per_partition': max_unique_ids_per_partition,
      'suggested_coo_buffer_size_per_device': (
          suggested_coo_buffer_size_per_device
      ),
  }
