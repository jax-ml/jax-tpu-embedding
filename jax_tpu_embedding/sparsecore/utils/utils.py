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
from jax._src import tpu_info
from jax.experimental import layout

Layout = layout.Layout

_ArrayType = typing.TypeVar('_ArrayType', bound=jax.typing.ArrayLike)

_SPARSECORE_INFOS: tuple[tpu_info.SparseCoreInfo, ...] = tuple(
    sc_info
    for chip in tpu_info.ChipVersion
    if (sc_info := tpu_info.get_tpu_info_for_chip(chip, 1).sparse_core)
    is not None
)
_MAX_NUM_SC_PER_DEVICE: int = max(sc.num_cores for sc in _SPARSECORE_INFOS)
_VALID_SC_SIMD_WIDTHS: frozenset[int] = frozenset(
    sc.num_lanes for sc in _SPARSECORE_INFOS
)
_VALID_HBM_WORD_SIZES_IN_4B: frozenset[int] = frozenset(
    sc.dma_granule_size_bytes // 4 for sc in _SPARSECORE_INFOS
)


class DeviceLike(typing.Protocol):
  @property
  def device_kind(self) -> str:
    ...


MeshLike: typing.TypeAlias = jax.sharding.Mesh | jax.sharding.AbstractMesh


def _get_sparsecore_info(
    device: DeviceLike | None = None,
) -> tpu_info.SparseCoreInfo:
  """Returns SparseCore hardware info for a device (or the default device)."""
  target_device = device or jax.devices()[0]

  if not hasattr(target_device, 'device_kind'):
    raise ValueError(
        f'Cannot determine device kind for device: {target_device}'
    )

  device_kind = target_device.device_kind
  chip_version = tpu_info.chip_version_from_device_kind(device_kind)
  if chip_version is not None:
    sc_info = tpu_info.get_tpu_info_for_chip(chip_version, 1).sparse_core
    if sc_info is not None:
      return sc_info

  raise ValueError(
      f'SparseCore is not supported on device kind: {device_kind}.'
  )


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
  return _get_sparsecore_info(device).num_cores


def is_valid_num_sparsecores_per_device(num_sc_per_device: int) -> bool:
  """Returns whether a device can use `num_sc_per_device` sparsecores.

  A device may use only a subset of its sparsecores for embeddings, e.g. to
  leave the remaining sparsecores available for other work. Therefore, any power
  of two up to the largest number of sparsecores per device is valid. Note that
  the compiler must then be configured to run the embedding lookups on the same
  number of sparsecores per device.

  Args:
    num_sc_per_device: The number of sparsecores per device.

  Returns:
    True if `num_sc_per_device` is a valid number of sparsecores per device.
  """
  return (
      0 < num_sc_per_device <= _MAX_NUM_SC_PER_DEVICE
      and num_sc_per_device & (num_sc_per_device - 1) == 0
  )


def sparsecore_simd_width(
    device: DeviceLike | None = None,
) -> int:
  """Determine the SparseCore SIMD width available on a device.

  Args:
    device: JAX device to check.  If None, queries the first device in
      jax.devices().

  Returns:
    SparseCore SIMD width.

  Raises:
    ValueError: if the SparseCore SIMD width cannot be determined.
  """
  return _get_sparsecore_info(device).num_lanes


def is_valid_sparsecore_simd_width(sc_simd_width: int) -> bool:
  """Returns whether `sc_simd_width` is a valid SparseCore SIMD width."""
  return sc_simd_width in _VALID_SC_SIMD_WIDTHS


def hbm_word_size_in_4b(
    device: DeviceLike | None = None,
) -> int:
  """Determine the HBM word size in 4-byte words for a device.

  Args:
    device: JAX device to check. If None, queries the first device in
      jax.devices().

  Returns:
    HBM word size in 4-byte words.

  Raises:
    ValueError: if the HBM word size in 4-byte words cannot be determined.
  """
  return _get_sparsecore_info(device).dma_granule_size_bytes // 4


def is_valid_hbm_word_size_in_4b(hbm_word_size_in_4b_val: int) -> bool:
  """Returns whether `hbm_word_size_in_4b_val` is a valid HBM word size."""
  return hbm_word_size_in_4b_val in _VALID_HBM_WORD_SIZES_IN_4B


sparsecore_hbm_word_size_in_4b = hbm_word_size_in_4b


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
    num_sc_per_device: Number of SparseCores per device (e.g. 4 for TPU v5p, 2
      for TPU v6e and TPU7x).
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
