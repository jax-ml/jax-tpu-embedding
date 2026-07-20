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
"""Utilities for examples."""

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
