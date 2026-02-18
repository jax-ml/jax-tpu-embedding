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

from absl import flags
import jax
from jax.experimental import layout

if jax.__version_info__ >= (0, 6, 3):
  Layout = layout.Layout
else:
  Layout = layout.DeviceLocalLayout  # type: ignore


_DUMP_DIR = flags.DEFINE_string(
    'dump_dir', None, 'Directory to write debug dumps to.'
)

NUM_SC_PER_DEVICE_MAP = {
    'TPU v5': 4,
    'TPU v5p': 4,  # In pathways setup, TPU v5 shows up as TPU v5p.
    'TPU v6 lite': 2,
    'TPU7x': 2,  # Megacore is disabled.
}


def num_sparsecores_per_device(device: jax.Device | None = None):
  """Determine the number of sparsecores available on a device.

  Args:
    device: JAX device to check.  If None, queries the first device in
      jax.devices().

  Returns:
    Number of sparsecores.

  Raises:
    ValueError: if the number of sparsecores cannot be determined.
  """
  device = device or jax.devices()[0]

  if not hasattr(device, 'device_kind'):
    raise ValueError(f'Cannot determine device kind for device: {device}')

  device_kind = device.device_kind
  if device_kind not in NUM_SC_PER_DEVICE_MAP:
    raise ValueError(f'Unknown sparsecore count for device kind: {device_kind}')

  return NUM_SC_PER_DEVICE_MAP[device_kind]


def tree_summary(tree):
  """Returns the shape and dtype of each leaf in the tree."""
  return jax.tree.map(lambda x: (x.shape, x.dtype), tree)


def embedding_table_format(
    mesh: jax.sharding.Mesh, partition_spec: jax.sharding.PartitionSpec
) -> jax.sharding.Sharding:
  """Returns the layout format of the embedding table."""
  return embedding_table_format_with_sharding(
      jax.sharding.NamedSharding(mesh, partition_spec)
  )


def embedding_table_format_with_sharding(
    sharding: jax.sharding.Sharding,
) -> jax.sharding.Sharding:
  """Returns the layout format of the embedding table."""
  return layout.Format(  # pytype: disable=bad-return-type
      Layout(
          major_to_minor=(0, 1),
          tiling=((8,),),
      ),
      sharding,
  )
