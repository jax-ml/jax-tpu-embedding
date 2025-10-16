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


_DUMP_DIR = flags.DEFINE_string(
    'dump_dir', None, 'Directory to write debug dumps to.'
)

NUM_SC_PER_DEVICE_MAP = {
    'TPU v5': 4,
    'TPU v6 lite': 2,
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
