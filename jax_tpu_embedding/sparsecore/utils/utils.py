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

import logging
import os

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


def jit_with_dump(fn, name=None, source_info=None, *jit_args, **jit_kwargs):
  """A wrapper for a jitted function that dumps the jaxpr to a file."""
  jitted_fn = jax.jit(fn, *jit_args, **jit_kwargs)
  dirname = (
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
      if _DUMP_DIR.value == 'sponge'
      else _DUMP_DIR.value
  )
  dir_list = [dirname] if dirname else []
  if name is None:
    name = fn.__name__

  make_jaxpr_param_list = ('static_argnums',)
  make_jaxpr_params = {
      k: jit_kwargs[k] for k in make_jaxpr_param_list if k in jit_kwargs
  }

  def call(*args, **kwargs):
    if dir_list:
      jaxpr = jax.make_jaxpr(fn, **make_jaxpr_params)(*args, **kwargs)
      dump_path = os.path.join(dir_list[0], f'jaxpr_{name}.txt')
      logging.info('Writing %s Jaxpr to %s', name, dump_path)
      with open(dump_path, 'w') as f:
        f.write(jaxpr.pretty_print(source_info=source_info))
      # Only dump for the first time
      dir_list.clear()
    return jitted_fn(*args, **kwargs)

  return call


def tree_summary(tree):
  """Returns the shape and dtype of each leaf in the tree."""
  return jax.tree.map(lambda x: (x.shape, x.dtype), tree)
