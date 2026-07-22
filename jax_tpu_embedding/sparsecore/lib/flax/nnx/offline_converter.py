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
"""Offline checkpoint converter for Shakespeare model across different TPU topologies."""

import collections
import os

from absl import app
from absl import flags
from absl import logging
from etils import epath
from jax_tpu_embedding.sparsecore.lib.flax.nnx import checkpoint_utils
from jax_tpu_embedding.sparsecore.utils import utils

_INPUT_CHECKPOINT = flags.DEFINE_string(
    'input_checkpoint',
    None,
    'Path to the input checkpoint directory to convert.',
    required=True,
)
_OUTPUT_CHECKPOINT = flags.DEFINE_string(
    'output_checkpoint',
    None,
    'Path to the output checkpoint directory where converted checkpoint is '
    'saved.',
    required=True,
)
_TARGET_NUM_DEVICES = flags.DEFINE_integer(
    'target_num_devices',
    4,
    'Number of TPU devices in the target topology.',
)
_TARGET_DEVICE_KIND = flags.DEFINE_string(
    'target_device_kind',
    'TPU v5',
    'Device kind of the target TPU topology.',
)
_TARGET_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'target_global_batch_size',
    None,
    'Optional target global batch size for the target topology.',
)


class FakeDevice:
  """A lightweight mock JAX device for offline topology construction."""

  def __init__(self, device_id: int, device_kind: str = 'TPU v5'):
    """Initializes a FakeDevice."""
    self.id = device_id
    self.process_index = 0
    self.slice_index = 0
    self.device_kind = device_kind
    self.platform = 'tpu'
    self.memory_kind = 'tpu_hbm'


def expand_directory_path(input_dir: str) -> str | None:
  """Expands the input directory if it refers to a test output directory."""
  testdir_key = 'TEST_UNDECLARED_OUTPUTS_DIR'
  testdir_env_var = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')

  if input_dir is None:
    return None

  # Leave absolute paths and URI schemes (like gs://) alone.
  if input_dir.startswith('/') or '://' in input_dir:
    return input_dir

  # Special handling for test output subdirectories.
  if input_dir.startswith(testdir_key + '/'):
    if not testdir_env_var:
      raise ValueError(f'{testdir_key} environment variable is not set.')
    expanded_dir = os.fspath(
        epath.Path(testdir_env_var) / input_dir[len(testdir_key + '/') :]
    )
    os.makedirs(expanded_dir, exist_ok=True)
    return expanded_dir

  # Special handling for writing to the test output directory.
  if input_dir == testdir_key:
    if not testdir_env_var:
      raise ValueError(f'{testdir_key} environment variable is not set.')
    return testdir_env_var

  return os.fspath(epath.Path(input_dir).resolve())


def run_conversion() -> None:
  """Executes general offline checkpoint conversion for the target TPU topology."""
  fake_devices = [
      FakeDevice(i, device_kind=_TARGET_DEVICE_KIND.value.replace('_', ' '))
      for i in range(_TARGET_NUM_DEVICES.value)
  ]
  num_sc_per_device = utils.num_sparsecores_per_device(fake_devices[0])

  input_path = expand_directory_path(_INPUT_CHECKPOINT.value)
  if input_path is not None and input_path.endswith('.tgz'):
    input_path = checkpoint_utils.decompress_checkpoint(input_path)

  checkpoint_utils.convert_cross_topology_checkpoint(
      input_checkpoint_path=input_path,
      output_checkpoint_path=expand_directory_path(_OUTPUT_CHECKPOINT.value),
      num_global_devices=len(fake_devices),
      num_sc_per_device=num_sc_per_device,
      target_batch_size=_TARGET_GLOBAL_BATCH_SIZE.value,
  )
  logging.info(
      'Exiting cleanly after offline cross-topology checkpoint conversion.'
  )


def main(argv: collections.abc.Sequence[str]) -> None:
  del argv  # Unused.
  run_conversion()


if __name__ == '__main__':
  app.run(main)
