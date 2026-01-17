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
"""Test utils for sparsecore input preprocessing."""

import numpy as np


def assert_equal_coo_buffer(
    local_device_count: int,
    num_sc_per_device: int,
    row_pointers: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
):
  """Compare COO buffers ignoring end of SC/device padding."""
  local_sc_count = local_device_count * num_sc_per_device

  # Ignore leading dim
  row_pointers = row_pointers.reshape(-1)
  actual = actual.reshape(-1)
  expected = expected.reshape(-1)

  for row_pointer_slice, actual_sc_slice, expected_sc_slice in zip(
      np.split(row_pointers, local_sc_count),
      np.split(actual, local_sc_count),
      np.split(expected, local_sc_count),
  ):
    np.testing.assert_almost_equal(
        actual_sc_slice[: row_pointer_slice[-1]],
        expected_sc_slice[: row_pointer_slice[-1]],
        decimal=6,
    )
