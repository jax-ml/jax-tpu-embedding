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
"""Test for test utils."""

import logging

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)


class TestUtilsTest(parameterized.TestCase):

  def test_row_id_initializer(self):
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
        ],
        dtype=np.int32,
    )
    actual = test_utils.row_id_initializer(shape=(4, 5), dtype=np.int32)
    np.testing.assert_array_equal(expected, actual)

  def test_row_id_initializer_offset(self):
    expected = np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0, 13.0, 13.0],
        ],
        dtype=np.float32,
    )
    actual = test_utils.row_id_initializer(
        shape=(4, 5), offset=10, dtype=np.float32
    )
    np.testing.assert_array_equal(expected, actual)

  @parameterized.product(
      row_count=[3, 8, 123], col_count=[1, 4, 8], leading_value=[0, 1, 2]
  )
  def test_row_col_id_initializer(self, row_count, col_count, leading_value):
    shape = (row_count, col_count)
    initializer = test_utils.row_col_id_initializer(leading_value)
    table = initializer(jax.random.PRNGKey(0), shape)
    self.assertEqual(table.shape, shape)
    logging.info(
        "Row col id initialized table: shape: %s, content:%s",
        table.shape,
        test_utils.formatted_array2string(table),
    )
    process_index_part = jax.process_index() / 1000000
    for i in range(shape[0]):
      expected_row_base_value = leading_value + i / 1000
      for j in range(shape[1]):
        if j == 0:
          remaining_part = process_index_part
        else:
          remaining_part = j / 1000000
        self.assertEqual(table[i, j], expected_row_base_value + remaining_part)

  def test_stacking_simple(self):
    a = test_utils.row_id_initializer(shape=(4, 3), offset=10)
    b = test_utils.row_id_initializer(shape=(2, 3), offset=50)
    aa = test_utils.create_per_device_sharded_stacked_tables(
        [a, b], num_devices=2, num_sparsecore_per_device=1, rotation=1
    )
    np.testing.assert_array_equal(
        aa,
        np.array([
            [
                np.array([10, 10, 10], dtype=np.int32),
                np.array([12, 12, 12], dtype=np.int32),
                np.array([51, 51, 51], dtype=np.int32),
            ],
            [
                np.array([11, 11, 11], dtype=np.int32),
                np.array([13, 13, 13], dtype=np.int32),
                np.array([50, 50, 50], dtype=np.int32),
            ],
        ]),
    )

  def test_stacking_no_rotation(self):
    a = test_utils.row_id_initializer(shape=(8, 4))
    b = test_utils.row_id_initializer(shape=(16, 4), offset=100)
    c = test_utils.row_id_initializer(shape=(16, 4), offset=500)
    num_devices = 4
    num_sparsecore_per_device = 2
    stacked = test_utils.create_per_device_sharded_stacked_tables(
        [a, b, c], num_devices, num_sparsecore_per_device, rotation=0
    )
    expected = einops.rearrange(
        np.concatenate(
            [a, b, c],
            axis=0,
        ),
        "(v c s) f -> c (s v) f",
        c=num_devices,
        s=num_sparsecore_per_device,
    )
    np.testing.assert_array_equal(stacked, expected)

  def test_stacking_with_rotation(self):
    a = test_utils.row_id_initializer(shape=(8, 4))
    b = test_utils.row_id_initializer(shape=(16, 4), offset=100)
    c = test_utils.row_id_initializer(shape=(16, 4), offset=500)
    num_devices = 4
    num_sparsecore_per_device = 2
    rotation = 2
    stacked = test_utils.create_per_device_sharded_stacked_tables(
        [a, b, c], num_devices, num_sparsecore_per_device, rotation
    )
    np.testing.assert_array_equal(
        np.array([
            [
                np.array([0, 0, 0, 0], dtype=np.int32),
                np.array([106, 106, 106, 106], dtype=np.int32),
                np.array([114, 114, 114, 114], dtype=np.int32),
                np.array([504, 504, 504, 504], dtype=np.int32),
                np.array([512, 512, 512, 512], dtype=np.int32),
                np.array([1, 1, 1, 1], dtype=np.int32),
                np.array([107, 107, 107, 107], dtype=np.int32),
                np.array([115, 115, 115, 115], dtype=np.int32),
                np.array([505, 505, 505, 505], dtype=np.int32),
                np.array([513, 513, 513, 513], dtype=np.int32),
            ],
            [
                np.array([2, 2, 2, 2], dtype=np.int32),
                np.array([100, 100, 100, 100], dtype=np.int32),
                np.array([108, 108, 108, 108], dtype=np.int32),
                np.array([506, 506, 506, 506], dtype=np.int32),
                np.array([514, 514, 514, 514], dtype=np.int32),
                np.array([3, 3, 3, 3], dtype=np.int32),
                np.array([101, 101, 101, 101], dtype=np.int32),
                np.array([109, 109, 109, 109], dtype=np.int32),
                np.array([507, 507, 507, 507], dtype=np.int32),
                np.array([515, 515, 515, 515], dtype=np.int32),
            ],
            [
                np.array([4, 4, 4, 4], dtype=np.int32),
                np.array([102, 102, 102, 102], dtype=np.int32),
                np.array([110, 110, 110, 110], dtype=np.int32),
                np.array([500, 500, 500, 500], dtype=np.int32),
                np.array([508, 508, 508, 508], dtype=np.int32),
                np.array([5, 5, 5, 5], dtype=np.int32),
                np.array([103, 103, 103, 103], dtype=np.int32),
                np.array([111, 111, 111, 111], dtype=np.int32),
                np.array([501, 501, 501, 501], dtype=np.int32),
                np.array([509, 509, 509, 509], dtype=np.int32),
            ],
            [
                np.array([6, 6, 6, 6], dtype=np.int32),
                np.array([104, 104, 104, 104], dtype=np.int32),
                np.array([112, 112, 112, 112], dtype=np.int32),
                np.array([502, 502, 502, 502], dtype=np.int32),
                np.array([510, 510, 510, 510], dtype=np.int32),
                np.array([7, 7, 7, 7], dtype=np.int32),
                np.array([105, 105, 105, 105], dtype=np.int32),
                np.array([113, 113, 113, 113], dtype=np.int32),
                np.array([503, 503, 503, 503], dtype=np.int32),
                np.array([511, 511, 511, 511], dtype=np.int32),
            ],
        ]),
        stacked,
    )


if __name__ == "__main__":
  absltest.main()
