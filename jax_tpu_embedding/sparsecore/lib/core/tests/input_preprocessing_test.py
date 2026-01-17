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
import dataclasses
import logging

from absl.testing import absltest
import jax
from jax_tpu_embedding.sparsecore.lib.core import constants
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
import numpy as np


@dataclasses.dataclass(frozen=True)
class MockDevice:
  id: int


class InputPreprocessingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.input_features = np.array(
        [
            [5, 18, 0, 20, 0, 2, 31, 3],
            [18, 0, 20, 6, 1, 28, 5, 8],
            [0, 20, 6, 15, 12, 7, 3, 11],
            [18, 0, 7, 3, 6, 4, 19, 2],
        ],
        dtype=np.int32,
    )
    self.input_weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        np.float32,
    )

    self.global_devices = np.array([
        MockDevice(id=0),
        MockDevice(id=1),
        MockDevice(id=2),
        MockDevice(id=3),
    ])

  def test_error_on_bad_input_dimensions_small(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    bad_input_tensor = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=np.int32,
    )
    bad_weights_tensor = np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        bad_input_tensor,
        bad_weights_tensor,
        mesh,
        2,  # num_sc_per_device
    )

  def test_error_on_bad_input_dimensions(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    bad_input_features = np.array(
        [
            [[5, 18, 0, 20, 0, 2, 31, 3]],
            [[18, 0, 20, 6, 1, 28, 5, 8]],
            [[0, 20, 6, 15, 12, 7, 3, 11]],
            [[18, 0, 7, 3, 6, 4, 19, 2]],
        ],
        dtype=np.int32,
    )
    bad_weights = np.array(
        [
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        ],
        np.float32,
    )
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        bad_input_features,
        bad_weights,
        mesh,
        2,  # num_sc_per_device
    )

  def test_error_on_bad_weights(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    bad_input_weights = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        object,
    )
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        self.input_features,
        bad_input_weights,
        mesh,
    )
    # Note that the first array is shorter.
    bad_input_weights_2 = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        object
    )
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        self.input_features,
        bad_input_weights_2,
        mesh,
    )

  def test_error_bad_mesh(self):
    mesh = jax.sharding.Mesh([], "x")
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        self.input_features,
        self.input_weights,
        mesh,
    )

  def test_error_non_mod_sharding(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    self.assertRaises(
        ValueError,
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        self.input_features,
        self.input_weights,
        mesh,
        64,
        4,
        "DIV",
    )

  def test_correct_input_preprocessing_single_column(self):
    input_tensor = np.array(
        [
            [5],
            [18],
            [0],
            [20],
            [6],
            [15],
            [12],
            [7],
            [3],
            [11],
            [8],
            [26],
            [0],
            [18],
            [7],
            [2],
        ],
        dtype=object,
    )
    input_weights = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=object,
    )
    mesh = jax.sharding.Mesh(self.global_devices[:1], "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        input_tensor,
        input_weights,
        mesh,
        num_sc_per_device=4,
        max_ids_per_partition=12,
    )
    logging.info(lhs_local_embedding_ids.shape)

    # Prepare expected outputs.
    expected_lhs_row_pointers = np.array(
        [
            2,
            9,
            17,
            24,
            24,
            24,
            24,
            24,
            1,
            8,
            9,
            18,
            24,
            24,
            24,
            24,
            1,
            8,
            9,
            18,
            24,
            24,
            24,
            24,
            1,
            8,
            10,
            17,
            24,
            24,
            24,
            24,
        ],
        dtype=np.int32,
    )

    expected_lhs_local_sample_ids = np.full(
        (16 * 4 * 4,),
        constants.PADDING_VALUE,
        dtype=np.int32,
    )
    expected_lhs_local_sample_ids[0] = 2
    expected_lhs_local_sample_ids[1] = 3
    expected_lhs_local_sample_ids[8] = 0
    expected_lhs_local_sample_ids[16] = 1
    expected_lhs_local_sample_ids[64] = 2
    expected_lhs_local_sample_ids[72] = 0
    expected_lhs_local_sample_ids[80] = 3
    expected_lhs_local_sample_ids[81] = 1
    expected_lhs_local_sample_ids[128] = 2
    expected_lhs_local_sample_ids[136] = 3
    expected_lhs_local_sample_ids[144] = 0
    expected_lhs_local_sample_ids[145] = 1
    expected_lhs_local_sample_ids[192] = 0
    expected_lhs_local_sample_ids[200] = 3
    expected_lhs_local_sample_ids[201] = 1
    expected_lhs_local_sample_ids[208] = 2

    expected_lhs_local_embedding_ids = np.full(
        (16 * 4 * 4,),
        constants.PADDING_VALUE,
        dtype=np.int32,
    )
    expected_lhs_local_embedding_ids[0] = 0
    expected_lhs_local_embedding_ids[1] = 5
    expected_lhs_local_embedding_ids[8] = 1
    expected_lhs_local_embedding_ids[16] = 4
    expected_lhs_local_embedding_ids[64] = 3
    expected_lhs_local_embedding_ids[72] = 1
    expected_lhs_local_embedding_ids[80] = 1
    expected_lhs_local_embedding_ids[81] = 3
    expected_lhs_local_embedding_ids[128] = 2
    expected_lhs_local_embedding_ids[136] = 6
    expected_lhs_local_embedding_ids[144] = 0
    expected_lhs_local_embedding_ids[145] = 2
    expected_lhs_local_embedding_ids[192] = 0
    expected_lhs_local_embedding_ids[200] = 0
    expected_lhs_local_embedding_ids[201] = 4
    expected_lhs_local_embedding_ids[208] = 1

    expected_lhs_gains = np.full(
        (16 * 4 * 4,),
        np.nan,
        dtype=np.float32,
    )
    expected_lhs_gains[0] = 1.0
    expected_lhs_gains[1] = 1.0
    expected_lhs_gains[8] = 1.0
    expected_lhs_gains[16] = 1.0
    expected_lhs_gains[64] = 1.0
    expected_lhs_gains[72] = 1.0
    expected_lhs_gains[80] = 1.0
    expected_lhs_gains[81] = 1.0
    expected_lhs_gains[128] = 1.0
    expected_lhs_gains[136] = 1.0
    expected_lhs_gains[144] = 1.0
    expected_lhs_gains[145] = 1.0
    expected_lhs_gains[192] = 1.0
    expected_lhs_gains[200] = 1.0
    expected_lhs_gains[201] = 1.0
    expected_lhs_gains[208] = 1.0

    # Compare the results.
    np.testing.assert_equal(lhs_row_pointers, expected_lhs_row_pointers)
    np.testing.assert_equal(lhs_local_sample_ids, expected_lhs_local_sample_ids)
    np.testing.assert_equal(
        lhs_local_embedding_ids, expected_lhs_local_embedding_ids
    )
    np.testing.assert_equal(lhs_gains, expected_lhs_gains)

  def test_correct_input_preprocessing_multiple_columns(self):
    mesh = jax.sharding.Mesh(self.global_devices[:1], "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_features,
        self.input_weights,
        mesh,
        num_sc_per_device=4,
        max_ids_per_partition=32,
    )
    with self.subTest(name="RowPointerEquality"):
      expected_lhs_row_pointers = np.array(
          [
              2,
              9,
              18,
              26,
              32,
              32,
              32,
              32,
              4,
              10,
              18,
              24,
              24,
              24,
              24,
              24,
              3,
              8,
              9,
              20,
              24,
              24,
              24,
              24,
              2,
              8,
              11,
              19,
              24,
              24,
              24,
              24,
          ],
          dtype=np.int32,
      )
      np.testing.assert_equal(lhs_row_pointers, expected_lhs_row_pointers)

    coo_buffer_size = 32 * 4 * 4
    with self.subTest(name="EmbeddingIdsEqaulity"):
      coo_buffer_size_per_sc = (
          coo_buffer_size // 4
      )

      expected_lhs_local_embedding_ids = np.full(
          (coo_buffer_size,),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_lhs_local_embedding_ids[0] = 0
      expected_lhs_local_embedding_ids[1] = 5
      expected_lhs_local_embedding_ids[8] = 1
      expected_lhs_local_embedding_ids[16] = 0
      expected_lhs_local_embedding_ids[17] = 4
      expected_lhs_local_embedding_ids[24] = 0
      expected_lhs_local_embedding_ids[25] = 7

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_local_embedding_ids[sc_1_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_1_start + 1] = 2
      expected_lhs_local_embedding_ids[sc_1_start + 2] = 5
      expected_lhs_local_embedding_ids[sc_1_start + 3] = 7
      expected_lhs_local_embedding_ids[sc_1_start + 8] = 0
      expected_lhs_local_embedding_ids[sc_1_start + 9] = 1
      expected_lhs_local_embedding_ids[sc_1_start + 16] = 1
      expected_lhs_local_embedding_ids[sc_1_start + 17] = 4

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_local_embedding_ids[sc_2_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_2_start + 1] = 3
      expected_lhs_local_embedding_ids[sc_2_start + 2] = 5
      expected_lhs_local_embedding_ids[sc_2_start + 8] = 1
      expected_lhs_local_embedding_ids[sc_2_start + 16] = 0
      expected_lhs_local_embedding_ids[sc_2_start + 17] = 1
      expected_lhs_local_embedding_ids[sc_2_start + 18] = 2
      expected_lhs_local_embedding_ids[sc_2_start + 19] = 3

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_local_embedding_ids[sc_3_start + 0] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 1] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 8] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 9] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 10] = 4
      expected_lhs_local_embedding_ids[sc_3_start + 16] = 0
      expected_lhs_local_embedding_ids[sc_3_start + 17] = 1
      expected_lhs_local_embedding_ids[sc_3_start + 18] = 4

      np.testing.assert_equal(
          lhs_local_embedding_ids, expected_lhs_local_embedding_ids
      )

    with self.subTest(name="SampleIdsEqaulity"):
      expected_lhs_local_sample_ids = np.full(
          (coo_buffer_size,),
          constants.PADDING_VALUE,
          dtype=np.int32,
      )
      expected_lhs_local_sample_ids[0] = 0
      expected_lhs_local_sample_ids[1] = 0
      expected_lhs_local_sample_ids[8] = 0
      expected_lhs_local_sample_ids[16] = 0
      expected_lhs_local_sample_ids[17] = 0
      expected_lhs_local_sample_ids[24] = 0
      expected_lhs_local_sample_ids[25] = 0

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_local_sample_ids[sc_1_start + 0] = 0
      expected_lhs_local_sample_ids[sc_1_start + 1] = 0
      expected_lhs_local_sample_ids[sc_1_start + 2] = 0
      expected_lhs_local_sample_ids[sc_1_start + 3] = 0
      expected_lhs_local_sample_ids[sc_1_start + 8] = 0
      expected_lhs_local_sample_ids[sc_1_start + 9] = 0
      expected_lhs_local_sample_ids[sc_1_start + 16] = 0
      expected_lhs_local_sample_ids[sc_1_start + 17] = 0

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_local_sample_ids[sc_2_start + 0] = 0
      expected_lhs_local_sample_ids[sc_2_start + 1] = 0
      expected_lhs_local_sample_ids[sc_2_start + 2] = 0
      expected_lhs_local_sample_ids[sc_2_start + 8] = 0
      expected_lhs_local_sample_ids[sc_2_start + 16] = 0
      expected_lhs_local_sample_ids[sc_2_start + 17] = 0
      expected_lhs_local_sample_ids[sc_2_start + 18] = 0
      expected_lhs_local_sample_ids[sc_2_start + 19] = 0

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_local_sample_ids[sc_3_start + 0] = 0
      expected_lhs_local_sample_ids[sc_3_start + 1] = 0
      expected_lhs_local_sample_ids[sc_3_start + 8] = 0
      expected_lhs_local_sample_ids[sc_3_start + 9] = 0
      expected_lhs_local_sample_ids[sc_3_start + 10] = 0
      expected_lhs_local_sample_ids[sc_3_start + 16] = 0
      expected_lhs_local_sample_ids[sc_3_start + 17] = 0
      expected_lhs_local_sample_ids[sc_3_start + 18] = 0

      np.testing.assert_equal(
          lhs_local_sample_ids, expected_lhs_local_sample_ids
      )

    with self.subTest(name="GainsEqualityTest"):
      coo_buffer_size_per_sc = coo_buffer_size // 4

      expected_lhs_gains = np.full(
          (coo_buffer_size,),
          np.nan,
          dtype=np.float32,
      )
      expected_lhs_gains[0] = 2.0
      expected_lhs_gains[1] = 1.0
      expected_lhs_gains[8] = 1.0
      expected_lhs_gains[16] = 1.0
      expected_lhs_gains[17] = 1.0
      expected_lhs_gains[24] = 1.0
      expected_lhs_gains[25] = 1.0

      sc_1_start = coo_buffer_size_per_sc
      expected_lhs_gains[sc_1_start + 0] = 1.0
      expected_lhs_gains[sc_1_start + 1] = 1.0
      expected_lhs_gains[sc_1_start + 2] = 1.0
      expected_lhs_gains[sc_1_start + 3] = 1.0
      expected_lhs_gains[sc_1_start + 8] = 1.0
      expected_lhs_gains[sc_1_start + 9] = 1.0
      expected_lhs_gains[sc_1_start + 16] = 1.0
      expected_lhs_gains[sc_1_start + 17] = 1.0

      sc_2_start = coo_buffer_size_per_sc * 2
      expected_lhs_gains[sc_2_start + 0] = 1.0
      expected_lhs_gains[sc_2_start + 1] = 1.0
      expected_lhs_gains[sc_2_start + 2] = 1.0
      expected_lhs_gains[sc_2_start + 8] = 1.0
      expected_lhs_gains[sc_2_start + 16] = 1.0
      expected_lhs_gains[sc_2_start + 17] = 1.0
      expected_lhs_gains[sc_2_start + 18] = 1.0
      expected_lhs_gains[sc_2_start + 19] = 1.0

      sc_3_start = coo_buffer_size_per_sc * 3
      expected_lhs_gains[sc_3_start + 0] = 1.0
      expected_lhs_gains[sc_3_start + 1] = 1.0
      expected_lhs_gains[sc_3_start + 8] = 1.0
      expected_lhs_gains[sc_3_start + 9] = 1.0
      expected_lhs_gains[sc_3_start + 10] = 1.0
      expected_lhs_gains[sc_3_start + 16] = 1.0
      expected_lhs_gains[sc_3_start + 17] = 1.0
      expected_lhs_gains[sc_3_start + 18] = 1.0

      np.testing.assert_equal(
          lhs_gains, expected_lhs_gains
      )


if __name__ == "__main__":
  absltest.main()
