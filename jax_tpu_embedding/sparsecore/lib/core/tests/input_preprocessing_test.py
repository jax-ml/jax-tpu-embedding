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
        num_sc_per_device=2,
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
        num_sc_per_device=2,
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
        max_ids_per_partition=64,
        num_sc_per_device=4,
        sharding_strategy="DIV",
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

    coo_buffer_size = 32 * 4 * 4
    coo_buffer_size_per_sc = coo_buffer_size // 4

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

    with self.subTest(name="RowPointerEquality"):
      np.testing.assert_equal(lhs_row_pointers, expected_lhs_row_pointers)

    with self.subTest(name="EmbeddingIdsEqaulity"):
      np.testing.assert_equal(
          lhs_local_embedding_ids, expected_lhs_local_embedding_ids
      )

    with self.subTest(name="SampleIdsEqaulity"):
      np.testing.assert_equal(
          lhs_local_sample_ids, expected_lhs_local_sample_ids
      )

    with self.subTest(name="GainsEqualityTest"):
      np.testing.assert_equal(
          lhs_gains, expected_lhs_gains
      )


class InputPreprocessingMinibatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.global_devices = np.array([MockDevice(id=i) for i in range(4)])

  def test_single_minibatch_parity(self):
    """Verifies that [[samples]] is equivalent to [samples]."""
    features = np.array([[5, 18, 0], [18, 0, 6]], dtype=np.int32)
    weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    mesh = jax.sharding.Mesh(self.global_devices, "x")

    # Standard input
    standard_out = input_preprocessing.preprocess_sparse_dense_matmul_input(
        features,
        weights,
        mesh,
        num_sc_per_device=2,
        max_ids_per_partition=16,
    )

    # Minibatched input (nested list)
    minibatched_out = (
        input_preprocessing.preprocess_sparse_dense_matmul_input_minibatched(
            [features.tolist()],
            [weights.tolist()],
            mesh,
            num_sc_per_device=2,
            max_ids_per_partition=16,
        )
    )

    for actual, expected in zip(minibatched_out, standard_out):
      np.testing.assert_array_equal(actual, expected)

  def test_multi_minibatch_exact_contents(self):
    # 4 Samples per minibatch, 2 SCs per chip.
    # So Sample 0,1 -> SC 0, Sample 2,3 -> SC 1.
    mb1_feat = [[5], [3], [9, 1], [6, 12, 0]]
    mb2_feat = [[4], [15, 13, 11], [7, 8, 14, 2], [10]]
    mb1_weight = [[1.0], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]]
    mb2_weight = [[1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0]]

    # MB1 Sample 0 (SC 0, row 0): id   5 (Col 2, Partition 1)
    # MB1 Sample 1 (SC 0, row 1): id   3 (Col 1, Partition 1)
    # MB1 Sample 2 (SC 1, row 0): ids  9 (Col 4, Partition 1),
    #                                  1 (Col 0, Partition 1)
    # MB1 Sample 3 (SC 1, row 1): ids  6 (Col 3, Partition 0),
    #                                 12 (Col 6, Partition 0),
    #                                  0 (Col 0, Partition 0)
    # MB2 Sample 0 (SC 0, row 0): id   4 (Col 2, Partition 0)
    # MB2 Sample 1 (SC 0, row 1): ids 15 (Col 7, Partition 1),
    #                                 13 (Col 6, Partition 1),
    #                                 11 (Col 5, Partition 1)
    # MB2 Sample 2 (SC 1, row 0): ids  7 (Col 3, Partition 1),
    #                                  8 (Col 4, Partition 0),
    #                                 14 (Col 7, Partition 0),
    #                                  2 (Col 1, Partition 0)
    # MB2 Sample 3 (SC 1, row 1): id  10 (Col 5, Partition 0)

    features = [mb1_feat, mb2_feat]
    weights = [mb1_weight, mb2_weight]
    mesh = jax.sharding.Mesh(self.global_devices[:1], "x")

    out = input_preprocessing.preprocess_sparse_dense_matmul_input_minibatched(
        features,
        weights,
        mesh,
        num_sc_per_device=2,
        max_ids_per_partition=16,
    )

    row_pointers, col_ids, row_ids, gains = out

    with self.subTest(name="RowPointersEquality"):
      expected_rp = [
          # SC 0
          0,  # Ptr0 (MB1 Partition 0): len 0, next starts at 0
          2,  # Ptr1 (MB1 Partition 1): len 2, next starts at 8
          9,  # Ptr2 (MB2 Partition 0): len 1, next starts at 16
          19,  # Ptr3 (MB2 Partition 1): len 3, next starts at 24
          24,
          24,
          24,
          24,  # Padding to minimum 8 row pointers per SC
          # SC 1
          3,  # Ptr0 (MB1 Partition 0): len 3, next starts at 8
          10,  # Ptr1 (MB1 Partition 1): len 2, next starts at 16
          20,  # Ptr2 (MB2 Partition 0): len 4, next starts at 24
          25,  # Ptr3 (MB2 Partition 1): len 1, next starts at 32
          32,
          32,
          32,
          32,  # Padding to minimum 8 row pointers per SC
      ]
      np.testing.assert_array_equal(row_pointers, np.array(expected_rp))

    # 2. Complete Buffer Verification
    # Buffer size: rounded(max_ids*2)*scs*scs_per_device = 32 * 2 * 2 = 128
    # Buffer Size per SC = 64. Offset SC 1 = 64.
    expected_col_ids = np.full((2, 64), constants.PADDING_VALUE, np.int32)
    expected_row_ids = np.full((2, 64), constants.PADDING_VALUE, np.int32)
    expected_gains = np.full((2, 64), np.nan, np.float32)

    # fmt: off
    # SC 0
    # MB1 Partition 1
    expected_col_ids[0, 0], expected_row_ids[0, 0], expected_gains[0, 0] = 1, 1, 1.0
    expected_col_ids[0, 1], expected_row_ids[0, 1], expected_gains[0, 1] = 2, 0, 1.0
    # MB2 Partition 0
    expected_col_ids[0, 8], expected_row_ids[0, 8], expected_gains[0, 8] = 2, 0, 1.0
    # MB2 Partition 1
    expected_col_ids[0, 16], expected_row_ids[0, 16], expected_gains[0, 16] = 5, 1, 1.0
    expected_col_ids[0, 17], expected_row_ids[0, 17], expected_gains[0, 17] = 6, 1, 1.0
    expected_col_ids[0, 18], expected_row_ids[0, 18], expected_gains[0, 18] = 7, 1, 1.0

    # SC 1
    # MB1 Partition 0
    expected_col_ids[1, 0], expected_row_ids[1, 0], expected_gains[1, 0] = 0, 1, 1.0
    expected_col_ids[1, 1], expected_row_ids[1, 1], expected_gains[1, 1] = 3, 1, 1.0
    expected_col_ids[1, 2], expected_row_ids[1, 2], expected_gains[1, 2] = 6, 1, 1.0
    # MB1 Partition 1
    expected_col_ids[1, 8], expected_row_ids[1, 8], expected_gains[1, 8] = 0, 0, 1.0
    expected_col_ids[1, 9], expected_row_ids[1, 9], expected_gains[1, 9] = 4, 0, 1.0
    # MB2 Partition 0
    expected_col_ids[1, 16], expected_row_ids[1, 16], expected_gains[1, 16] = 1, 0, 1.0
    expected_col_ids[1, 17], expected_row_ids[1, 17], expected_gains[1, 17] = 4, 0, 1.0
    expected_col_ids[1, 18], expected_row_ids[1, 18], expected_gains[1, 18] = 5, 1, 1.0
    expected_col_ids[1, 19], expected_row_ids[1, 19], expected_gains[1, 19] = 7, 0, 1.0
    # MB2 Partition 1
    expected_col_ids[1, 24], expected_row_ids[1, 24], expected_gains[1, 24] = 3, 0, 1.0
    # fmt: on

    with self.subTest(name="ColIdsEquality"):
      np.testing.assert_array_equal(col_ids, expected_col_ids.flatten())

    with self.subTest(name="RowIdsEquality"):
      np.testing.assert_array_equal(row_ids, expected_row_ids.flatten())

    with self.subTest(name="GainsEquality"):
      np.testing.assert_allclose(gains, expected_gains.flatten(), atol=1e-6)

  def test_deduplication_with_summing_gains(self):
    # col 18 mod 4 = 2.
    features = [[[18, 18], [0]]]
    weights = [[[1.0, 2.0], [3.0]]]
    mesh = jax.sharding.Mesh(self.global_devices[:1], "x")

    out = input_preprocessing.preprocess_sparse_dense_matmul_input_minibatched(
        features,
        weights,
        mesh,
        num_sc_per_device=1,
        max_ids_per_partition=16,
    )
    row_pointers, col_ids, row_ids, gains = out

    expected_rp = np.array([2, 8, 8, 8, 8, 8, 8, 8])

    # Buffers size 16.
    expected_col_ids = np.full((16,), constants.PADDING_VALUE, np.int32)
    expected_row_ids = np.full((16,), constants.PADDING_VALUE, np.int32)
    expected_gains = np.full((16,), np.nan, np.float32)

    # Sorted by (Col, Row): (0, 1), then (18, 0)
    expected_col_ids[0], expected_row_ids[0], expected_gains[0] = 0, 1, 3.0
    expected_col_ids[1], expected_row_ids[1], expected_gains[1] = 18, 0, 3.0

    with self.subTest(name="RowPointersEquality"):
      # row_pointers size 8, ptr0 = 2, padded to 8 for remaining.
      np.testing.assert_array_equal(row_pointers, expected_rp)

    with self.subTest(name="ColIdsEquality"):
      np.testing.assert_array_equal(col_ids, expected_col_ids)

    with self.subTest(name="RowIdsEquality"):
      np.testing.assert_array_equal(row_ids, expected_row_ids)

    with self.subTest(name="GainsEquality"):
      np.testing.assert_allclose(gains, expected_gains, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
