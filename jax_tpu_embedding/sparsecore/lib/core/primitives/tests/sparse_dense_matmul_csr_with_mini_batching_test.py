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
import logging
from unittest import mock

from absl.testing import absltest
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr_with_mini_batching
import numpy as np


def _generate_input_tensors_and_expected_output_for_2_batches(
    max_mini_batch_size=0, output_padded_width=0):
  batch_size = 32
  num_minibatches_per_physical_sparse_core = 2

  # The inputs are repeated twice, so the resulting embedding
  # activations also repeated twice.
  # sample 0
  #         [5],
  #         [3],
  #         [9],
  #         [1],
  #         [6],
  #         [12],
  #         [0],
  #         [4],
  #         [15],
  #         [13],
  #         [11],
  #         [7],
  #         [8],
  #         [14],
  #         [2],
  #         [10],
  # sample 16
  #         [5],
  #         [3],
  #         [9],
  #         [1],
  #         [6],
  #         [12],
  #         [0],
  #         [4],
  #         [15],
  #         [13],
  #         [11],
  #         [7],
  #         [8],
  #         [14],
  #         [2],
  #         [10],
  # For mini-batching, we choose to cut the vocabulary dimension into two
  # buckets.
  # bucket 0: embedding ids 0-7
  # bucket 1: embedding ids 8-31

  # Process the input.
  # Each destination SC takes data from 4 sourceSCs (including itself, there
  # are 4 SCs in this test setup), padded to 8 multiples. The values
  # represent numbers of valid entries in each section in the embedding_ids
  # and sample_ids arrays.
  # It actually points the beginning of the next section, and each section
  # length must be multiple of 8. Repeating the same value indicates 0, no
  # data from this SC. Each section must be padded to 8 multiples, so the
  # beginning of each SC must be the next multples of 8.  Note that 8 is the
  # width of sparsecore tile register, which is 8 for VFC and GLC, but is
  # 16 for GFC.
  lhs_row_pointers = jnp.array([
      # SC0 mini batch 0 taking data from SCs: 2, 2, 1, 1, padding
      2, 10, 17, 25, 25, 25, 25, 25,
      # SC0 mini batch 1 taking data from SCs: 1, 1, 0, 0, padding
      33, 41, 41, 41, 41, 41, 41, 41,

      # SC1 mini batch 0 taking data from SCs: 0, 0, 1, 1, padding
      48, 48, 49, 57, 57, 57, 57, 57,
      # SC1 mini batch 1 taking data from SCs: 1, 1, 2, 2, padding
      65, 73, 82, 90, 90, 90, 90, 90,

      # SC2 mini batch 0 taking data from SCs: 2, 2, 1, 1, padding
      98, 106, 113, 121, 121, 121, 121, 121,
      # SC2 mini batch 1 taking data from SCs: 1, 1, 0, 0, padding
      129, 137, 137, 137, 137, 137, 137, 137,

      # SC3 mini batch 0 taking data from SCs: 0, 0, 1, 1, padding
      144, 144, 145, 153, 153, 153, 153, 153,
      # SC3 mini batch 1 taking data from SCs: 1, 1, 2, 2, padding
      161, 169, 178, 186, 186, 186, 186, 186,
  ])
  imax = 2147483647

  # Each section represents the row pointers from a source SC. Number of
  # entries in each section must be multiple of 8, and lower than
  # max_ids_per_partition. The number of unique values in each section must
  # be lower than max_unique_ids_per_partition. Note that 8 is the
  # width of sparsecore tile register, which is 8 for VFC and GLC, but is
  # 16 for GFC.
  lhs_local_embedding_ids = jnp.array([
      # SC 0 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      0, 1, imax, imax, imax, imax, imax, imax,
      # SC 1 rows 0, 1: 1, 5
      0, 1, imax, imax, imax, imax, imax, imax,
      # SC 2 row 1: 6
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 0: 3
      0, imax, imax, imax, imax, imax, imax, imax,
      # SC 0 mini batch 1 taking data from
      # SC 0 row 3: 12
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 2: 9
      2, imax, imax, imax, imax, imax, imax, imax,

      # SC 1 mini batch 0 taking data from
      # SC 2 row 0: 2
      0, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 1: 7
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 mini batch 1 taking data from
      # SC 0 row 2: 8
      2, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 3: 13
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 2 rows 2, 3: 10, 14
      2, 3, imax, imax, imax, imax, imax, imax,
      # SC 3 rows 2, 3: 11, 15
      2, 3, imax, imax, imax, imax, imax, imax,

      # SC 2 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      0, 1, imax, imax, imax, imax, imax, imax,
      # SC 1 rows 0, 1: 1, 5
      0, 1, imax, imax, imax, imax, imax, imax,
      # SC 2 row 1: 6
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 0: 3
      0, imax, imax, imax, imax, imax, imax, imax,
      # SC 2 mini batch 1 taking data from
      # SC 0 row 3: 12
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 2: 9
      2, imax, imax, imax, imax, imax, imax, imax,

      # SC 3 mini batch 0 taking data from
      # SC 2 row 0: 2
      0, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 1: 7
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 mini batch 1 taking data from
      # SC 0 rows 2: 8
      2, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 3: 13
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 2 rows 2, 3: 10, 14
      2, 3, imax, imax, imax, imax, imax, imax,
      # SC 3 rows 2, 3: 11, 15
      2, 3, imax, imax, imax, imax, imax, imax,
  ])

  # Each section represents the row pointers into a destination SC.
  lhs_local_sample_ids = jnp.array([
      # SC 0 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      6, 7, imax, imax, imax, imax, imax, imax,
      # SC 1 rows 0, 1: 1, 5
      3, 0, imax, imax, imax, imax, imax, imax,
      # SC 2 row 1: 6
      4, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 0: 3
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 0 mini batch 1 taking data from
      # SC 0 row 3: 12
      5, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 2: 9
      2, imax, imax, imax, imax, imax, imax, imax,

      # SC 1 mini batch 0 taking data from
      # SC 2 row 0: 2
      6, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 1: 7
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 mini batch 1 taking data from
      # SC 0 rows 2: 8
      4, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 3: 13
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 2 rows 2, 3: 10, 14
      7, 5, imax, imax, imax, imax, imax, imax,
      # SC 3 rows 2, 3: 11, 15
      2, 0, imax, imax, imax, imax, imax, imax,

      # SC 2 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      6, 7, imax, imax, imax, imax, imax, imax,
      # SC 1 rows 0, 1: 1, 5
      3, 0, imax, imax, imax, imax, imax, imax,
      # SC 2 row 1: 6
      4, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 0: 3
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 0 mini batch 1 taking data from
      # SC 0 row 3: 12
      5, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 2: 9
      2, imax, imax, imax, imax, imax, imax, imax,

      # SC3 mini batch 0 taking data from
      # SC 2 row 0: 2
      6, imax, imax, imax, imax, imax, imax, imax,
      # SC 3 row 1: 7
      3, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 mini batch 1 taking data from
      # SC 0 rows 2: 8
      4, imax, imax, imax, imax, imax, imax, imax,
      # SC 1 row 3: 13
      1, imax, imax, imax, imax, imax, imax, imax,
      # SC 2 rows 2, 3: 10, 14
      7, 5, imax, imax, imax, imax, imax, imax,
      # SC 3 rows 2, 3: 11, 15
      2, 0, imax, imax, imax, imax, imax, imax,
  ])

  nan = float("nan")
  # All weights are 1.0
  lhs_gains = jnp.array([
      # SC 0 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 1 rows 0, 1: 1, 5
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 2 row 1: 6
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 3 row 0: 3
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 0 mini batch 1 taking data from
      # SC 0 row 3: 12
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 row 2: 9
      1, nan, nan, nan, nan, nan, nan, nan,

      # SC 1 mini batch 0 taking data from
      # SC 2 row 0: 2
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 3 row 1: 7
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 mini batch 1 taking data from
      # SC 0 rows 2: 8
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 row 3: 13
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 2 rows 2, 3: 10, 14
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 3 rows 2, 3: 11, 15
      1, 1, nan, nan, nan, nan, nan, nan,

      # SC 2 mini batch 0 taking data from
      # SC 0 rows 0, 1: 0, 4
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 1 rows 0, 1: 1, 5
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 2 row 1: 6
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 3 row 0: 3
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 0 mini batch 1 taking data from
      # SC 0 row 3: 12
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 row 2: 9
      1, nan, nan, nan, nan, nan, nan, nan,

      # SC3 mini batch 0 taking data from
      # SC 2 row 0: 2
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 3 row 1: 7
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 mini batch 1 taking data from
      # SC 0 rows 2: 8
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 1 row 3: 13
      1, nan, nan, nan, nan, nan, nan, nan,
      # SC 2 rows 2, 3: 10, 14
      1, 1, nan, nan, nan, nan, nan, nan,
      # SC 3 rows 2, 3: 11, 15
      1, 1, nan, nan, nan, nan, nan, nan,
  ])

  if max_mini_batch_size != 0:
    row_pointer_width = max_mini_batch_size * 8 * 4
    lhs_row_pointers = jnp.pad(
        lhs_row_pointers,
        (0, row_pointer_width - lhs_row_pointers.shape[0]),
        mode="constant",
        constant_values=lhs_row_pointers[63])

  if output_padded_width != 0:
    lhs_local_embedding_ids = jnp.pad(
        lhs_local_embedding_ids,
        (0, output_padded_width - lhs_local_embedding_ids.shape[0]),
        mode="constant",
        constant_values=imax,
    )
    lhs_local_sample_ids = jnp.pad(
        lhs_local_sample_ids,
        (0, output_padded_width - lhs_local_sample_ids.shape[0]),
        mode="constant",
        constant_values=imax,
    )
    lhs_gains = jnp.pad(
        lhs_gains,
        (0, output_padded_width - lhs_gains.shape[0]),
        mode="constant",
        constant_values=nan,
    )

  logging.debug("lhs_row_pointers: %s", lhs_row_pointers)
  logging.debug("lhs_local_embedding_ids: %s", lhs_local_embedding_ids)
  logging.debug("lhs_local_sample_ids: %s", lhs_local_sample_ids)
  logging.debug("lhs_gains: %s", lhs_gains)

  assert lhs_local_sample_ids.shape == lhs_local_embedding_ids.shape
  assert lhs_gains.shape == lhs_local_embedding_ids.shape

  expected_emb_activations = np.array(
      [
          [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
          [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

          [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
          [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],

          [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
          [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
          [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
          [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],

          [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
          [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
          [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],

          [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
          [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
          [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

          [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
          [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],

          [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
          [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
          [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
          [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],

          [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
          [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
          [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
      ],
      dtype=np.float32,
  )

  return (
      batch_size,
      num_minibatches_per_physical_sparse_core,
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      expected_emb_activations,
  )


class SparseDenseMatmulCsrWithMiniBatchingValidationTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    jax.config.update("jax_traceback_filtering", "off")

    self.num_chips = 1
    self.vocab_size = 32
    self.emb_size = 8
    self.input_tensor = np.array(
        [
            [5],
            [3],
            [9],
            [1],
            [6],
            [12],
            [0],
            [4],
            [15],
            [13],
            [11],
            [7],
            [8],
            [14],
            [2],
            [10],
        ],
        dtype=np.int32,
    )
    self.input_weights = np.array(
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
        dtype=np.float32,
    )

    # Define the embedding table.
    self.emb_table = (
        np.array(
            [[i for _ in range(self.emb_size)] for i in range(self.vocab_size)]
        )
        .reshape(self.vocab_size, self.emb_size)
        .astype(np.float32)
    )
    self.global_devices = np.array([mock.create_autospec(jax.Device)])

    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )

    self.tpu_sparse_dense_matmul_csr_with_mini_batching = jax.named_call(
        sparse_dense_matmul_csr_with_mini_batching.tpu_sparse_dense_matmul_csr_with_mini_batching_primitive.bind,
        name="tpu_sparse_dense_matmul_csr_with_mini_batching",
    )

  def test_sc_emb_forward_pass_invalid_input_dtypes(self):
    batch_size = 16
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor, self.input_weights, mesh, max_ids_per_partition=16
    )

    num_minibatches_per_physical_sparse_core = 1

    with self.subTest("invalid_row_pointer_type"):
      bad_row_pointers = np.array(lhs_row_pointers, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          bad_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

    with self.subTest("invalid_local_embedding_ids_type"):
      bad_local_embedding_ids = np.array(
          lhs_local_embedding_ids, dtype=np.float32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          bad_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

    with self.subTest("invalid_local_sample_ids_type"):
      bad_local_sample_ids = np.array(lhs_local_sample_ids, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

    with self.subTest("invalid_gains_type"):
      bad_gains = np.array(lhs_gains, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          bad_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

    with self.subTest("invalid_emb_table_type"):
      bad_emb_table = np.array(self.emb_table, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          bad_emb_table,
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

  def test_sc_emb_forward_pass_invalid_input_shapes(self):
    batch_size = 16
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor, self.input_weights, mesh, max_ids_per_partition=16
    )

    num_minibatches_per_physical_sparse_core = 1
    with self.subTest("invalid_sample_id_shape"):
      bad_sample_id = jnp.full(
          (len(lhs_local_sample_ids) - 1,), 2, dtype=np.int32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_sample_id,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
      )

  def test_sc_emb_forward_pass_invalid_max_ids_per_partition(self):
    batch_size = 16
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor, self.input_weights, mesh, max_ids_per_partition=16
    )

    num_minibatches_per_physical_sparse_core = 1
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=0,
        max_unique_ids_per_partition=256,
        sharding_strategy=1,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=0,
        sharding_strategy=1,
    )

  def test_sc_emb_forward_pass_invalid_sharding(self):
    batch_size = 16
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor, self.input_weights, mesh, max_ids_per_partition=16
    )

    num_minibatches_per_physical_sparse_core = 1
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=256,
        sharding_strategy=2,
    )

  def test_sc_emb_forward_pass(self):
    # Process the input.
    batch_size = 16

    lhs_row_pointers = jnp.array([
        # SC0 taking data from SCs: 0, 3, 0, 1, padding
        0, 3, 3, 9, 9, 9, 9, 9,

        # SC1 taking data from SCs: 3, 0, 1, 0, padding
        19, 19, 25, 25, 25, 25, 25, 25,

        # SC2 taking data from SCs: 0, 1, 0, 3, padding
        32, 33, 33, 43, 43, 43, 43, 43,

        # SC3 taking data from SCs: 1, 0, 3, 0, padding
        49, 49, 59, 59, 59, 59, 59, 59,
    ])
    imax = 2147483647

    embedding_ids = jnp.array([
        # SC0 taking data from SC 1 rows 0, 1, 2, and then SC 3 row 0
        0, 1, 2, imax, imax, imax, imax, imax,
        0, imax, imax, imax, imax, imax, imax, imax,

        # SC1 taking data from SC 0 rows 0, 1, 3, and then SC 2 row 1
        0, 1, 3, imax, imax, imax, imax, imax,
        1, imax, imax, imax, imax, imax, imax, imax,

        # SC 2 taking data from SC 1 rows 3, and then SC 3 rows 1, 2, 3
        3, imax, imax, imax, imax, imax, imax, imax,
        1, 2, 3, imax, imax, imax, imax, imax,

        # SC 3 taking data from SC 0 rows 2, and then SC 2 rows 0, 2, 3
        2, imax, imax, imax, imax, imax, imax, imax,
        0, 2, 3, imax, imax, imax, imax, imax])

    sample_ids = jnp.array([
        # SC0 taking data from SC 1 should be summed into
        # resulting activations rows 3, 0, 2, and then SC 3 into row 1
        3, 0, 2, imax, imax, imax, imax, imax,
        1, imax, imax, imax, imax, imax, imax, imax,

        # SC1 taking data from SC 0 should be summed into
        # resulting activations rows 2, 3, 1, and then SC 2 into row 0
        2, 3, 1, imax, imax, imax, imax, imax,
        0, imax, imax, imax, imax, imax, imax, imax,

        # SC2 taking data from SC 1 should be summed into
        # resulting activations row 1, and then SC 3 into rows 3, 2, 0
        1, imax, imax, imax, imax, imax, imax, imax,
        3, 2, 0, imax, imax, imax, imax, imax,

        # SC3 taking data from SC 0 should be summed into
        # resulting activations row 0, and then SC 2 into rows 2, 3, 1
        0, imax, imax, imax, imax, imax, imax, imax,
        2, 3, 1, imax, imax, imax, imax, imax])

    nan = float("nan")
    # All weights are 1.0
    gains = jnp.array([
        1., 1., 1., nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan, 1.,
        nan, nan, nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan])

    logging.debug("lhs_row_pointers: %s", lhs_row_pointers)
    logging.debug("lhs_local_embedding_ids: %s", embedding_ids)
    logging.debug("lhs_local_sample_ids: %s", sample_ids)
    logging.debug("lhs_gains: %s", gains)

    assert(sample_ids.shape == embedding_ids.shape)
    assert(gains.shape == embedding_ids.shape)

    num_minibatches_per_physical_sparse_core = 1
    logging.debug(
        "num_minibatches_per_physical_sparse_core: %d",
        num_minibatches_per_physical_sparse_core,
    )

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
    )

    logging.debug("emb_activations: %s", emb_activations)

    # Check the embedding activations.
    expected_emb_activations = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],

            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],

            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(emb_activations, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core(self):
    (
        batch_size,
        num_minibatches_per_physical_sparse_core,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        expected_emb_activations,
    ) = _generate_input_tensors_and_expected_output_for_2_batches()

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
    )

    logging.debug("emb_activations: %s", emb_activations)

    # Check the embedding activations.
    np.testing.assert_equal(emb_activations, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core_with_padded_tensors(self):
    (
        batch_size,
        num_minibatches_per_physical_sparse_core,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        expected_emb_activations,
    ) = _generate_input_tensors_and_expected_output_for_2_batches(4, 256)

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
    )

    logging.debug("emb_activations: %s", emb_activations)
    np.testing.assert_equal(emb_activations, expected_emb_activations)

if __name__ == "__main__":
  absltest.main()
