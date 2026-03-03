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
from unittest import mock

from absl import logging
from absl.testing import absltest
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd
import numpy as np


def _pad_input_tensors(
    lhs_row_pointers: jnp.ndarray,
    lhs_local_embedding_ids: jnp.ndarray,
    lhs_local_sample_ids: jnp.ndarray,
    lhs_gains: jnp.ndarray,
    max_mini_batch_size: int,
    output_padded_width: int,
):
  # mini_batch_size * max(num_scs, 8) * num_sc_per_device
  row_pointer_width = max_mini_batch_size * 8 * 4
  imax = 2147483647
  nan = float("nan")
  lhs_row_pointers = jnp.pad(
      lhs_row_pointers,
      (0, row_pointer_width - lhs_row_pointers.shape[0]),
      mode="constant",
      constant_values=lhs_row_pointers[63])

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
  return (
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
  )


class SparseDenseMatmulGradWithSgdWithMiniBatchingTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    jax.config.update("jax_traceback_filtering", "off")

    self.num_chips = 1
    self.vocab_size = 32
    self.emb_size = 8

    # This is to shape the gradient tensor for backward pass.
    # The dimension needs to be the same for all test cases to avoid
    # recompilation.
    self.max_device_batch_size = 32

    # Define the embedding table.
    self.emb_table = (
        np.array(
            [[i for _ in range(self.emb_size)] for i in range(self.vocab_size)]
        )
        .reshape(self.vocab_size, self.emb_size)
        .astype(np.float32)
    )
    self.global_devices = np.array([mock.create_autospec(jax.Device)])

    # Shard the embedding table.
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )
    logging.debug("self.emb_table_sharded: %s", self.emb_table_sharded)

    self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching = jax.named_call(
        sparse_dense_matmul_grad_with_sgd.tpu_sparse_dense_matmul_grad_with_sgd_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching",
    )

  def test_sc_emb_backward_pass(self):
    # Prescribe the input.

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

    lhs_local_embedding_ids = jnp.array([
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

    lhs_local_sample_ids = jnp.array([
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
    lhs_gains = jnp.array([
        1., 1., 1., nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan, 1.,
        nan, nan, nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan,
        1., nan, nan, nan, nan, nan, nan, nan,
        1., 1., 1., nan, nan, nan, nan, nan])

    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains
    ) = _pad_input_tensors(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        max_mini_batch_size=4,
        output_padded_width=256,
    )

    logging.debug("lhs_row_pointers: %s", lhs_row_pointers)
    logging.debug("lhs_local_embedding_ids: %s", lhs_local_embedding_ids)
    logging.debug("lhs_local_sample_ids: %s", lhs_local_sample_ids)
    logging.debug("lhs_gains: %s", lhs_gains)

    z_grad = jnp.full(
        (
            # The gradient is padded to max_device_batch_size, no matter how
            # many rows are actually used.
            # This is to make sure we don't have different gradient dimensions
            # among test cases to avoid recompilation.
            self.max_device_batch_size,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    num_minibatches_per_physical_sparse_core = 1

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],
            z_grad,
            0.01,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
        )
    )

    logging.debug("updated_emb_table: %s", updated_emb_table)

    # Check the embedding activations.
    # For embedding id 0-15,
    # each has -0.01 (gradient) x 0.01 (learning rate) x 1 (sample) = -1e-4
    # For embedding id 16-31, nothing is updated.
    expected_emb_activations = np.array(
        [
            [
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
            ],
            [
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
            ],
            [
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
            ],
            [
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
            ],
            [
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
            ],
            [
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
            ],
            [
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
            ],
            [
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
            ],
            [
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
            ],
            [
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
            ],
            [
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
            ],
            [
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
            ],
            [
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
            ],
            [
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
            ],
            [
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
            ],
            [
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
            ],
            [
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
            ],
            [
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
            ],
            [
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
            ],
            [
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
            ],
            [
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
            ],
            [
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
            ],
            [
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
            ],
            [
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
            ],
            [
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
            ],
            [
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
            ],
            [
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
            ],
            [
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
            ],
            [
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
            ],
            [
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
            ],
            [
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
            ],
            [
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(updated_emb_table, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core(self):
    # Essentialy the inputs are repeated twice, so the expected updates are
    # twice as large.
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
    #
    # It actually points the beginning of the next section, and each section
    # length must be multiple of 8. Repeating the same value indicates 0, no
    # data from this SC. Each section must be padded to 8 multiples, so the
    # beginning of each SC must be the next multples of 8. Note that 8 is the
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

    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains
    ) = _pad_input_tensors(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        max_mini_batch_size=4,
        output_padded_width=256,
    )

    logging.debug("lhs_row_pointers: %s", lhs_row_pointers)
    logging.debug("lhs_local_embedding_ids: %s", lhs_local_embedding_ids)
    logging.debug("lhs_local_sample_ids: %s", lhs_local_sample_ids)
    logging.debug("lhs_gains: %s", lhs_gains)

    assert(lhs_local_sample_ids.shape == lhs_local_embedding_ids.shape)
    assert(lhs_gains.shape == lhs_local_embedding_ids.shape)

    z_grad = jnp.full(
        (
            # The gradient is padded to max_device_batch_size, no matter how
            # many rows are actually used.
            # This is to make sure we don't have different gradient dimensions
            # among test cases to avoid recompilation.
            self.max_device_batch_size,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    num_minibatches_per_physical_sparse_core = 2

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],
            z_grad,
            0.01,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
        )
    )

    logging.debug("updated_emb_table: %s", updated_emb_table)

    # Check the embedding activations.
    # Note that the expected updates are twice as large as the 1 batch case, for
    # we have 2 input samples for each embedding id.
    # For embedding id 0-15,
    # each has -0.01 (gradient) x 0.01 (learning rate) x 2 (samples) = -2e-4
    # For embedding id 16-31, nothing is updated.
    expected_emb_activations = np.array(
        [
            [
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
                -2.00000e-04,
            ],
            [
                3.99980e00,
                3.99980e00,
                3.99980e00,
                3.99980e00,
                3.99980e00,
                3.99980e00,
                3.99980e00,
                3.99980e00,
            ],
            [
                7.99980e00,
                7.99980e00,
                7.99980e00,
                7.99980e00,
                7.99980e00,
                7.99980e00,
                7.99980e00,
                7.99980e00,
            ],
            [
                1.19998e01,
                1.19998e01,
                1.19998e01,
                1.19998e01,
                1.19998e01,
                1.19998e01,
                1.19998e01,
                1.19998e01,
            ],
            [
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
            ],
            [
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
            ],
            [
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
            ],
            [
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
            ],
            [
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
                9.99800e-01,
            ],
            [
                4.99980e00,
                4.99980e00,
                4.99980e00,
                4.99980e00,
                4.99980e00,
                4.99980e00,
                4.99980e00,
                4.99980e00,
            ],
            [
                8.99980e00,
                8.99980e00,
                8.99980e00,
                8.99980e00,
                8.99980e00,
                8.99980e00,
                8.99980e00,
                8.99980e00,
            ],
            [
                1.29998e01,
                1.29998e01,
                1.29998e01,
                1.29998e01,
                1.29998e01,
                1.29998e01,
                1.29998e01,
                1.29998e01,
            ],
            [
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
            ],
            [
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
            ],
            [
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
            ],
            [
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
            ],
            [
                1.99980e00,
                1.99980e00,
                1.99980e00,
                1.99980e00,
                1.99980e00,
                1.99980e00,
                1.99980e00,
                1.99980e00,
            ],
            [
                5.99980e00,
                5.99980e00,
                5.99980e00,
                5.99980e00,
                5.99980e00,
                5.99980e00,
                5.99980e00,
                5.99980e00,
            ],
            [
                9.99980e00,
                9.99980e00,
                9.99980e00,
                9.99980e00,
                9.99980e00,
                9.99980e00,
                9.99980e00,
                9.99980e00,
            ],
            [
                1.39998e01,
                1.39998e01,
                1.39998e01,
                1.39998e01,
                1.39998e01,
                1.39998e01,
                1.39998e01,
                1.39998e01,
            ],
            [
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
            ],
            [
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
            ],
            [
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
            ],
            [
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
            ],
            [
                2.99980e00,
                2.99980e00,
                2.99980e00,
                2.99980e00,
                2.99980e00,
                2.99980e00,
                2.99980e00,
                2.99980e00,
            ],
            [
                6.99980e00,
                6.99980e00,
                6.99980e00,
                6.99980e00,
                6.99980e00,
                6.99980e00,
                6.99980e00,
                6.99980e00,
            ],
            [
                1.09998e01,
                1.09998e01,
                1.09998e01,
                1.09998e01,
                1.09998e01,
                1.09998e01,
                1.09998e01,
                1.09998e01,
            ],
            [
                1.49998e01,
                1.49998e01,
                1.49998e01,
                1.49998e01,
                1.49998e01,
                1.49998e01,
                1.49998e01,
                1.49998e01,
            ],
            [
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
            ],
            [
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
            ],
            [
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
            ],
            [
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(updated_emb_table, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core_with_bounds(self):
    # Tests a single SC evaluating gradients from _BATCH_SIZE sequences.
    # The SC separates this batch logically into _NUM_MINIBATCHES chunks.
    batch_size = 16
    num_sc = 1
    num_minibatches_per_sc = 2
    max_ids_per_partition = 16
    assert batch_size % (num_sc * num_minibatches_per_sc) == 0
    mini_batch_size = batch_size // (num_sc * num_minibatches_per_sc)

    # Process the input.
    # Total ids = 15. Pad by 1 to reach `max_ids_per_partition=16`.
    #
    # Assume 1 SC device, `num_minibatches_per_sc` = 2 minibatches.
    # Assume 4 samples (rows) per minibatch.
    #
    # minibatch 1:
    #   sample 0: [5]
    #   sample 1: [3]
    #   sample 2: [9, 1]
    #   sample 3: [6, 12, 0]
    # minibatch 2:
    #   sample 4: [4]
    #   sample 5: [15, 13, 11]
    #   sample 6: [7, 8, 14, 2]
    #   sample 7: [10]

    # Note that `lhs_local_sample_ids` is expected to be strictly increasing.
    # Also, `lhs_local_sample_ids` will indicate the sample idx globally
    # within the minibatches.
    # That means that if we provide the samples: [s3, s1, s2] and [s4, s2],
    # the unpadded inputs must be arranged as:
    # lhs_local_sample_ids = [s1, s2, s3,   s2, s4]
    # minibatches =            [mb_1]       [mb_2]
    # Which gives us:
    # lhs_row_pointers = [0, 1, 2, 3, 4, 5]
    unpadded_lhs_row_pointers = jnp.array(
        [0, 1, 2, 4, 7, 8, 11, 15, 16], jnp.int32
    )
    unpadded_lhs_local_embedding_ids = jnp.array(
        [5, 3, 1, 9, 0, 6, 12, 4, 11, 13, 15, 2, 7, 8, 14, 10], jnp.int32
    )
    unpadded_lhs_local_sample_ids = jnp.array(
        [0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7], jnp.int32
    )
    unpadded_lhs_gains = jnp.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        jnp.float32,
    )

    # The primitive expects fixed-size dense tensors.
    # Padding requirements:
    # 1. Row pointers require that padding corresponds to empty rows at the end.
    # 2. Embedding ids, sample ids, and gains must just be right-padded by 0s.
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = _pad_input_tensors(
        unpadded_lhs_row_pointers,
        unpadded_lhs_local_embedding_ids,
        unpadded_lhs_local_sample_ids,
        unpadded_lhs_gains,
        mini_batch_size,
        max_ids_per_partition,
    )

    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=1,
    )

    # Provide gradients for the physical batch size.
    z_grad = jnp.full(
        (
            self.max_device_batch_size,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_sc,
            emb_table_sharded[0],
            z_grad,
            0.01,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_minibatching_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
            min_value=2.0,
            max_value=12.0,
        )
    )

    expected_emb_activations = np.array(
        [
            [0.0] * 8,
            [1.0] * 8,
            [2.0] * 8,
            [2.9999] * 8,
            [3.9999] * 8,
            [4.9999] * 8,
            [5.9999] * 8,
            [6.9999] * 8,
            [7.9999] * 8,
            [8.9999] * 8,
            [9.9999] * 8,
            [10.9999] * 8,
            [11.9999] * 8,
            [12.0] * 8,
            [14.0] * 8,
            [15.0] * 8,
            [16.0] * 8,
            [17.0] * 8,
            [18.0] * 8,
            [19.0] * 8,
            [20.0] * 8,
            [21.0] * 8,
            [22.0] * 8,
            [23.0] * 8,
            [24.0] * 8,
            [25.0] * 8,
            [26.0] * 8,
            [27.0] * 8,
            [28.0] * 8,
            [29.0] * 8,
            [30.0] * 8,
            [31.0] * 8,
        ],
        dtype=np.float32,
    )
    updated_emb_table = einops.rearrange(
        updated_emb_table[jnp.newaxis, :, :],
        "c (s v) f -> (v c s) f",
        c=1,
        s=1,
    )

    logging.info("updated %s", updated_emb_table)
    np.testing.assert_allclose(
        updated_emb_table, expected_emb_activations, rtol=1e-4, atol=1e-4
    )


if __name__ == "__main__":
  absltest.main()
