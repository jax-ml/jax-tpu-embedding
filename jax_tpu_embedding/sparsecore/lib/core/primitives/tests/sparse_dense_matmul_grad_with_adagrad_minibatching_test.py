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
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
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


class SparseDenseMatmulGradWithAdagradWithMiniBatchingValidatorTest(
    parameterized.TestCase
):
  # None of the tensors are shaped correctly, as these bunch of tests are
  # intended to raise ValueError.
  row_pointers = np.array([0, 1, 2, 4], dtype=np.int32)
  sample_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  embedding_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  gains = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
  embedding_table = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  accumulator = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  learning_rate = 0.001
  max_ids_per_partition = 16
  max_unique_ids_per_partition = 16

  @parameterized.named_parameters(
      dict(
          testcase_name="row_pointers_dtype is not np.int32",
          row_pointers=row_pointers.astype(np.float32),
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="sample_ids_dtype is not np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids.astype(np.float32),
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_ids_dtype is not np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids.astype(np.float32),
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="gains_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains.astype(np.int32),
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table.astype(np.int32),
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="accumulator_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad.astype(np.int32),
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="learning_rate_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=1,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="row_pointers_shape rank is not 1",
          row_pointers=np.array([[0, 1, 2, 3]]),
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_shape doesn't match accumulator shape",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0, 4.0]]),
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dim is not 2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([1.0, 2.0, 3.0, 4.0]),
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dim is not 2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "embedding_table_activations_width doesn't match grad width"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0, 4.0]]),
          accumulator=accumulator,
          activations_grad=np.array([[1.0, 2.0, 3.0]]),
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="max_ids_per_partition is less than or equal to 0",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=0,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "max_unique_ids_per_partition is less than or equal to 0"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=0,
      ),
  )
  def test_raising_value_error_while_evaluating_abstracts(
      self,
      row_pointers,
      sample_ids,
      embedding_ids,
      gains,
      embedding_table,
      accumulator,
      activations_grad,
      learning_rate,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    num_minibatches_per_physical_sparse_core = 1
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          num_minibatches_per_physical_sparse_core,
          embedding_table,
          accumulator,
          activations_grad,
          learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
          enable_minibatching=True,
      )


class SparseDenseMatmulGradWithAdagradWithMiniBatchingTest(absltest.TestCase):
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

    self.accumulator_init = jnp.full(
        self.emb_table_sharded[0].shape,
        0.00,
        np.float32,
    )

    self.sparse_dense_matmul_grad_with_adagrad_with_mini_batching = jax.named_call(
        sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching",
    )

  def test_sc_emb_backward_pass_with_adagrad(self):
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

    # Gradient is padded to max_device_batch_size, no matter how many rows are
    # actually used.
    # This is to make sure we don't have different gradient dimensions among
    # test cases to avoid recompilation.
    z_grad = jnp.full(
        (
            self.max_device_batch_size,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )
    learning_rate = 0.01
    num_minibatches_per_physical_sparse_core = 1

    (updated_table, updated_accumulator) = (
        self.sparse_dense_matmul_grad_with_adagrad_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],
            self.accumulator_init,
            z_grad,
            learning_rate,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
        )
    )

    # Check the embedding activations.
    # For embedding id 0-15,
    # each has -0.01 (gradient) x 0.01 (learning rate) x 1 (sample) = -1e-4
    # For embedding id 16-31, nothing is updated.
    expected_table = np.array(
        [
            [-1.000e-02] * 8,
            [3.990e00] * 8,
            [7.990e00] * 8,
            [1.199e01] * 8,
            [1.600e01] * 8,
            [2.000e01] * 8,
            [2.400e01] * 8,
            [2.800e01] * 8,
            [9.900e-01] * 8,
            [4.990e00] * 8,
            [8.990e00] * 8,
            [1.299e01] * 8,
            [1.700e01] * 8,
            [2.100e01] * 8,
            [2.500e01] * 8,
            [2.900e01] * 8,
            [1.990e00] * 8,
            [5.990e00] * 8,
            [9.990e00] * 8,
            [1.399e01] * 8,
            [1.800e01] * 8,
            [2.200e01] * 8,
            [2.600e01] * 8,
            [3.000e01] * 8,
            [2.990e00] * 8,
            [6.990e00] * 8,
            [1.099e01] * 8,
            [1.499e01] * 8,
            [1.900e01] * 8,
            [2.300e01] * 8,
            [2.700e01] * 8,
            [3.100e01] * 8,
        ],
        dtype=np.float32,
    )

    expected_accumulator = np.array(
        [
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [1.0e-04] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
            [0.0e00] * 8,
        ],
        dtype=np.float32,
    )

    # Verify the expected tables and accumulators. Note that the inputs
    # contained a sample of each column between 0 and 15.
    np.testing.assert_equal(expected_accumulator, updated_accumulator)
    np.testing.assert_equal(expected_table, updated_table)

if __name__ == "__main__":
  absltest.main()
