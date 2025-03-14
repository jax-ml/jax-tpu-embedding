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

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithAdagradTest(parameterized.TestCase):
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
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          embedding_table,
          accumulator,
          activations_grad,
          learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )

  def test_sc_emb_backward_pass_with_adagrad(self):
    # Process the input.
    input_tensor = np.array(
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
    input_weights = np.array(
        [[1.0] for _ in range(16)],
        dtype=np.float32,
    )
    global_devices = np.array([mock.create_autospec(jax.Device)])
    mesh = jax.sharding.Mesh(global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        input_tensor,
        input_weights,
        mesh,
        max_ids_per_partition=16,
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )
    emb_table = (
        np.array([[i for _ in range(_EMB_SIZE)] for i in range(_VOCAB_SIZE)])
        .reshape(_VOCAB_SIZE, _EMB_SIZE)
        .astype(np.float32)
    )
    emb_table_sharded = einops.rearrange(
        emb_table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )
    accumulator_init = jnp.full(
        emb_table_sharded[0].shape,
        0.00,
        np.float32,
    )

    z_grad = jnp.full(
        (
            _BATCH_SIZE,
            _EMB_SIZE,
        ),
        0.01,
        np.float32,
    )
    learning_rate = 0.01

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

    (updated_table, updated_accumulator) = (
        sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive.bind(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            emb_table_sharded[0],
            accumulator_init,
            z_grad,
            learning_rate,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )

    # Verify the expected tables and accumulators. Note that the inputs
    # contained a sample of each column between 0 and 15.
    np.testing.assert_equal(expected_accumulator, updated_accumulator)
    np.testing.assert_equal(expected_table, updated_table)


if __name__ == "__main__":
  absltest.main()
