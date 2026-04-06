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
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


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
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=4,
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
    self.features = [[
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
    ]]
    self.weights = [np.ones_like(sample) for sample in self.features]
    self.mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        self.lhs_row_pointers,
        self.lhs_local_embedding_ids,
        self.lhs_local_sample_ids,
        self.lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.features,
        self.weights,
        self.mesh,
        num_sc_per_device=4,
        max_ids_per_partition=8,
        max_unique_ids_per_partition=8,
        enable_minibatching=True,
    )

  def test_sc_emb_backward_pass_with_adagrad(self):
    # fmt: off
    mb0_feat = [
        [5], [3], [], [1], [6], [], [0], [4], [], [], [], [7], [], [], [2], [],
        [5], [3], [], [1], [6], [], [0], [4], [], [], [], [7], [], [], [2], [],
    ]
    mb1_feat = [
        [], [], [9], [], [], [12], [], [], [15], [13], [11], [], [8], [14], [], [10],
        [], [], [9], [], [], [12], [], [], [15], [13], [11], [], [8], [14], [], [10],
    ]
    mb0_weight = [
        [1.0], [1.0], [], [1.0], [1.0], [], [1.0], [1.0], [], [], [], [1.0], [], [], [1.0], [],
        [1.0], [1.0], [], [1.0], [1.0], [], [1.0], [1.0], [], [], [], [1.0], [], [], [1.0], [],
    ]
    mb1_weight = [
        [], [], [1.0], [], [], [1.0], [], [], [1.0], [1.0], [1.0], [], [1.0], [1.0], [], [1.0],
        [], [], [1.0], [], [], [1.0], [], [], [1.0], [1.0], [1.0], [], [1.0], [1.0], [], [1.0],
    ]
    # fmt: on

    features = [mb0_feat, mb1_feat]
    weights = [mb0_weight, mb1_weight]
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        features,
        weights,
        mesh,
        num_sc_per_device=4,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        enable_minibatching=True,
    )

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

    updated_table, updated_accumulator = (
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

  def test_sc_emb_backward_pass_with_adagrad_with_bounds(self):
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

    updated_table, updated_accumulator = (
        self.sparse_dense_matmul_grad_with_adagrad_with_mini_batching(
            self.lhs_row_pointers,
            self.lhs_local_embedding_ids,
            self.lhs_local_sample_ids,
            self.lhs_gains,
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
            min_value=2.0,
            max_value=10.0,
        )
    )

    # Check the embedding activations.
    # For embedding id 0-15,
    # each has -0.01 (gradient) x 0.01 (learning rate) x 1 (sample) = -1e-4
    # For embedding id 16-31, nothing is updated.
    expected_table = np.array(
        [
            [2.0] * 8,
            [3.990000009536743] * 8,
            [7.989999771118164] * 8,
            [10.0] * 8,
            [16.0] * 8,
            [20.0] * 8,
            [24.0] * 8,
            [28.0] * 8,
            [2.0] * 8,
            [4.989999771118164] * 8,
            [8.989999771118164] * 8,
            [10.0] * 8,
            [17.0] * 8,
            [21.0] * 8,
            [25.0] * 8,
            [29.0] * 8,
            [2.0] * 8,
            [5.989999771118164] * 8,
            [9.989999771118164] * 8,
            [10.0] * 8,
            [18.0] * 8,
            [22.0] * 8,
            [26.0] * 8,
            [30.0] * 8,
            [2.990000009536743] * 8,
            [6.989999771118164] * 8,
            [10.0] * 8,
            [10.0] * 8,
            [19.0] * 8,
            [23.0] * 8,
            [27.0] * 8,
            [31.0] * 8,
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
