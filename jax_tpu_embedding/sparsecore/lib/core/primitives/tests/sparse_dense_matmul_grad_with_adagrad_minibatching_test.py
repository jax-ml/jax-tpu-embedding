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


class SparseDenseMatmulGradWithAdagradWithMiniBatchingTest(
    parameterized.TestCase
):

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
        self.emb_table_sharded[0].shape,  # pyrefly: ignore[bad-index]
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

  @parameterized.named_parameters(
      ("no_clipping", None, None),
      ("clipping", 2.0, 10.0),
  )
  def test_sc_emb_backward_pass_with_adagrad(self, min_value, max_value):
    # Arrange
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

    # Act
    updated_table, updated_accumulator = (
        self.sparse_dense_matmul_grad_with_adagrad_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],  # pyrefly: ignore[bad-index]
            self.accumulator_init,
            z_grad,
            learning_rate,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
            min_value=min_value,
            max_value=max_value,
        )
    )

    # Assert
    # Check the embedding activations.
    actual_table_unsharded = utils.unshard_emb_table(
        updated_table[jnp.newaxis, :, :], num_sc_per_device=4
    )
    actual_accumulator_unsharded = utils.unshard_emb_table(
        updated_accumulator[jnp.newaxis, :, :], num_sc_per_device=4
    )

    # Compute the expected results on CPU while the primitive runs on TPU.
    expected_table_unsharded = self.emb_table.copy()
    updated_rows = np.unique(jax.tree_util.tree_flatten(features)[0])
    # Adagrad update: weight = weight - lr * 1.0 = weight - 0.01
    expected_table_unsharded[updated_rows, :] -= 0.01

    expected_table_unsharded[updated_rows, :] = np.clip(
        expected_table_unsharded[updated_rows, :], min_value, max_value
    )

    expected_accumulator_unsharded = np.zeros_like(self.emb_table)
    # Accumulator update: accum = 0 + (0.01)^2 = 1e-4
    expected_accumulator_unsharded[updated_rows, :] = 1e-4

    # Verify the expected tables and accumulators. Note that the inputs
    # contained a sample of each column between 0 and 15.
    np.testing.assert_allclose(
        actual_accumulator_unsharded, expected_accumulator_unsharded, atol=1e-5
    )
    np.testing.assert_allclose(
        actual_table_unsharded, expected_table_unsharded, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
