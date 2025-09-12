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
from typing import Tuple
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import (
    sparse_dense_matmul_grad_with_adagrad_momentum,
)
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithAdagradMomentumTest(parameterized.TestCase):
  """Unit-tests for the Adagrad-Momentum SparseCore gradient/update primitive."""

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
  momentum_buffer = np.zeros_like(embedding_table)
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )

  learning_rate = np.float32(0.01)
  momentum_param = np.float32(0.9)
  beta2 = np.float32(0.99)
  epsilon = np.float32(1e-5)
  exponent = np.float32(2.0)
  use_nesterov = np.bool_(False)

  max_ids_per_partition = 16
  max_unique_ids_per_partition = 16

  _BASE_TEST_PARAMS = dict(
      row_pointers=row_pointers,
      sample_ids=sample_ids,
      embedding_ids=embedding_ids,
      gains=gains,
      embedding_table=embedding_table,
      accumulator=accumulator,
      momentum_buffer=momentum_buffer,
      activations_grad=activations_grad,
      learning_rate=learning_rate,
      momentum_param=momentum_param,
      beta2=beta2,
      epsilon=epsilon,
      exponent=exponent,
      use_nesterov=use_nesterov,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
  )

  @parameterized.named_parameters(
      dict(
          testcase_name="row_pointers_dtype_is_not_np.int32",
          **{
              **_BASE_TEST_PARAMS,
              "row_pointers": row_pointers.astype(np.float32),
          },
      ),
      dict(
          testcase_name="embedding_ids_dtype_is_not_np.int32",
          **{
              **_BASE_TEST_PARAMS,
              "embedding_ids": embedding_ids.astype(np.float32),
          },
      ),
      dict(
          testcase_name="gains_dtype_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "gains": gains.astype(np.int32)},
      ),
      dict(
          testcase_name="embedding_table_dtype_is_not_np.float32",
          **{
              **_BASE_TEST_PARAMS,
              "embedding_table": embedding_table.astype(np.int32),
          },
      ),
      dict(
          testcase_name="accumulator_dtype_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "accumulator": accumulator.astype(np.int32)},
      ),
      dict(
          testcase_name="momentum_buffer_dtype_is_not_np.float32",
          **{
              **_BASE_TEST_PARAMS,
              "momentum_buffer": momentum_buffer.astype(np.int32),
          },
      ),
      dict(
          testcase_name="activations_grad_dtype_is_not_np.float32",
          **{
              **_BASE_TEST_PARAMS,
              "activations_grad": activations_grad.astype(np.int32),
          },
      ),
      dict(
          testcase_name="learning_rate_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "learning_rate": 1},
      ),
      dict(
          testcase_name="momentum_param_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "momentum_param": 1},
      ),
      dict(
          testcase_name="beta2_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "beta2": 1},
      ),
      dict(
          testcase_name="epsilon_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "epsilon": 1},
      ),
      dict(
          testcase_name="exponent_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "exponent": 2},
      ),
      dict(
          testcase_name="row_pointers_shape_rank_is_not_1",
          **{**_BASE_TEST_PARAMS, "row_pointers": np.array([[0, 1, 2, 3]])},
      ),
      dict(
          testcase_name=(
              "embedding_table_shape_does_not_match_accumulator_shape"
          ),
          **{
              **_BASE_TEST_PARAMS,
              "embedding_table": np.array([[1.0, 2.0, 3.0, 4.0]]),
          },
      ),
      dict(
          testcase_name="embedding_table_dim_is_not_2",
          **{
              **_BASE_TEST_PARAMS,
              "embedding_table": np.array([1.0, 2.0, 3.0, 4.0]),
              "accumulator": np.array([1.0, 2.0, 3.0, 4.0]),
              "momentum_buffer": np.array([0.0, 0.0, 0.0, 0.0]),
          },
      ),
      dict(
          testcase_name="activations_grad_dim_is_not_2",
          **{
              **_BASE_TEST_PARAMS,
              "activations_grad": np.array([1.0, 2.0, 3.0]),
          },
      ),
      dict(
          testcase_name="embedding_table_width_does_not_match_grad_width",
          **{
              **_BASE_TEST_PARAMS,
              "embedding_table": np.array([[1.0, 2.0, 3.0]]),
              "accumulator": np.array([[1.0, 2.0, 3.0]]),
              "momentum_buffer": np.array([[0.0, 0.0, 0.0]]),
              "activations_grad": np.array([[1.0, 2.0, 3.0, 4.0]]),
          },
      ),
      dict(
          testcase_name="max_ids_per_partition_is_less_than_or_equal_to_0",
          **{**_BASE_TEST_PARAMS, "max_ids_per_partition": 0},
      ),
      dict(
          testcase_name=(
              "max_unique_ids_per_partition_is_less_than_or_equal_to_0"
          ),
          **{**_BASE_TEST_PARAMS, "max_unique_ids_per_partition": 0},
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
      momentum_buffer,
      activations_grad,
      learning_rate,
      momentum_param,
      beta2,
      epsilon,
      exponent,
      use_nesterov,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_adagrad_momentum.tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          np.int32(1),  # num_minibatches_per_physical_sparse_core
          embedding_table,
          accumulator,
          momentum_buffer,
          activations_grad,
          learning_rate,
          momentum_param,
          beta2,
          epsilon,
          exponent,
          use_nesterov,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
          enable_minibatching=False,
      )

  def _compute_table_grad(
      self, inputs_ids, inputs_weights, activations_grad_samples
  ):
    """Dense embedding-table gradient for sanity checks."""
    inputs_ids_jnp = jnp.asarray(inputs_ids)
    inputs_weights_jnp = jnp.asarray(inputs_weights)
    activations_grad_jnp = jnp.asarray(activations_grad_samples)

    batch_size = activations_grad_jnp.shape[0]
    if inputs_ids_jnp.ndim == 2:
      sample_lengths = jnp.array([inputs_ids_jnp.shape[1]] * batch_size)
    else:
      sample_lengths = jnp.array([len(r) for r in inputs_ids_jnp])

    rows = jnp.repeat(jnp.arange(batch_size), sample_lengths)
    cols = inputs_ids_jnp.flatten()
    vals = inputs_weights_jnp.flatten().reshape(-1, 1)

    table_grad = jnp.zeros((_VOCAB_SIZE, _EMB_SIZE), dtype=jnp.float32)
    table_grad = table_grad.at[cols, :].add(
        vals * activations_grad_jnp[rows, :]
    )
    return np.asarray(table_grad)

  def _shard_table(self, table: np.ndarray) -> np.ndarray:
    """Shards a dense table for SparseCore input."""
    return einops.rearrange(
        table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=_NUM_SC_PER_DEVICE,
    )

  def _unshard_table(self, table: np.ndarray) -> np.ndarray:
    """Unshards a table from SparseCore output format."""
    return einops.rearrange(
        table,
        "c (s v) f -> (v c s) f",
        c=1,  # Devices.
        s=_NUM_SC_PER_DEVICE,  # SparseCores per device.
    )

  def _compute_adagrad_momentum_update(
      self,
      e_o: np.ndarray,
      g: np.ndarray,
      a_o: np.ndarray,
      m_o: np.ndarray,
      lr: np.float32,
      mom: np.float32,
      beta2: np.float32,
      eps: np.float32,
      k: np.float32,
      use_nesterov_flag: np.bool_,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference numpy implementation of one Adagrad-Momentum step (row-wise)."""
    # Accumulator
    if beta2 == np.float32(1.0):
      a_n = a_o + g * g
    else:
      a_n = beta2 * a_o + (np.float32(1.0) - beta2) * (g * g)

    # Scaled gradient
    neg_inv_exp = -np.float32(1.0) / k
    scale_factor = jnp.power(a_n + eps, neg_inv_exp)
    g_hat = scale_factor * g

    # Momentum
    m_n = mom * m_o + g_hat

    # Delta E
    if use_nesterov_flag:
      # delta = lr * (mom *m_n + g_hat)
      delta = lr * (mom * m_n + g_hat)
    else:
      # delta = lr * m_n
      delta = lr * m_n

    # Weight update
    e_n = e_o - delta

    return (
        np.asarray(e_n, dtype=np.float32),
        np.asarray(a_n, dtype=np.float32),
        np.asarray(m_n, dtype=np.float32),
    )

  def test_adagrad_momentum_optimizer_update(self):
    input_tensor_ids = np.array(
        [[i % _VOCAB_SIZE] for i in range(_BATCH_SIZE)],
        dtype=np.int32,
    )
    input_tensor_weights = np.ones_like(input_tensor_ids, dtype=np.float32)

    global_devices = np.array([mock.create_autospec(jax.Device)])
    mesh = jax.sharding.Mesh(global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        input_tensor_ids,
        input_tensor_weights,
        mesh,
        max_ids_per_partition=self.max_ids_per_partition,
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    embedding_table_np = (
        np.array([
            [(i * 0.1 + j * 0.01) for j in range(_EMB_SIZE)]
            for i in range(_VOCAB_SIZE)
        ])
        .reshape(_VOCAB_SIZE, _EMB_SIZE)
        .astype(np.float32)
    )
    accumulator_np = np.full_like(embedding_table_np, 0.1, np.float32)
    momentum_buffer_np = np.full_like(embedding_table_np, 0.01, np.float32)

    embedding_table_sharded = self._shard_table(embedding_table_np)
    accumulator_sharded = self._shard_table(accumulator_np)
    momentum_buffer_sharded = self._shard_table(momentum_buffer_np)

    learning_rate_val = np.float32(0.1)
    momentum_param_val = np.float32(0.9)
    beta2_param_val = np.float32(0.99)
    epsilon_val = np.float32(1e-10)
    k_power_val = np.float32(0.5)
    use_nesterov_flag = np.bool_(True)

    activations_grad_np = jnp.full((_BATCH_SIZE, _EMB_SIZE), 0.01, np.float32)
    table_grad_np = self._compute_table_grad(
        input_tensor_ids, input_tensor_weights, activations_grad_np
    )

    sparse_rows_to_update = np.unique(input_tensor_ids.flatten())

    expected_embedding_table_np = embedding_table_np.copy()
    expected_accumulator_np = accumulator_np.copy()
    expected_momentum_buffer_np = momentum_buffer_np.copy()

    for row in sparse_rows_to_update:
      (
          expected_embedding_table_np[row, :],
          expected_accumulator_np[row, :],
          expected_momentum_buffer_np[row, :],
      ) = self._compute_adagrad_momentum_update(
          embedding_table_np[row, :],
          table_grad_np[row, :],
          accumulator_np[row, :],
          momentum_buffer_np[row, :],
          learning_rate_val,
          momentum_param_val,
          beta2_param_val,
          epsilon_val,
          k_power_val,
          use_nesterov_flag,
      )
    (
        updated_table,
        updated_accumulator,
        updated_momentum,
    ) = sparse_dense_matmul_grad_with_adagrad_momentum.tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.bind(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        np.int32(1),  # num_minibatches_per_physical_sparse_core
        embedding_table_sharded[0],
        accumulator_sharded[0],
        momentum_buffer_sharded[0],
        np.asarray(activations_grad_np),
        learning_rate_val,
        momentum_param_val,
        beta2_param_val,
        epsilon_val,
        k_power_val,
        use_nesterov_flag,
        max_ids_per_partition=self.max_ids_per_partition,
        max_unique_ids_per_partition=self.max_unique_ids_per_partition,
        computation_name="adagrad_momentum_optimizer_test_computation",
        sharding_strategy=1,
        enable_minibatching=False,
    )

    updated_table_unsharded = self._unshard_table(
        updated_table[jnp.newaxis, :, :]
    )
    updated_accumulator_unsharded = self._unshard_table(
        updated_accumulator[jnp.newaxis, :, :]
    )
    updated_momentum_unsharded = self._unshard_table(
        updated_momentum[jnp.newaxis, :, :]
    )

    np.testing.assert_allclose(
        expected_accumulator_np,
        updated_accumulator_unsharded,
    )
    np.testing.assert_allclose(
        expected_momentum_buffer_np,
        updated_momentum_unsharded,
    )
    np.testing.assert_allclose(
        expected_embedding_table_np,
        updated_table_unsharded,
    )


if __name__ == "__main__":
  absltest.main()
