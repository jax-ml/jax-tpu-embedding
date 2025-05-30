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
  """Unit-tests for the Adagrad+Momentum sparse-core gradient/update primitive."""

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
  beta2_param = np.float32(0.99)
  epsilon = np.float32(1e-5)
  k_power = np.float32(2.0)
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="momentum_buffer_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          momentum_buffer=momentum_buffer.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad.astype(np.int32),
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=1,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="momentum_param_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=1,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="beta2_param_dtype is not np.float32",
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
          beta2_param=1,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="epsilon_dtype is not np.float32",
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
          beta2_param=beta2_param,
          epsilon=1,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="k_power_dtype is not np.float32",
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
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=2,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      # Shape/rank checks
      dict(
          testcase_name="row_pointers_shape rank is not 1",
          row_pointers=np.array([[0, 1, 2, 3]]),
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "momentum_buffer_shape doesn't match embedding_table shape"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          momentum_buffer=np.zeros((1, 4), dtype=np.float32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_width doesn't match grad width",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0]]),
          accumulator=np.array([[1.0, 2.0, 3.0]]),
          momentum_buffer=np.array([[0.0, 0.0, 0.0]]),
          activations_grad=np.array([[1.0, 2.0, 3.0, 4.0]]),
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
          momentum_buffer=momentum_buffer,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          momentum_param=momentum_param,
          beta2_param=beta2_param,
          epsilon=epsilon,
          k_power=k_power,
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
      momentum_buffer,
      activations_grad,
      learning_rate,
      momentum_param,
      beta2_param,
      epsilon,
      k_power,
      max_ids_per_partition=None,
      max_unique_ids_per_partition=None,
  ):
    # Use class defaults if specific test case values are None
    effective_max_ids = (
        max_ids_per_partition
        if max_ids_per_partition is not None
        else self.max_ids_per_partition
    )
    effective_max_unique_ids = (
        max_unique_ids_per_partition
        if max_unique_ids_per_partition is not None
        else self.max_unique_ids_per_partition
    )
    current_beta2_param = (
        beta2_param if beta2_param is not None else self.beta2_param
    )
    current_epsilon = epsilon if epsilon is not None else self.epsilon
    current_k_power = k_power if k_power is not None else self.k_power

    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_adagrad_momentum.tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive.bind(
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
          current_beta2_param,
          current_epsilon,
          current_k_power,
          max_ids_per_partition=effective_max_ids,
          max_unique_ids_per_partition=effective_max_unique_ids,
      )

  def _compute_table_grad(
      self, inputs_ids, inputs_weights, activations_grad_samples
  ) -> np.ndarray:
    """Computes the dense gradient for the embedding table."""
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
      old_table_row: np.ndarray,
      grad_row: np.ndarray,
      old_accumulator_row: np.ndarray,
      old_momentum_row: np.ndarray,
      learning_rate: np.float32,
      momentum_param: np.float32,
      beta2_param: np.float32,
      epsilon: np.float32,
      k_power: np.float32,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_o = old_table_row
    g = grad_row
    a_o = old_accumulator_row
    m_o = old_momentum_row

    lr = learning_rate
    mom = momentum_param
    beta2 = beta2_param
    eps = epsilon
    k = k_power

    a_n = beta2 * a_o + (np.float32(1.0) - beta2) * (g * g)

    exponent = -np.float32(1.0) / k
    adj_grad = lr * g * jnp.power(a_n + eps, exponent)

    m_n = mom * m_o + adj_grad

    e_n = e_o - m_n

    return (np.asarray(e_n, dtype=np.float32),
            np.asarray(a_n, dtype=np.float32),
            np.asarray(m_n, dtype=np.float32))

  def test_sc_emb_backward_pass_with_adagrad_momentum(self):
    input_tensor_ids = np.array(
        [
            [i % _VOCAB_SIZE] for i in range(_BATCH_SIZE)
        ],
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
    epsilon_val = np.float32(1e-5)
    k_power_val = np.float32(0.5)

    activations_grad_np = jnp.full(
        (_BATCH_SIZE, _EMB_SIZE), 0.01, np.float32
    )
    table_grad_np = self._compute_table_grad(
        input_tensor_ids, input_tensor_weights, activations_grad_np
    )

    sparse_rows_to_update = np.unique(input_tensor_ids.flatten())

    expected_embedding_table_np = embedding_table_np.copy()
    expected_accumulator_np = accumulator_np.copy()
    expected_momentum_buffer_np = momentum_buffer_np.copy()

    for row_idx in sparse_rows_to_update:
      (
          expected_embedding_table_np[row_idx, :],
          expected_accumulator_np[row_idx, :],
          expected_momentum_buffer_np[row_idx, :],
      ) = self._compute_adagrad_momentum_update(
          embedding_table_np[row_idx, :],
          table_grad_np[row_idx, :],
          accumulator_np[row_idx, :],
          momentum_buffer_np[row_idx, :],
          learning_rate_val,
          momentum_param_val,
          beta2_param_val,
          epsilon_val,
          k_power_val,
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
        embedding_table_sharded[0],
        accumulator_sharded[0],
        momentum_buffer_sharded[0],
        np.asarray(activations_grad_np),
        learning_rate_val,
        momentum_param_val,
        beta2_param_val,
        epsilon_val,
        k_power_val,
        max_ids_per_partition=self.max_ids_per_partition,
        max_unique_ids_per_partition=self.max_unique_ids_per_partition,
        computation_name="adagrad_momentum_optimizer_test_computation",
        sharding_strategy=1,
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
