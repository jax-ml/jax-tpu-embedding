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
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_ftrl
import numpy as np

_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithFtrlTest(parameterized.TestCase):
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
  linear = np.array(
      [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
  )
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )

  learning_rate = np.float32(0.01)
  learning_rate_power = np.float32(-0.5)
  l1_regularization_strength = np.float32(0.0)
  l2_regularization_strength = np.float32(0.0)
  beta = np.float32(0.0)
  multiply_linear_by_learning_rate = np.bool(False)

  max_ids_per_partition = 16
  max_unique_ids_per_partition = 16

  _BASE_TEST_PARAMS = dict(
      row_pointers=row_pointers,
      sample_ids=sample_ids,
      embedding_ids=embedding_ids,
      gains=gains,
      embedding_table=embedding_table,
      accumulator=accumulator,
      linear=linear,
      activations_grad=activations_grad,
      learning_rate=learning_rate,
      learning_rate_power=learning_rate_power,
      l1_regularization_strength=l1_regularization_strength,
      l2_regularization_strength=l2_regularization_strength,
      beta=beta,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
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
          testcase_name="sample_ids_dtype_is_not_np.int32",
          **{**_BASE_TEST_PARAMS, "sample_ids": sample_ids.astype(np.float32)},
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
          testcase_name="accumulator_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "accumulator": accumulator.astype(np.int32)},
      ),
      dict(
          testcase_name="linear_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "linear": linear.astype(np.int32)},
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
          testcase_name="learning_rate_power_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "learning_rate_power": 1},
      ),
      dict(
          testcase_name="l1_regularization_strength_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "l1_regularization_strength": 1},
      ),
      dict(
          testcase_name="l2_regularization_strength_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "l2_regularization_strength": 1},
      ),
      dict(
          testcase_name="beta_is_not_np.float32",
          **{**_BASE_TEST_PARAMS, "beta": 1},
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
              "linear": np.array([0.1, 0.2, 0.3, 0.4]),
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
              "linear": np.array([[0.1, 0.2, 0.3]]),
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
      linear,
      activations_grad,
      learning_rate,
      learning_rate_power,
      l1_regularization_strength,
      l2_regularization_strength,
      beta,
      multiply_linear_by_learning_rate,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          1,  # num_minibatches_per_physical_sparse_core
          embedding_table,
          accumulator,
          linear,
          activations_grad,
          learning_rate,
          learning_rate_power,
          l1_regularization_strength,
          l2_regularization_strength,
          beta,
          multiply_linear_by_learning_rate=multiply_linear_by_learning_rate,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )

  def _compute_ftrl(
      self,
      old_table_e_o: np.ndarray,
      grad_g: np.ndarray,
      old_accum_a_o: np.ndarray,
      old_linear_l_o: np.ndarray,
      lr_lambda: np.float32,
      lr_power_k: np.float32,
      l1_gamma1: np.float32,
      l2_gamma2: np.float32,
      beta_b: np.float32,
      mlr_flag: np.bool,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the updated FTRL states (embedding_table, accumulator, linear).

    Args:
      old_table_e_o: Old embedding table row.
      grad_g: Gradient for the row.
      old_accum_a_o: Old accumulator for the row.
      old_linear_l_o: Old linear term for the row.
      lr_lambda: Learning rate.
      lr_power_k: Learning rate power, used as exponent's negative: Ā_α ^ (-k).
      l1_gamma1: L1 regularization strength.
      l2_gamma2: L2 regularization strength.
      beta_b: Beta value.
      mlr_flag: Whether to use multiply linear by learning rate.

    Returns:
      A tuple containing the updated embedding table, accumulator, and linear
      term.
    """
    two = jnp.array(2.0, dtype=jnp.float32)
    zero = jnp.array(0.0, dtype=jnp.float32)

    # Accumulator
    a_new = old_accum_a_o + grad_g * grad_g

    # Power‑law terms
    p_old = jnp.power(old_accum_a_o, -lr_power_k)
    p_new = jnp.power(a_new, -lr_power_k)
    delta_p = p_new - p_old

    # Linear State
    new_linear_l_o = jnp.where(
        mlr_flag,
        # mlr = True
        old_linear_l_o + lr_lambda * grad_g - delta_p * old_table_e_o,
        # mlr == False
        old_linear_l_o + grad_g - (delta_p / lr_lambda) * old_table_e_o,
    )

    # Thresholding and Numerator
    l_threshold = jnp.where(mlr_flag, lr_lambda * l1_gamma1, l1_gamma1)
    numerator = jnp.sign(new_linear_l_o) * l_threshold - new_linear_l_o
    abs_l_new = jnp.abs(new_linear_l_o)

    # Denominator
    denominator = jnp.where(
        mlr_flag,
        # mlr = True
        p_new + two * lr_lambda * l2_gamma2 + beta_b,
        # mlr == False
        (p_new + beta_b) / lr_lambda + two * l2_gamma2,
    )

    # Weight update
    w_new = jnp.where(abs_l_new > l_threshold, numerator / denominator, zero)

    return (
        np.asarray(w_new, dtype=np.float32),
        np.asarray(a_new, dtype=np.float32),
        np.asarray(new_linear_l_o, dtype=np.float32),
    )

  def _shard_table(self, table):
    """Shards a dense table for SparseCore input."""
    return einops.rearrange(
        table,
        "(v c s) f -> c (s v) f",
        c=1,  # Devices.
        s=_NUM_SC_PER_DEVICE,  # SparseCores per device.
    )

  def _unshard_table(self, table):
    """Unshards a table from SparseCore output format."""
    return einops.rearrange(
        table,
        "c (s v) f -> (v c s) f",
        c=1,  # Devices.
        s=_NUM_SC_PER_DEVICE,  # SparseCores per device.
    )

  def _compute_table_grad(
      self, inputs_ids, inputs_weights, activations_grad_samples
  ):
    """Computes the dense gradient for the embedding table."""
    inputs_ids_jnp = jnp.asarray(inputs_ids)
    inputs_weights_jnp = jnp.asarray(inputs_weights)
    activations_grad_samples_jnp = jnp.asarray(activations_grad_samples)

    batch_size = activations_grad_samples_jnp.shape[0]
    if inputs_ids_jnp.ndim == 2:  # [batch_size, features_per_sample]
      sample_lengths = jnp.array([inputs_ids_jnp.shape[1]] * batch_size)
    else:  # Fallback for 1D or other structures
      sample_lengths = jnp.array([len(sample) for sample in inputs_ids_jnp])

    rows = jnp.repeat(jnp.arange(batch_size), sample_lengths)
    cols = inputs_ids_jnp.flatten()
    vals = inputs_weights_jnp.flatten().reshape(-1, 1)

    table_grad = jnp.zeros(shape=(_VOCAB_SIZE, _EMB_SIZE), dtype=jnp.float32)
    table_grad = table_grad.at[cols, :].add(
        vals * activations_grad_samples_jnp[rows, :]
    )
    return np.asarray(table_grad)

  def test_ftrl_optimizer_update(self):
    input_tensor_ids = np.array(
        [
            [i % _VOCAB_SIZE] for i in range(_BATCH_SIZE)
        ],  # Each sample has one feature ID
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
    linear_np = np.full_like(embedding_table_np, 0.01, np.float32)

    embedding_table_sharded = self._shard_table(embedding_table_np)
    accumulator_sharded = self._shard_table(accumulator_np)
    linear_sharded = self._shard_table(linear_np)

    learning_rate_val = np.float32(0.1)
    learning_rate_power_val = np.float32(-0.5)
    l1_val = np.float32(0.001)
    l2_val = np.float32(0.002)
    beta_val = np.float32(0.01)
    multiply_linear_by_learning_rate_flag = False

    activations_grad_np = jnp.full((_BATCH_SIZE, _EMB_SIZE), 0.01, np.float32)
    table_grad_np = self._compute_table_grad(
        input_tensor_ids, input_tensor_weights, activations_grad_np
    )

    # Compute expected values for rows involved in the forward pass.
    # These are guaranteed to be valid by `i % _VOCAB_SIZE` in input_tensor_ids
    sparse_rows_to_update = np.unique(input_tensor_ids.flatten())

    expected_embedding_table_np = embedding_table_np.copy()
    expected_accumulator_np = accumulator_np.copy()
    expected_linear_np = linear_np.copy()

    for row_idx in sparse_rows_to_update:
      (
          expected_embedding_table_np[row_idx, :],
          expected_accumulator_np[row_idx, :],
          expected_linear_np[row_idx, :],
      ) = self._compute_ftrl(
          embedding_table_np[row_idx, :],
          table_grad_np[row_idx, :],
          accumulator_np[row_idx, :],
          linear_np[row_idx, :],
          learning_rate_val,
          learning_rate_power_val,
          l1_val,
          l2_val,
          beta_val,
          multiply_linear_by_learning_rate_flag,
      )

    (updated_table, updated_accumulator, updated_linear) = (
        sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive.bind(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            1,  # num_minibatches_per_physical_sparse_core
            embedding_table_sharded[0],
            accumulator_sharded[0],
            linear_sharded[0],
            np.asarray(activations_grad_np),
            learning_rate_val,
            learning_rate_power_val,
            l1_val,
            l2_val,
            beta_val,
            multiply_linear_by_learning_rate=multiply_linear_by_learning_rate_flag,
            max_ids_per_partition=self.max_ids_per_partition,
            max_unique_ids_per_partition=self.max_unique_ids_per_partition,
            computation_name="ftrl_optimizer_test_computation",
            sharding_strategy=1,
            minibatches=False,
        )
    )
    updated_table_unsharded = self._unshard_table(
        updated_table[jnp.newaxis, :, :]
    )
    updated_accumulator_unsharded = self._unshard_table(
        updated_accumulator[jnp.newaxis, :, :]
    )
    updated_linear_unsharded = self._unshard_table(
        updated_linear[jnp.newaxis, :, :]
    )

    np.testing.assert_allclose(
        expected_accumulator_np,
        updated_accumulator_unsharded,
    )
    np.testing.assert_allclose(
        expected_linear_np,
        updated_linear_unsharded,
    )
    np.testing.assert_allclose(
        expected_embedding_table_np,
        updated_table_unsharded,
    )


if __name__ == "__main__":
  absltest.main()
