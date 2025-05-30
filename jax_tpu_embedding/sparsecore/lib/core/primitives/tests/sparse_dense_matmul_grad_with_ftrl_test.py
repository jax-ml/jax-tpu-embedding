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
import jax.numpy as jnp  # For _compute_ftrl helper
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_ftrl
import numpy as np

_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithftrlTest(parameterized.TestCase):
  """Unit-tests for the FTRL sparse-core gradient/update primitive."""

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
  clip_weight_min = np.finfo(np.float32).min
  clip_weight_max = np.finfo(np.float32).max

  max_ids_per_partition = 16
  max_unique_ids_per_partition = 16

  @parameterized.named_parameters(
      dict(
          testcase_name="row_pointers_dtype_is_not_np.int32",
          row_pointers=row_pointers.astype(np.float32),
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="sample_ids_dtype_is_not_np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids.astype(np.float32),
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_ids_dtype_is_not_np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids.astype(np.float32),
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="gains_dtype_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains.astype(np.int32),
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dtype_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table.astype(np.int32),
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="accumulator_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator.astype(np.int32),
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="linear_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dtype_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad.astype(np.int32),
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="learning_rate_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=1,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="learning_rate_power_is_not_np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=1,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="l1_regularization_strength_is_not_np.float32",
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
          l1_regularization_strength=1,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="l2_regularization_strength_is_not_np.float32",
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
          l2_regularization_strength=1,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="beta_is_not_np.float32",
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
          beta=1,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="clip_weight_min_is_not_np.float32",
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
          clip_weight_min=1,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="clip_weight_max_is_not_np.float32",
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=1,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="row_pointers_shape_rank_is_not_1",
          row_pointers=np.array([[0, 1, 2, 3]]),
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "embedding_table_shape_does_not_match_accumulator_shape"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0, 4.0]]),
          accumulator=accumulator,
          linear=linear,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dim_is_not_2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([1.0, 2.0, 3.0, 4.0]),
          accumulator=np.array([1.0, 2.0, 3.0, 4.0]),
          linear=np.array([0.1, 0.2, 0.3, 0.4]),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dim_is_not_2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          linear=linear,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_width_does_not_match_grad_width",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0]]),
          accumulator=np.array([[1.0, 2.0, 3.0]]),
          linear=np.array([[0.1, 0.2, 0.3]]),
          activations_grad=np.array([[1.0, 2.0, 3.0, 4.0]]),
          learning_rate=learning_rate,
          learning_rate_power=learning_rate_power,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          beta=beta,
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="max_ids_per_partition_is_less_than_or_equal_to_0",
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
          max_ids_per_partition=0,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "max_unique_ids_per_partition_is_less_than_or_equal_to_0"
          ),
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
          clip_weight_min=clip_weight_min,
          clip_weight_max=clip_weight_max,
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
      linear,
      activations_grad,
      learning_rate,
      learning_rate_power,
      l1_regularization_strength,
      l2_regularization_strength,
      beta,
      clip_weight_min,
      clip_weight_max,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive.bind(
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
          clip_weight_min,
          clip_weight_max,
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
      clip_weight_min_val: np.float32,
      clip_weight_max_val: np.float32,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes the updated FTRL states (embedding_table, accumulator, linear).

    Notation matches the one used in the MLIR lowering implementation comments.
    Ē_o: old_table_e_o (old embedding table row)
    g: grad_g (gradient for the row)
    Ā_o: old_accum_a_o (old accumulator for the row)
    L_o: old_linear_l_o (old linear term for the row)
    λ: lr_lambda (learning_rate)
    k: lr_power_k (learning_rate_power, used as exponent's negative: Ā_α ^ (-k))
    γ₁: l1_gamma1 (l1_regularization_strength)
    γ₂: l2_gamma2 (l2_regularization_strength)
    β: beta_b (beta)

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
      clip_weight_min_val: Minimum value for weight clipping.
      clip_weight_max_val: Maximum value for weight clipping.

    Returns:
      A tuple containing the updated embedding table, accumulator, and linear
      term.
    """
    g_squared = grad_g * grad_g
    new_accum_a_n = old_accum_a_o + g_squared

    neg_lr_power_k = -lr_power_k

    p_o = jnp.power(old_accum_a_o, neg_lr_power_k)
    p_n = jnp.power(new_accum_a_n, neg_lr_power_k)

    one_over_lambda = np.float32(1.0) / lr_lambda
    delta_p = p_n - p_o
    delta_p_times_w = delta_p * old_table_e_o
    term_to_subtract = one_over_lambda * delta_p_times_w
    new_linear_l_n = old_linear_l_o + grad_g - term_to_subtract

    sign_linear_new = jnp.sign(new_linear_l_n)
    numer_selected = (sign_linear_new * l1_gamma1) - new_linear_l_n

    p_n_plus_beta = p_n + beta_b
    term1_denominator = p_n_plus_beta * one_over_lambda
    two_gamma2 = np.float32(2.0) * l2_gamma2
    denom_selected = term1_denominator + two_gamma2

    abs_lin_new = jnp.abs(new_linear_l_n)
    update_mask = abs_lin_new > l1_gamma1

    new_table_e_n = jnp.zeros_like(old_table_e_o)

    safe_denom = jnp.where(
        denom_selected == 0.0, np.float32(1e-8), denom_selected
    )

    w_update_values = numer_selected / safe_denom

    new_table_e_n = jnp.where(update_mask, w_update_values, new_table_e_n)

    new_table_e_n = jnp.clip(
        new_table_e_n, clip_weight_min_val, clip_weight_max_val
    )

    return (
        np.asarray(new_table_e_n, dtype=np.float32),
        np.asarray(new_accum_a_n, dtype=np.float32),
        np.asarray(new_linear_l_n, dtype=np.float32),
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
    clip_min_val = np.float32(-1.0)
    clip_max_val = np.float32(1.0)

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
          clip_min_val,
          clip_max_val,
      )

    (updated_table, updated_accumulator, updated_linear) = (
        sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive.bind(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            embedding_table_sharded[0],
            accumulator_sharded[0],
            linear_sharded[0],
            np.asarray(activations_grad_np),
            learning_rate_val,
            learning_rate_power_val,
            l1_val,
            l2_val,
            beta_val,
            clip_min_val,
            clip_max_val,
            max_ids_per_partition=self.max_ids_per_partition,
            max_unique_ids_per_partition=self.max_unique_ids_per_partition,
            computation_name="ftrl_optimizer_test_computation",
            sharding_strategy=1,
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
