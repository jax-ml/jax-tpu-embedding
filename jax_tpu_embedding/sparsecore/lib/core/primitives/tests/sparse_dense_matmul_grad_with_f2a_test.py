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
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_f2a
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithF2aTest(parameterized.TestCase):
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
  local_step = np.array(
      [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
  )
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  learning_rate = 0.01
  rho = 0.5
  global_step = 10.0
  l1_regularization_strength = 0.0
  l2_regularization_strength = 0.0
  max_lr_multiplier = 100.0
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="local_step_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_shape doesn't match local_step shape",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=np.array([[0.0, 0.0, 0.0]]),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad.astype(np.int32),
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=1,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="rho_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=1,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="l1_regularization_strength_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=1,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="l2_regularization_strength_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=1,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="max_lr_multiplier_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=1,
          global_step=global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="global_step_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          accumulator=accumulator,
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=1,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
          local_step=local_step,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          rho=rho,
          l1_regularization_strength=l1_regularization_strength,
          l2_regularization_strength=l2_regularization_strength,
          max_lr_multiplier=max_lr_multiplier,
          global_step=global_step,
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
      local_step,
      activations_grad,
      learning_rate,
      rho,
      l1_regularization_strength,
      l2_regularization_strength,
      max_lr_multiplier,
      global_step,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_f2a.tpu_sparse_dense_matmul_grad_with_f2a_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          1,
          embedding_table,
          accumulator,
          local_step,
          activations_grad,
          learning_rate,
          rho,
          l1_regularization_strength,
          l2_regularization_strength,
          max_lr_multiplier,
          global_step,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="default",
          min_value=None,
          max_value=None,
      ),
      dict(
          testcase_name="bounds",
          min_value=2.0,
          max_value=12.0,
      ),
  )
  def test_sc_emb_backward_pass_with_f2a(self, min_value, max_value):
    input_tensor = jnp.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
    ])
    input_weights = jnp.ones((_BATCH_SIZE, 4), dtype=np.float32)

    embedding_table = jnp.arange(
        _VOCAB_SIZE * _EMB_SIZE, dtype=np.float32
    ).reshape(_VOCAB_SIZE, _EMB_SIZE)

    accumulator = jnp.full(embedding_table.shape, 0.1, np.float32)
    local_step = jnp.zeros(embedding_table.shape, np.float32)

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
        num_sc_per_device=4,
    )

    def _shard_table(table):
      return einops.rearrange(
          table,
          "(v c s) f -> c (s v) f",
          c=1,  # Devices.
          s=4,  # SparseCores per device.
      )

    embedding_table_sharded = _shard_table(np.asarray(embedding_table))
    accumulator_sharded = _shard_table(np.asarray(accumulator))
    local_step_sharded = _shard_table(np.asarray(local_step))

    learning_rate = 0.01
    rho = 1.0
    global_step = 100.0
    l1_regularization_strength = 0.01
    l2_regularization_strength = 0.01
    max_lr_multiplier = 10.0

    activations_grad = jnp.full((_BATCH_SIZE, _EMB_SIZE), 0.1, np.float32)

    def _compute_table_grad(inputs, weights, activation_grad):
      batch_size = activation_grad.shape[0]
      sample_lengths = jnp.array([len(sample) for sample in inputs])
      rows = jnp.repeat(jnp.arange(batch_size), sample_lengths)
      cols = jnp.concatenate(np.unstack(inputs))
      vals = jnp.concatenate(np.unstack(weights)).reshape(-1, 1)

      grad = jnp.zeros(shape=(_VOCAB_SIZE, _EMB_SIZE))
      grad = grad.at[cols, :].add(vals * activation_grad[rows, :])
      return grad

    def _compute_f2a(
        theta, g, a, l_step, alpha, rho, l1, l2, max_lr_mult, g_step
    ):
      """Compute F2A update on host.

      Args:
        theta: Embedding table weights.
        g: Gradients.
        a: Accumulator.
        l_step: Local step.
        alpha: Learning rate.
        rho: Exponent for frequency ratio.
        l1: L1 regularization strength.
        l2: L2 regularization strength.
        max_lr_mult: Maximum learning rate multiplier.
        g_step: Global step.

      Returns:
        A tuple containing (theta_new, a_new, l_step_new).
      """
      l_step_new = l_step + 1.0
      safe_g_step = jnp.maximum(g_step, 1.0)
      frequency_ratio = safe_g_step / l_step_new
      fa_multiplier = jnp.power(frequency_ratio, rho)
      fa_multiplier = jnp.minimum(fa_multiplier, max_lr_mult)
      a_new = a + g * g
      denominator = jnp.sqrt(a_new)

      l1_decay = l1 * jnp.sign(theta)
      l2_shrinkage = l2 * theta

      update = ((fa_multiplier * g) / denominator) + l1_decay + l2_shrinkage
      theta_new = theta - alpha * update
      return theta_new, a_new, l_step_new

    table_grad = _compute_table_grad(
        input_tensor, input_weights, activations_grad
    )

    expected_embedding_table, expected_accumulator, expected_local_step = (
        _compute_f2a(
            np.asarray(embedding_table),
            np.asarray(table_grad),
            np.asarray(accumulator),
            np.asarray(local_step),
            learning_rate,
            rho,
            l1_regularization_strength,
            l2_regularization_strength,
            max_lr_multiplier,
            global_step,
        )
    )

    if min_value is not None and max_value is not None:
      expected_embedding_table = jnp.clip(
          expected_embedding_table, min_value, max_value
      )

    tpu_f2a = (
        sparse_dense_matmul_grad_with_f2a.tpu_sparse_dense_matmul_grad_with_f2a_primitive.bind
    )

    updated_vars = tpu_f2a(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,
        embedding_table_sharded[0],
        accumulator_sharded[0],
        local_step_sharded[0],
        activations_grad,
        learning_rate,
        rho,
        l1_regularization_strength,
        l2_regularization_strength,
        max_lr_multiplier,
        global_step,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="f2a_test_computation_parameterized",
        sharding_strategy=1,
        min_value=min_value,
        max_value=max_value,
    )

    updated_embedding_table, updated_accumulator, updated_local_step = (
        updated_vars
    )

    def _unshard_table(table):
      return einops.rearrange(
          jnp.expand_dims(table, axis=0),
          "c (s v) f -> (v c s) f",
          c=1,  # Devices.
          s=4,  # SparseCores per device.
      )

    updated_embedding_table_unsharded = _unshard_table(updated_embedding_table)
    updated_accumulator_unsharded = _unshard_table(updated_accumulator)
    updated_local_step_unsharded = _unshard_table(updated_local_step)

    sparse_rows = jnp.unique(jnp.concatenate(np.unstack(input_tensor)))

    np.testing.assert_allclose(
        updated_embedding_table_unsharded[sparse_rows],
        expected_embedding_table[sparse_rows],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        updated_accumulator_unsharded[sparse_rows],
        expected_accumulator[sparse_rows],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        updated_local_step_unsharded[sparse_rows],
        expected_local_step[sparse_rows],
        rtol=1e-4,
        atol=1e-4,
    )


if __name__ == "__main__":
  absltest.main()
