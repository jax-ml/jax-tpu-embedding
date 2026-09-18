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
from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_optimizer_grad
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


class SparseDenseMatmulGradWithOptimizerTest(parameterized.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = utils.num_sparsecores_per_device()
    self.input_tensor = np.array(
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
    self.input_weights = np.ones_like(self.input_tensor, np.float32)

    # Define the embedding table.
    self.emb_table = (
        np.array(
            [[i for _ in range(self.emb_size)] for i in range(self.vocab_size)]
        )
        .reshape(self.vocab_size, self.emb_size)
        .astype(np.float32)
    )
    self.global_devices = np.array([mock.create_autospec(jax.Device)])

    self.z_init = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.0,
        np.float32,
    )

    self.tpu_sparse_dense_matmul_grad_with_optimizer = jax.named_call(
        sparse_dense_matmul_optimizer_grad.tpu_sparse_dense_matmul_optimizer_grad_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_optimizer",
    )

  def _get_expected_updated_table(
      self, emb_table, z_grad, indices, weights, optimizer_fn, *rest_args
  ):
    emb_table = jnp.array(emb_table)
    nz_grads = weights * z_grad
    grad_table_full = jnp.zeros_like(emb_table)
    grad_table_full = grad_table_full.at[indices[:, 0]].add(nz_grads)

    unique_indices = jnp.unique(indices[:, 0])
    accessed_table = emb_table[unique_indices]
    accessed_grads = grad_table_full[unique_indices]

    accessed_rest_args = []
    for arg in rest_args:
      if (
          isinstance(arg, jnp.ndarray)
          and arg.ndim >= 1
          and arg.shape[0] == emb_table.shape[0]
      ):
        accessed_rest_args.append(arg[unique_indices])
      else:
        accessed_rest_args.append(arg)

    outputs = optimizer_fn(accessed_grads, accessed_table, *accessed_rest_args)

    if isinstance(outputs, tuple):
      updated_table = emb_table.at[unique_indices].set(outputs[0])
      updated_outputs = [updated_table]

      tensor_arg_indices = [
          i
          for i, arg in enumerate(rest_args)
          if (
              isinstance(arg, jnp.ndarray)
              and arg.ndim >= 1
              and arg.shape[0] == emb_table.shape[0]
          )
      ]

      for i, out in enumerate(outputs[1:]):
        if i < len(tensor_arg_indices):
          arg_idx = tensor_arg_indices[i]
          var = rest_args[arg_idx]
          # Pytype might think var can be a float because rest_args contains
          # floats. We assert it's an array to satisfy Pytype.
          assert isinstance(
              var, jnp.ndarray
          ), f"Expected array, got {type(var)}"
          updated_var = var.at[unique_indices].set(out)
          updated_outputs.append(updated_var)

      sharded_outputs = []
      for out in updated_outputs:
        sharded_out = utils.shard_emb_table(
            out,
            num_devices=1,
            num_sc_per_device=self.num_sc_per_device,
        )
        sharded_outputs.append(sharded_out[0])
      return tuple(sharded_outputs)
    else:
      updated_table = emb_table.at[unique_indices].set(outputs)
      updated_table_sharded = utils.shard_emb_table(
          updated_table,
          num_devices=1,
          num_sc_per_device=self.num_sc_per_device,
      )
      return updated_table_sharded[0]

  @parameterized.named_parameters(
      dict(testcase_name="2d", is_dim1=False),
      dict(testcase_name="dim1", is_dim1=True),
  )
  def test_sc_emb_backward_pass_with_sgd(self, is_dim1: bool):
    if is_dim1:
      input_tensor = np.array(
          [[i % self.vocab_size] for i in range(32)],
          dtype=np.int32,
      )
      input_weights = np.ones_like(input_tensor, dtype=np.float32)
      emb_table = np.arange(self.vocab_size, dtype=np.float32) + 1.0
      z_grad = jnp.full((32 // self.num_chips,), 0.01, np.float32)
      grad_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      lr_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      max_unique_ids = 64
      expected_input_weights = input_weights.squeeze()
    else:
      input_tensor = self.input_tensor
      input_weights = self.input_weights
      emb_table = self.emb_table
      z_grad = jnp.full(
          (self.batch_size // self.num_chips, self.emb_size),
          0.01,
          np.float32,
      )
      grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      lr_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      max_unique_ids = 16
      expected_input_weights = input_weights

    # Process the input.
    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_tables = [emb_table_sharded[0]]
    hyperparams = [0.01]

    def sgd_jax(grad, table, lr):
      return table - lr * grad

    stablehlo = (
        jax.jit(sgd_jax)
        .lower(grad_aval, table_aval, lr_aval)
        .as_text(dialect="stablehlo")
    )

    # Do the embedding update.
    (updated_emb_table,) = self.tpu_sparse_dense_matmul_grad_with_optimizer(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        np.int32(1),
        z_grad,
        *hyperparams,
        *emb_tables,
        num_hyperparameters=len(hyperparams),
        stablehlo=stablehlo,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=max_unique_ids,
        computation_name="optimizer_test_computation",
        sharding_strategy=1,
    )

    expected_updated_emb_table = self._get_expected_updated_table(
        emb_table,
        z_grad,
        input_tensor,
        expected_input_weights,
        sgd_jax,
        hyperparams[0],
    )
    np.testing.assert_allclose(updated_emb_table, expected_updated_emb_table)

  @parameterized.named_parameters(
      dict(testcase_name="2d", is_dim1=False),
      dict(testcase_name="dim1", is_dim1=True),
  )
  def test_sc_emb_backward_pass_with_adagrad(self, is_dim1: bool):
    if is_dim1:
      input_tensor = np.array(
          [[i % self.vocab_size] for i in range(32)],
          dtype=np.int32,
      )
      input_weights = np.ones_like(input_tensor, dtype=np.float32)
      emb_table = np.arange(self.vocab_size, dtype=np.float32) + 1.0
      z_grad = jnp.full((32 // self.num_chips,), 0.01, np.float32)
      grad_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      accum_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      lr_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      max_unique_ids = 64
      expected_input_weights = input_weights.squeeze()
    else:
      input_tensor = self.input_tensor
      input_weights = self.input_weights
      emb_table = self.emb_table
      z_grad = jnp.full(
          (self.batch_size // self.num_chips, self.emb_size),
          0.01,
          np.float32,
      )
      grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      accum_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      lr_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      max_unique_ids = 16
      expected_input_weights = input_weights

    # Process the input.
    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    accumulator_init = jnp.zeros(
        emb_table_sharded[0].shape,
        np.float32,
    )

    emb_tables = [emb_table_sharded[0], accumulator_init]
    hyperparams = [0.01]

    def adagrad_jax(grad, table, accum, lr):
      new_accum = accum + grad * grad
      return table - lr * grad / jnp.sqrt(new_accum), new_accum

    stablehlo = (
        jax.jit(adagrad_jax)
        .lower(grad_aval, table_aval, accum_aval, lr_aval)
        .as_text(dialect="stablehlo")
    )

    updated_table, updated_accumulator = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=max_unique_ids,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    global_accum_init = jnp.zeros_like(emb_table)
    expected_updated_table, expected_updated_accum = (
        self._get_expected_updated_table(
            emb_table,
            z_grad,
            input_tensor,
            expected_input_weights,
            adagrad_jax,
            global_accum_init,
            hyperparams[0],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_accumulator, expected_updated_accum)

  def test_sc_emb_backward_pass_with_ftrl(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor,
        self.input_weights,
        mesh,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    accumulator_init = np.full_like(emb_table_sharded[0], 0.1, np.float32)
    linear_init = np.full_like(emb_table_sharded[0], 0.01, np.float32)
    emb_tables = [emb_table_sharded[0], accumulator_init, linear_init]

    hyperparams = [0.1, -0.5, 0.001, 0.002, 0.01]

    def ftrl_jax(grad, table, accum, linear, lr, lr_power, l1, l2, beta):
      two = jnp.array(2.0, dtype=jnp.float32)
      zero = jnp.array(0.0, dtype=jnp.float32)

      a_new = accum + grad * grad
      p_old = jnp.power(accum, -lr_power)
      p_new = jnp.power(a_new, -lr_power)
      delta_p = p_new - p_old

      new_linear = linear + grad - (delta_p / lr) * table
      l_threshold = l1
      numerator = jnp.sign(new_linear) * l_threshold - new_linear
      abs_l_new = jnp.abs(new_linear)

      denominator = (p_new + beta) / lr + two * l2
      w_new = jnp.where(abs_l_new > l_threshold, numerator / denominator, zero)

      return w_new, a_new, new_linear

    emb_size = self.emb_size
    grad_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    accum_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    linear_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)

    lr_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    lr_power_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    l1_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    l2_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    beta_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)

    stablehlo = (
        jax.jit(ftrl_jax)
        .lower(
            grad_aval,
            table_aval,
            accum_aval,
            linear_aval,
            lr_aval,
            lr_power_aval,
            l1_aval,
            l2_aval,
            beta_aval,
        )
        .as_text(dialect="stablehlo")
    )

    updated_table, updated_accumulator, updated_linear = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )

    global_accum_init = jnp.full_like(self.emb_table, 0.1, np.float32)
    global_linear_init = jnp.full_like(self.emb_table, 0.01, np.float32)
    expected_updated_table, expected_updated_accum, expected_updated_linear = (
        self._get_expected_updated_table(
            self.emb_table,
            z_grad,
            self.input_tensor,
            self.input_weights,
            ftrl_jax,
            global_accum_init,
            global_linear_init,
            *hyperparams,
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_accumulator, expected_updated_accum)
    np.testing.assert_allclose(updated_linear, expected_updated_linear)

  @parameterized.named_parameters(
      dict(testcase_name="2d", is_dim1=False),
      dict(testcase_name="dim1", is_dim1=True),
  )
  def test_sc_emb_backward_pass_with_adam(self, is_dim1: bool):
    if is_dim1:
      input_tensor = np.array(
          [[i % self.vocab_size] for i in range(32)],
          dtype=np.int32,
      )
      input_weights = np.ones_like(input_tensor, dtype=np.float32)
      emb_table = np.arange(self.vocab_size, dtype=np.float32) + 1.0
      z_grad = jnp.full((32 // self.num_chips,), 0.01, np.float32)
      grad_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      m_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      v_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      alpha_t_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      beta_1_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      beta_2_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      epsilon_hat_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
      max_unique_ids = 64
      expected_input_weights = input_weights.squeeze()
    else:
      input_tensor = self.input_tensor
      input_weights = self.input_weights
      emb_table = self.emb_table
      z_grad = jnp.full(
          (self.batch_size // self.num_chips, self.emb_size),
          0.01,
          np.float32,
      )
      grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      m_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      v_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      alpha_t_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      beta_1_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      beta_2_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      epsilon_hat_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
      max_unique_ids = 16
      expected_input_weights = input_weights

    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    momentum_init = np.full_like(emb_table_sharded[0], 0.002, np.float32)
    velocity_init = np.full_like(emb_table_sharded[0], 0.004, np.float32)
    emb_tables = [emb_table_sharded[0], momentum_init, velocity_init]

    learning_rate = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    c_2 = np.sqrt(1.0 - beta_2)
    alpha_t = learning_rate * c_2 / (1.0 - beta_1)
    epsilon_hat = epsilon * c_2

    hyperparams = [alpha_t, beta_1, beta_2, epsilon_hat]

    def adam_jax(grad, table, m, v, alpha_t, beta_1, beta_2, epsilon_hat):
      new_m = beta_1 * m + (1.0 - beta_1) * grad
      new_v = beta_2 * v + (1.0 - beta_2) * (grad * grad)
      new_table = table - alpha_t * new_m / (jnp.sqrt(new_v) + epsilon_hat)
      return new_table, new_m, new_v

    stablehlo = (
        jax.jit(adam_jax)
        .lower(
            grad_aval,
            table_aval,
            m_aval,
            v_aval,
            alpha_t_aval,
            beta_1_aval,
            beta_2_aval,
            epsilon_hat_aval,
        )
        .as_text(dialect="stablehlo")
    )

    updated_table, updated_momentum, updated_velocity = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=max_unique_ids,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    global_m_init = jnp.full_like(emb_table, 0.002, np.float32)
    global_v_init = jnp.full_like(emb_table, 0.004, np.float32)
    expected_updated_table, expected_updated_m, expected_updated_v = (
        self._get_expected_updated_table(
            emb_table,
            z_grad,
            input_tensor,
            expected_input_weights,
            adam_jax,
            global_m_init,
            global_v_init,
            *hyperparams,
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_momentum, expected_updated_m)
    np.testing.assert_allclose(updated_velocity, expected_updated_v)

  def test_sc_emb_backward_pass_with_clipping(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor,
        self.input_weights,
        mesh,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        10.0,
        np.float32,
    )
    emb_tables = [emb_table_sharded[0]]
    hyperparams = [1.0]

    def sgd_jax(grad, table, lr):
      return table - lr * grad

    emb_size = self.emb_size
    grad_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    lr_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)

    min_value = -2.0
    max_value = 5.0

    def sgd_with_clip(grad, table, lr):
      return jnp.clip(sgd_jax(grad, table, lr), min_value, max_value)

    stablehlo = (
        jax.jit(sgd_with_clip)
        .lower(grad_aval, table_aval, lr_aval)
        .as_text(dialect="stablehlo")
    )

    (updated_emb_table,) = self.tpu_sparse_dense_matmul_grad_with_optimizer(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        np.int32(1),
        z_grad,
        *hyperparams,
        *emb_tables,
        num_hyperparameters=len(hyperparams),
        stablehlo=stablehlo,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="optimizer_test_computation",
        sharding_strategy=1,
    )

    expected_updated_emb_table = self._get_expected_updated_table(
        self.emb_table,
        z_grad,
        self.input_tensor,
        self.input_weights,
        sgd_with_clip,
        hyperparams[0],
    )
    np.testing.assert_allclose(updated_emb_table, expected_updated_emb_table)

  def test_sc_emb_backward_pass_with_rowwise_adagrad(self):
    input_tensor = np.array(
        [[i % self.vocab_size] for i in range(32)],
        dtype=np.int32,
    )
    input_weights = np.ones_like(input_tensor, dtype=np.float32)
    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    # 2D embedding table
    emb_table = (
        np.arange(self.vocab_size * self.emb_size, dtype=np.float32).reshape(
            self.vocab_size, self.emb_size
        )
        + 1.0
    )
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    # 1D accumulator (row-wise scalar)
    accumulator_init = jnp.zeros(
        (self.vocab_size,),
        np.float32,
    )
    accumulator_sharded = utils.shard_emb_table(
        accumulator_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    # 2D activation gradient
    z_grad = jnp.full(
        (32 // self.num_chips, self.emb_size),
        0.01,
        np.float32,
    )

    emb_tables = [emb_table_sharded[0], accumulator_sharded[0]]
    hyperparams = [0.01]

    def rowwise_adagrad_jax(grad, table, accum, lr):
      grad_sq = grad * grad
      new_accum = accum + jnp.sum(grad_sq, axis=-1)
      new_accum_2d = jax.lax.broadcast_in_dim(
          new_accum, shape=table.shape, broadcast_dimensions=(0,)
      )
      new_table = table - lr * grad / jnp.sqrt(new_accum_2d)
      return new_table, new_accum

    grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    accum_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
    lr_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    stablehlo = (
        jax.jit(rowwise_adagrad_jax)
        .lower(grad_aval, table_aval, accum_aval, lr_aval)
        .as_text(dialect="stablehlo")
    )

    updated_table, updated_accumulator = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=64,
            computation_name="optimizer_test_computation_rowwise_adagrad",
            sharding_strategy=1,
        )
    )
    global_accum_init = jnp.zeros_like(accumulator_init)
    expected_updated_table, expected_updated_accum = (
        self._get_expected_updated_table(
            emb_table,
            z_grad,
            input_tensor,
            input_weights,
            rowwise_adagrad_jax,
            global_accum_init,
            hyperparams[0],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_accumulator, expected_updated_accum)

  def test_sc_emb_backward_pass_with_rowwise_adam(self):
    input_tensor = np.array(
        [[i % self.vocab_size] for i in range(32)],
        dtype=np.int32,
    )
    input_weights = np.ones_like(input_tensor, dtype=np.float32)
    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table = (
        np.arange(self.vocab_size * self.emb_size, dtype=np.float32).reshape(
            self.vocab_size, self.emb_size
        )
        + 1.0
    )
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    momentum_init = np.full((self.vocab_size,), 0.002, np.float32)
    velocity_init = np.full((self.vocab_size,), 0.004, np.float32)
    m_sharded = utils.shard_emb_table(
        momentum_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    v_sharded = utils.shard_emb_table(
        velocity_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    emb_tables = [emb_table_sharded[0], m_sharded[0], v_sharded[0]]

    learning_rate = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    c_2 = np.sqrt(1.0 - beta_2)
    alpha_t = learning_rate * c_2 / (1.0 - beta_1)
    epsilon_hat = epsilon * c_2

    hyperparams = [alpha_t, beta_1, beta_2, epsilon_hat]

    def rowwise_adam_jax(
        grad, table, m, v, alpha_t, beta_1, beta_2, epsilon_hat
    ):
      grad_mean = jnp.mean(grad, axis=-1)
      grad_sq_mean = jnp.mean(grad * grad, axis=-1)
      beta_1_1d = jnp.mean(beta_1, axis=-1) if jnp.ndim(beta_1) > 0 else beta_1
      beta_2_1d = jnp.mean(beta_2, axis=-1) if jnp.ndim(beta_2) > 0 else beta_2
      new_m = beta_1_1d * m + (1.0 - beta_1_1d) * grad_mean
      new_v = beta_2_1d * v + (1.0 - beta_2_1d) * grad_sq_mean
      new_v_2d = jax.lax.broadcast_in_dim(
          new_v, shape=table.shape, broadcast_dimensions=(0,)
      )
      new_table = table - alpha_t * grad / (jnp.sqrt(new_v_2d) + epsilon_hat)
      return new_table, new_m, new_v

    grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    m_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
    v_aval = jax.ShapeDtypeStruct((1,), jnp.float32)

    alpha_t_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    beta_1_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    beta_2_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    epsilon_hat_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)

    stablehlo = (
        jax.jit(rowwise_adam_jax)
        .lower(
            grad_aval,
            table_aval,
            m_aval,
            v_aval,
            alpha_t_aval,
            beta_1_aval,
            beta_2_aval,
            epsilon_hat_aval,
        )
        .as_text(dialect="stablehlo")
    )

    z_grad = jnp.full(
        (32 // self.num_chips, self.emb_size),
        0.01,
        np.float32,
    )

    updated_table, updated_m, updated_v = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=64,
            computation_name="optimizer_test_computation_rowwise_adam",
            sharding_strategy=1,
        )
    )
    global_m_init = jnp.full_like(momentum_init, 0.002, np.float32)
    global_v_init = jnp.full_like(velocity_init, 0.004, np.float32)
    expected_updated_table, expected_updated_m, expected_updated_v = (
        self._get_expected_updated_table(
            emb_table,
            z_grad,
            input_tensor,
            input_weights,
            rowwise_adam_jax,
            global_m_init,
            global_v_init,
            hyperparams[0],
            hyperparams[1],
            hyperparams[2],
            hyperparams[3],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_m, expected_updated_m)
    np.testing.assert_allclose(updated_v, expected_updated_v)

  def test_sc_emb_backward_pass_with_mixed_slots_adam(self):
    input_tensor = np.array(
        [[i % self.vocab_size] for i in range(32)],
        dtype=np.int32,
    )
    input_weights = np.ones_like(input_tensor, dtype=np.float32)
    mesh = jax.sharding.Mesh(self.global_devices, "x")
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
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
    )

    emb_table = (
        np.arange(self.vocab_size * self.emb_size, dtype=np.float32).reshape(
            self.vocab_size, self.emb_size
        )
        + 1.0
    )
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    momentum_init = np.full((self.vocab_size, self.emb_size), 0.002, np.float32)
    velocity_init = np.full((self.vocab_size,), 0.004, np.float32)
    m_sharded = utils.shard_emb_table(
        momentum_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    v_sharded = utils.shard_emb_table(
        velocity_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    emb_tables = [emb_table_sharded[0], m_sharded[0], v_sharded[0]]

    learning_rate = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    c_2 = np.sqrt(1.0 - beta_2)
    alpha_t = learning_rate * c_2 / (1.0 - beta_1)
    epsilon_hat = epsilon * c_2

    hyperparams = [alpha_t, beta_1, beta_2, epsilon_hat]

    def mixed_adam_jax(grad, table, m, v, alpha_t, beta_1, beta_2, epsilon_hat):
      grad_sq_mean = jnp.mean(grad * grad, axis=-1)
      beta_2_1d = jnp.mean(beta_2, axis=-1) if jnp.ndim(beta_2) > 0 else beta_2
      new_m = beta_1 * m + (1.0 - beta_1) * grad
      new_v = beta_2_1d * v + (1.0 - beta_2_1d) * grad_sq_mean
      new_v_2d = jax.lax.broadcast_in_dim(
          new_v, shape=table.shape, broadcast_dimensions=(0,)
      )
      new_table = table - alpha_t * new_m / (jnp.sqrt(new_v_2d) + epsilon_hat)
      return new_table, new_m, new_v

    grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    m_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    v_aval = jax.ShapeDtypeStruct((1,), jnp.float32)

    alpha_t_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    beta_1_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    beta_2_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    epsilon_hat_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)

    stablehlo = (
        jax.jit(mixed_adam_jax)
        .lower(
            grad_aval,
            table_aval,
            m_aval,
            v_aval,
            alpha_t_aval,
            beta_1_aval,
            beta_2_aval,
            epsilon_hat_aval,
        )
        .as_text(dialect="stablehlo")
    )

    z_grad = jnp.full(
        (32 // self.num_chips, self.emb_size),
        0.01,
        np.float32,
    )

    updated_table, updated_m, updated_v = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            z_grad,
            *hyperparams,
            *emb_tables,
            num_hyperparameters=len(hyperparams),
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=64,
            computation_name="optimizer_test_computation_mixed_adam",
            sharding_strategy=1,
        )
    )
    global_m_init = jnp.full_like(momentum_init, 0.002, np.float32)
    global_v_init = jnp.full_like(velocity_init, 0.004, np.float32)
    expected_updated_table, expected_updated_m, expected_updated_v = (
        self._get_expected_updated_table(
            emb_table,
            z_grad,
            input_tensor,
            input_weights,
            mixed_adam_jax,
            global_m_init,
            global_v_init,
            hyperparams[0],
            hyperparams[1],
            hyperparams[2],
            hyperparams[3],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_m, expected_updated_m)
    np.testing.assert_allclose(updated_v, expected_updated_v)

  def test_sc_emb_backward_pass_with_minibatching_rowwise_adagrad(self):
    # Minibatching with 2 minibatches per physical sparse core.
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
        num_sc_per_device=self.num_sc_per_device,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        enable_minibatching=True,
    )

    emb_table = (
        np.arange(self.vocab_size * self.emb_size, dtype=np.float32).reshape(
            self.vocab_size, self.emb_size
        )
        + 1.0
    )
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    accumulator_init = jnp.full(
        (self.vocab_size,),
        0.1,
        np.float32,
    )
    accumulator_sharded = utils.shard_emb_table(
        accumulator_init,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    z_grad = jnp.full(
        (32, self.emb_size),
        1.0,
        np.float32,
    )

    num_minibatches_per_physical_sparse_core = 2

    def rowwise_adagrad_jax(grad, table, accum, lr):
      grad_sq = grad * grad
      new_accum = accum + jnp.sum(grad_sq, axis=-1)
      new_accum_2d = jax.lax.broadcast_in_dim(
          new_accum, shape=table.shape, broadcast_dimensions=(0,)
      )
      return table - lr * grad / jnp.sqrt(new_accum_2d), new_accum

    grad_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    accum_aval = jax.ShapeDtypeStruct((1,), jnp.float32)
    lr_aval = jax.ShapeDtypeStruct((1, self.emb_size), jnp.float32)
    stablehlo = (
        jax.jit(rowwise_adagrad_jax)
        .lower(grad_aval, table_aval, accum_aval, lr_aval)
        .as_text(dialect="stablehlo")
    )

    updated_table, updated_accumulator = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            z_grad,
            0.1,  # learning_rate
            emb_table_sharded[0],
            accumulator_sharded[0],
            num_hyperparameters=1,
            stablehlo=stablehlo,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation_mb_rowwise_adagrad",
            sharding_strategy=1,
            enable_minibatching=True,
        )
    )

    updated_table = utils.unshard_emb_table(
        jnp.stack([updated_table]), num_sc_per_device=self.num_sc_per_device
    )
    updated_accumulator = utils.unshard_emb_table(
        jnp.stack([updated_accumulator]),
        num_sc_per_device=self.num_sc_per_device,
    )

    expected_table = emb_table.copy()
    expected_accum = np.asarray(accumulator_init).copy()
    for emb_id in range(16):
      acc = expected_accum[emb_id].copy()
      tbl = expected_table[emb_id].copy()
      g = np.full((self.emb_size,), 2.0, dtype=np.float32)
      acc = acc + np.sum(g * g)
      tbl = tbl - 0.1 * g / np.sqrt(acc)
      expected_accum[emb_id] = acc
      expected_table[emb_id] = tbl

    np.testing.assert_allclose(updated_table, expected_table, atol=1e-5)
    np.testing.assert_allclose(updated_accumulator, expected_accum, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
