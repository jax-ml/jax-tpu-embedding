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
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_optimizer_grad
import numpy as np


class SparseDenseMatmulGradWithOptimizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = 4
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
      if isinstance(arg, jnp.ndarray) and arg.shape == emb_table.shape:
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
          if isinstance(arg, jnp.ndarray) and arg.shape == emb_table.shape
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
        sharded_out = einops.rearrange(
            out,
            "(v c s) f -> c (s v) f",
            c=1,
            s=4,
        )
        sharded_outputs.append(sharded_out[0])
      return tuple(sharded_outputs)
    else:
      updated_table = emb_table.at[unique_indices].set(outputs)
      updated_table_sharded = einops.rearrange(
          updated_table,
          "(v c s) f -> c (s v) f",
          c=1,
          s=4,
      )
      return updated_table_sharded[0]

  def test_sc_emb_backward_pass_with_sgd(self):
    # Process the input.
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

    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )
    emb_tables = [emb_table_sharded[0]]
    hyperparams = [0.01]

    def sgd_jax(grad, table, lr):
      return table - lr * grad

    emb_size = self.emb_size
    grad_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    lr_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    closed_jaxpr = jax.make_jaxpr(sgd_jax)(grad_aval, table_aval, lr_aval)

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
        jaxpr=closed_jaxpr,
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
        sgd_jax,
        hyperparams[0],
    )
    np.testing.assert_allclose(updated_emb_table, expected_updated_emb_table)

  def test_sc_emb_backward_pass_with_adagrad(self):
    # Process the input.
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
    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )

    accumulator_init = jnp.zeros(
        emb_table_sharded[0].shape,
        np.float32,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    emb_tables = [emb_table_sharded[0], accumulator_init]
    hyperparams = [0.01]

    def adagrad_jax(grad, table, accum, lr):
      new_accum = accum + grad * grad
      return table - lr * grad / jnp.sqrt(new_accum), new_accum

    emb_size = self.emb_size
    grad_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    accum_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    lr_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    closed_jaxpr = jax.make_jaxpr(adagrad_jax)(
        grad_aval, table_aval, accum_aval, lr_aval
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
            jaxpr=closed_jaxpr,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    global_accum_init = jnp.zeros_like(self.emb_table)
    expected_updated_table, expected_updated_accum = (
        self._get_expected_updated_table(
            self.emb_table,
            z_grad,
            self.input_tensor,
            self.input_weights,
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

    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
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

    closed_jaxpr = jax.make_jaxpr(ftrl_jax)(
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
            jaxpr=closed_jaxpr,
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
            hyperparams[0],
            hyperparams[1],
            hyperparams[2],
            hyperparams[3],
            hyperparams[4],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_accumulator, expected_updated_accum)
    np.testing.assert_allclose(updated_linear, expected_updated_linear)

  def test_sc_emb_backward_pass_with_adam(self):
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

    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.01,
        np.float32,
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

    emb_size = self.emb_size
    grad_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    table_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    m_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    v_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)

    alpha_t_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    beta_1_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    beta_2_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)
    epsilon_hat_aval = jax.ShapeDtypeStruct((1, emb_size), jnp.float32)

    closed_jaxpr = jax.make_jaxpr(adam_jax)(
        grad_aval,
        table_aval,
        m_aval,
        v_aval,
        alpha_t_aval,
        beta_1_aval,
        beta_2_aval,
        epsilon_hat_aval,
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
            jaxpr=closed_jaxpr,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    global_m_init = jnp.full_like(self.emb_table, 0.002, np.float32)
    global_v_init = jnp.full_like(self.emb_table, 0.004, np.float32)
    expected_updated_table, expected_updated_m, expected_updated_v = (
        self._get_expected_updated_table(
            self.emb_table,
            z_grad,
            self.input_tensor,
            self.input_weights,
            adam_jax,
            global_m_init,
            global_v_init,
            hyperparams[0],
            hyperparams[1],
            hyperparams[2],
            hyperparams[3],
        )
    )
    np.testing.assert_allclose(updated_table, expected_updated_table)
    np.testing.assert_allclose(updated_momentum, expected_updated_m)
    np.testing.assert_allclose(updated_velocity, expected_updated_v)


if __name__ == "__main__":
  absltest.main()
