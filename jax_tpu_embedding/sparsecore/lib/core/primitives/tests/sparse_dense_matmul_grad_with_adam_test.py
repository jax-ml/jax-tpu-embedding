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
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adam
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithadamTest(parameterized.TestCase):
  row_pointers = np.array([0, 1, 2, 4], dtype=np.int32)
  sample_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  embedding_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  gains = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
  embedding_table = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  momentum = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  velocity = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  learning_rate = 0.001
  beta_1 = 0.9
  beta_2 = 0.999
  epsilon = 1e-8
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="momentum is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          momentum=momentum.astype(np.int32),
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="velocity is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          momentum=momentum,
          velocity=velocity.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad.astype(np.int32),
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="beta_1 is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=1,
          beta_2=beta_2,
          epsilon=epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="beta_2 is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=1,
          epsilon=epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="epsilon is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=1,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum.astype(np.int32),
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=np.array([[1.0, 2.0, 3.0]]),
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
          momentum=momentum,
          velocity=velocity,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          beta_1=beta_1,
          beta_2=beta_2,
          epsilon=epsilon,
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
      momentum,
      velocity,
      activations_grad,
      learning_rate,
      beta_1,
      beta_2,
      epsilon,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_adam.tpu_sparse_dense_matmul_grad_with_adam_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          embedding_table,
          momentum,
          velocity,
          activations_grad,
          learning_rate,
          beta_1,
          beta_2,
          epsilon,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )

  def _compute_adam(self, theta, g, m, v, alpha, beta_1, beta_2, epsilon, t):
    """Compute's the updated Adam states (theta, m, v)."""
    t = t + 1
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * (g * g)
    m_hat = m / (1 - jnp.power(beta_1, t))
    v_hat = v / (1 - jnp.power(beta_2, t))
    theta = theta - (alpha * m_hat) / (jnp.sqrt(v_hat) + epsilon)
    return (theta, m, v)

  def _shard_table(self, table):
    return einops.rearrange(
        table,
        "(v c s) f -> c (s v) f",
        c=1,  # Devices.
        s=4,  # SparseCores per device.
    )

  def _unshard_table(self, table):
    return einops.rearrange(
        table,
        "c (s v) f -> (v c s) f",
        c=1,  # Devices.
        s=4,  # SparseCores per device.
    )

  def _compute_table_grad(self, inputs, weights, activation_grad):
    # Assemble input as matrix:
    batch_size = activation_grad.shape[0]
    sample_lengths = jnp.array([len(sample) for sample in inputs])
    # Flatten all samples into one COO matrix.
    rows = jnp.repeat(jnp.arange(batch_size), sample_lengths)
    cols = jnp.concatenate(np.unstack(inputs))
    vals = jnp.concatenate(np.unstack(weights)).reshape(-1, 1)

    # grad = transpose(A) @ activation_grad
    grad = jnp.zeros(shape=(_VOCAB_SIZE, _EMB_SIZE))
    grad = grad.at[cols, :].add(vals * activation_grad[rows, :])
    return grad

  def test_adam_optimizer_update(self):
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
    embedding_table = (
        np.array(
            [[(i + 1) for _ in range(_EMB_SIZE)] for i in range(_VOCAB_SIZE)]
        )
        .reshape(_VOCAB_SIZE, _EMB_SIZE)
        .astype(np.float32)
    )
    embedding_table_sharded = self._shard_table(embedding_table)

    momentum = jnp.full(embedding_table.shape, 0.002, np.float32)
    momentum_sharded = self._shard_table(momentum)

    velocity = jnp.full(embedding_table.shape, 0.004, np.float32)
    velocity_sharded = self._shard_table(velocity)

    learning_rate = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    activations_grad = jnp.full((_BATCH_SIZE, _EMB_SIZE), 0.012, np.float32)
    table_grad = self._compute_table_grad(
        input_tensor, input_weights, activations_grad
    )

    # Compute the expected values.  The optimizer only applies a sparse update:
    # only rows involved in the forward pass are updated.
    sparse_rows = jnp.unique(jnp.concatenate(np.unstack(input_tensor)))
    sparse_update_mask = jnp.zeros(embedding_table.shape, dtype=jnp.bool)
    sparse_update_mask = sparse_update_mask.at[sparse_rows, :].set(True)
    expected_embedding_table, expected_momentum, expected_velocity = (
        self._compute_adam(
            embedding_table,
            table_grad,
            momentum,
            velocity,
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
            t=0,
        )
    )
    # Restore unaffected rows.
    expected_embedding_table = jnp.where(
        sparse_update_mask, expected_embedding_table, embedding_table
    )
    expected_momentum = jnp.where(
        sparse_update_mask, expected_momentum, momentum
    )
    expected_velocity = jnp.where(
        sparse_update_mask, expected_velocity, velocity
    )

    c_2 = jnp.sqrt(1.0 - beta_2)
    alpha_t = learning_rate * c_2 / (1.0 - beta_1)
    epsilon_hat = epsilon * c_2

    (updated_table, updated_momentum, updated_velocity) = (
        sparse_dense_matmul_grad_with_adam.tpu_sparse_dense_matmul_grad_with_adam_primitive.bind(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            embedding_table_sharded[0],
            momentum_sharded[0],
            velocity_sharded[0],
            activations_grad,
            alpha_t,
            beta_1,
            beta_2,
            epsilon_hat,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    updated_table = self._unshard_table(updated_table[jnp.newaxis, :, :])
    updated_momentum = self._unshard_table(updated_momentum[jnp.newaxis, :, :])
    updated_velocity = self._unshard_table(updated_velocity[jnp.newaxis, :, :])

    np.testing.assert_allclose(expected_momentum, updated_momentum)
    np.testing.assert_allclose(expected_velocity, updated_velocity)
    np.testing.assert_allclose(expected_embedding_table, updated_table)


if __name__ == "__main__":
  absltest.main()
