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
from jax_tpu_embedding.sparsecore.lib.core.primitives import optimizers_computation
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
    emb_tables = np.array([emb_table_sharded[0]])
    hyperparams = np.array([0.01])
    # Do the embedding update.
    (updated_emb_table,) = self.tpu_sparse_dense_matmul_grad_with_optimizer(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        np.int32(1),
        emb_tables,
        z_grad,
        hyperparams,
        optimizer_generator=optimizers_computation.sgd,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="optimizer_test_computation",
        sharding_strategy=1,
    )

    # Check the embedding activations.
    expected_emb_activations = np.array(
        [
            [
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
                -1.00000e-04,
            ],
            [
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
                3.99990e00,
            ],
            [
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
                7.99990e00,
            ],
            [
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
                1.19999e01,
            ],
            [
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
                1.60000e01,
            ],
            [
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
                2.00000e01,
            ],
            [
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
                2.40000e01,
            ],
            [
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
                2.80000e01,
            ],
            [
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
                9.99900e-01,
            ],
            [
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
                4.99990e00,
            ],
            [
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
                8.99990e00,
            ],
            [
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
                1.29999e01,
            ],
            [
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
                1.70000e01,
            ],
            [
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
                2.10000e01,
            ],
            [
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
                2.50000e01,
            ],
            [
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
                2.90000e01,
            ],
            [
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
                1.99990e00,
            ],
            [
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
                5.99990e00,
            ],
            [
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
                9.99990e00,
            ],
            [
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
                1.39999e01,
            ],
            [
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
                1.80000e01,
            ],
            [
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
                2.20000e01,
            ],
            [
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
                2.60000e01,
            ],
            [
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
                3.00000e01,
            ],
            [
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
                2.99990e00,
            ],
            [
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
                6.99990e00,
            ],
            [
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
                1.09999e01,
            ],
            [
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
                1.49999e01,
            ],
            [
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
                1.90000e01,
            ],
            [
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
                2.30000e01,
            ],
            [
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
                2.70000e01,
            ],
            [
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
                3.10000e01,
            ],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(updated_emb_table, expected_emb_activations)

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
    emb_tables = np.array([emb_table_sharded[0], accumulator_init])
    hyperparams = np.array([0.01])
    (updated_table, updated_accumulator) = (
        self.tpu_sparse_dense_matmul_grad_with_optimizer(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            np.int32(1),
            emb_tables,
            z_grad,
            hyperparams,
            optimizer_generator=optimizers_computation.adagrad,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )
    np.testing.assert_equal(expected_accumulator, updated_accumulator)
    np.testing.assert_equal(expected_table, updated_table)


if __name__ == "__main__":
  absltest.main()
