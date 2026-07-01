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
import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


class SparseDenseMatmulGradWithSgdTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = 4
    self._shard_table = functools.partial(
        utils.shard_emb_table,
        num_devices=self.num_chips,
        num_sc_per_device=self.num_sc_per_device,
    )
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

    self.tpu_sparse_dense_matmul_grad_with_sgd = jax.named_call(
        sparse_dense_matmul_grad_with_sgd.tpu_sparse_dense_matmul_grad_with_sgd_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_sgd",
    )

  @parameterized.named_parameters(
      ("no_clipping", None, None),
      ("clipping", 2.0, 12.0),
  )
  def test_sc_emb_backward_pass(self, min_value, max_value):
    # Arrange
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
        num_sc_per_device=4,
    )

    emb_table_sharded = self._shard_table(
        self.emb_table,
    )

    z_grad = jnp.full(
        (
            self.batch_size // self.num_chips,
            self.emb_size,
        ),
        0.01,
        np.float32,
    )

    # Act

    # Do the embedding update.
    updated_emb_table = self.tpu_sparse_dense_matmul_grad_with_sgd(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        emb_table_sharded[0],  # pyrefly: ignore[bad-index]
        z_grad,
        0.01,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        computation_name="sgd_test_computation",
        sharding_strategy=1,
        min_value=min_value,
        max_value=max_value,
    )

    # Assert
    # Check the embedding activations.
    actual_emb_table_unsharded = utils.unshard_emb_table(
        updated_emb_table[np.newaxis, :, :], num_sc_per_device=4
    )

    # Compute the expected results on CPU while the primitive runs on TPU.
    # The optimizer only applies a sparse update: only rows involved in the
    # forward pass are updated.
    expected_emb_table_unsharded = self.emb_table.copy()
    updated_rows = np.unique(self.input_tensor.flatten())
    expected_emb_table_unsharded[updated_rows, :] -= 1e-4

    # Only clip updated rows!
    expected_emb_table_unsharded[updated_rows, :] = np.clip(
        expected_emb_table_unsharded[updated_rows, :], min_value, max_value
    )

    np.testing.assert_allclose(
        actual_emb_table_unsharded, expected_emb_table_unsharded, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
