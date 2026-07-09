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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


_NUM_SC_PER_DEVICE = utils.num_sparsecores_per_device(default_if_unknown=4)


class SparseDenseMatmulGradWithSgdWithMiniBatchingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_traceback_filtering", "off")

    self.num_chips = 1
    self.vocab_size = 32
    self.emb_size = 8

    # This is to shape the gradient tensor for backward pass.
    # The dimension needs to be the same for all test cases to avoid
    # recompilation.
    self.max_device_batch_size = 32

    # Define the embedding table.
    # Embedding table where row i is initialized to i.
    self.emb_table = np.tile(
        np.arange(self.vocab_size, dtype=np.float32)[:, np.newaxis],
        (1, self.emb_size),
    )
    self.global_devices = np.array([mock.create_autospec(jax.Device)])

    self._shard_table = functools.partial(
        utils.shard_emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )
    # Shard the embedding table.
    self.emb_table_sharded = self._shard_table(self.emb_table)
    logging.debug("self.emb_table_sharded: %s", self.emb_table_sharded)

    self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching = jax.named_call(
        sparse_dense_matmul_grad_with_sgd.tpu_sparse_dense_matmul_grad_with_sgd_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching",
    )

  def test_sc_emb_backward_pass(self):
    # Arrange
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
    input_weights = np.ones_like(input_tensor, dtype=np.float32)

    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        [input_tensor],
        [input_weights],
        mesh,
        num_sc_per_device=_NUM_SC_PER_DEVICE,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        enable_minibatching=True,
    )

    z_grad = jnp.full(
        (
            # The gradient is padded to max_device_batch_size, no matter how
            # many rows are actually used.
            # This is to make sure we don't have different gradient dimensions
            # among test cases to avoid recompilation.
            self.max_device_batch_size,
            self.emb_size,
        ),
        1.0,
        np.float32,
    )

    num_minibatches_per_physical_sparse_core = 1

    # Act
    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],  # pyrefly: ignore[bad-index]
            z_grad,
            0.1,  # learning_rate
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
        )
    )

    logging.debug("updated_emb_table: %s", updated_emb_table)

    # Assert
    # Check the embedding activations.
    actual_updated_unsharded = utils.unshard_emb_table(
        updated_emb_table[np.newaxis, :, :], num_sc_per_device=_NUM_SC_PER_DEVICE
    )
    # Compute the expected results on CPU while the primitive runs on TPU.
    # The optimizer only applies a sparse update: only rows involved in the
    # forward pass are updated.
    expected_unsharded = self.emb_table.copy()
    updated_rows = np.unique(input_tensor.flatten())
    expected_unsharded[updated_rows, :] -= 0.1

    np.testing.assert_allclose(
        actual_updated_unsharded, expected_unsharded, atol=1e-5
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="no_clipping",
          min_value=None,
          max_value=None,
      ),
      dict(
          testcase_name="clipping",
          min_value=2.0,
          max_value=12.0,
      ),
  )
  def test_sc_emb_backward_pass_2_batches_per_core(self, min_value, max_value):
    # Arrange
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
        num_sc_per_device=_NUM_SC_PER_DEVICE,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        enable_minibatching=True,
    )

    z_grad = jnp.full(
        (
            # The gradient is padded to max_device_batch_size, no matter how
            # many rows are actually used.
            # This is to make sure we don't have different gradient dimensions
            # among test cases to avoid recompilation.
            self.max_device_batch_size,
            self.emb_size,
        ),
        1.0,
        np.float32,
    )

    num_minibatches_per_physical_sparse_core = 2

    # Act
    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],  # pyrefly: ignore[bad-index]
            z_grad,
            0.1,  # learning_rate
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
            min_value=min_value,
            max_value=max_value,
        )
    )

    logging.debug("updated_emb_table: %s", updated_emb_table)

    # Assert
    # Check the embedding activations.
    actual_updated_unsharded = utils.unshard_emb_table(
        updated_emb_table[np.newaxis, :, :], num_sc_per_device=_NUM_SC_PER_DEVICE
    )
    # Compute the expected results on CPU while the primitive runs on TPU.
    # The optimizer only applies a sparse update: only rows involved in the
    # forward pass are updated.
    expected_unsharded = self.emb_table.copy()
    # Note that the expected updates are twice as large as the 1 batch case, for
    # we have 2 input samples for each embedding id.
    # For embedding id 0-15,
    # each has -1.0 (gradient) x 0.1 (learning rate) x 2 (samples) = -0.2
    # For embedding id 16-31, nothing is updated.
    updated_rows = np.unique(jax.tree_util.tree_flatten(features)[0])
    expected_unsharded[updated_rows, :] -= 0.2
    # Apply manual bounds clamping matching exactly what SparseCore should do.
    expected_unsharded[updated_rows, :] = np.clip(
        expected_unsharded[updated_rows, :], min_value, max_value
    )

    np.testing.assert_allclose(
        actual_updated_unsharded, expected_unsharded, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
