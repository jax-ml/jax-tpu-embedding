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

from absl import logging
from absl.testing import absltest
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd
import numpy as np


class SparseDenseMatmulGradWithSgdWithMiniBatchingTest(absltest.TestCase):
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

    # Shard the embedding table.
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=4,
    )
    logging.debug("self.emb_table_sharded: %s", self.emb_table_sharded)

    self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching = jax.named_call(
        sparse_dense_matmul_grad_with_sgd.tpu_sparse_dense_matmul_grad_with_sgd_primitive.bind,
        name="tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching",
    )

  def test_sc_emb_backward_pass(self):
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
        num_sc_per_device=4,
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

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],
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

    # Check the embedding activations.
    # For embedding id 0-15,
    # each has -1.0 (gradient) x 0.1 (learning rate) x 1 (sample) = -0.1
    # For embedding id 16-31, nothing is updated.
    expected_emb_activations = self.emb_table_sharded[0].copy()
    for i in range(16):
      # The table uses MOD sharding, so logical ID i is mapped to physical row:
      # (i % num_shards) * shard_size + (i // num_shards),
      # where num_shards=4 and shard_size=8.
      physical_row = (i % 4) * 8 + i // 4
      expected_emb_activations[physical_row, :] -= 0.1

    np.testing.assert_allclose(updated_emb_table, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core(self):
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
        num_sc_per_device=4,
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

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_physical_sparse_core,
            self.emb_table_sharded[0],
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

    # Check the embedding activations.
    # Note that the expected updates are twice as large as the 1 batch case, for
    # we have 2 input samples for each embedding id.
    # For embedding id 0-15,
    # each has -1.0 (gradient) x 0.1 (learning rate) x 2 (samples) = -0.2
    # For embedding id 16-31, nothing is updated.
    expected_emb_activations = self.emb_table_sharded[0].copy()
    for i in range(16):
      # The table uses MOD sharding, so logical ID i is mapped to physical row:
      # (i % num_shards) * shard_size + (i // num_shards),
      # where num_shards=4 and shard_size=8.
      physical_row = (i % 4) * 8 + i // 4
      expected_emb_activations[physical_row, :] -= 0.2

    np.testing.assert_allclose(
        updated_emb_table, expected_emb_activations, rtol=1e-5, atol=1e-5
    )

  @absltest.skip("b/496926428: Clipping with minibatching is not supported.")
  def test_sc_emb_forward_pass_2_batches_per_core_with_bounds(self):
    # Tests a single SC evaluating gradients from _BATCH_SIZE sequences.
    # The SC separates this batch logically into _NUM_MINIBATCHES chunks.
    batch_size = 16
    num_sc = 1
    num_minibatches_per_sc = 2
    max_ids_per_partition = 16
    assert batch_size % (num_sc * num_minibatches_per_sc) == 0

    mb0_feat = [[5], [3], [9, 1], [6, 12, 0]]
    mb1_feat = [[4], [15, 13, 11], [7, 8, 14, 2], [10]]
    mb0_weight = [[1.0], [1.0], [1.0, 1.0], [1.0, 1.0, 1.0]]
    mb1_weight = [[1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0]]

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
        num_sc_per_device=num_sc,
        max_ids_per_partition=max_ids_per_partition,
        max_unique_ids_per_partition=max_ids_per_partition,
        enable_minibatching=True,
    )

    emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=1,
    )

    # Provide gradients for the physical batch size.
    z_grad = jnp.full(
        (
            self.max_device_batch_size,
            self.emb_size,
        ),
        1.0,
        np.float32,
    )

    # Do the embedding update.
    updated_emb_table = (
        self.tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            num_minibatches_per_sc,
            emb_table_sharded[0],
            z_grad,
            0.1,  # learning_rate
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="sgd_minibatching_test_computation",
            sharding_strategy=1,
            enable_minibatching=True,
            min_value=2.0,
            max_value=12.0,
        )
    )

    # Embedding IDs 0-15 are each present once, so they are updated by -0.1*1*1
    # and clipped.
    expected_emb_activations = emb_table_sharded[0].copy()
    for i in range(16):
      expected_emb_activations[i, :] = np.clip(i - 0.1, 2.0, 12.0)
    updated_emb_table = einops.rearrange(
        updated_emb_table[jnp.newaxis, :, :],
        "c (s v) f -> (v c s) f",
        c=1,
        s=1,
    )

    logging.debug("updated %s", updated_emb_table)
    logging.debug("expected %s", expected_emb_activations)
    np.testing.assert_allclose(updated_emb_table, expected_emb_activations)


if __name__ == "__main__":
  absltest.main()
