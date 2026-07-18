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
import logging
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulCsrWithMiniBatchingValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update("jax_traceback_filtering", "off")

    self.global_devices = np.array([mock.create_autospec(jax.Device)])
    self.num_chips = 1
    self.num_sc_per_device = 4
    self.vocab_size = 32
    self.emb_size = 8
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
    self.input_weights = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    # Define the embedding table.
    self.emb_table = (
        np.array(
            [[i for _ in range(self.emb_size)] for i in range(self.vocab_size)]
        )
        .reshape(self.vocab_size, self.emb_size)
        .astype(np.float32)
    )

    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    self.tpu_sparse_dense_matmul_csr_with_mini_batching = jax.named_call(
        sparse_dense_matmul_csr.tpu_sparse_dense_matmul_csr_primitive.bind,
        name="tpu_sparse_dense_matmul_csr_with_mini_batching",
    )

  def test_sc_emb_forward_pass_invalid_input_dtypes(self):
    batch_size = 16
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
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    num_minibatches_per_physical_sparse_core = 1

    with self.subTest("invalid_row_pointer_type"):
      bad_row_pointers = np.array(lhs_row_pointers, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          bad_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

    with self.subTest("invalid_local_embedding_ids_type"):
      bad_local_embedding_ids = np.array(
          lhs_local_embedding_ids, dtype=np.float32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          bad_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

    with self.subTest("invalid_local_sample_ids_type"):
      bad_local_sample_ids = np.array(lhs_local_sample_ids, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

    with self.subTest("invalid_gains_type"):
      bad_gains = np.array(lhs_gains, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          bad_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

    with self.subTest("invalid_emb_table_type"):
      bad_emb_table = np.array(self.emb_table, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          bad_emb_table,
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

  def test_sc_emb_forward_pass_invalid_input_shapes(self):
    batch_size = 16
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
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    num_minibatches_per_physical_sparse_core = 1
    with self.subTest("invalid_sample_id_shape"):
      bad_sample_id = jnp.full(
          (len(lhs_local_sample_ids) - 1,), 2, dtype=np.int32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr_with_mini_batching,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_sample_id,
          lhs_gains,
          num_minibatches_per_physical_sparse_core,
          self.emb_table_sharded[0],
          device_batch_size=batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          enable_minibatching=True,
      )

  def test_sc_emb_forward_pass_invalid_max_ids_per_partition(self):
    batch_size = 16
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
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    num_minibatches_per_physical_sparse_core = 1
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=0,
        max_unique_ids_per_partition=256,
        sharding_strategy=1,
        enable_minibatching=True,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=0,
        sharding_strategy=1,
        enable_minibatching=True,
    )

  def test_sc_emb_forward_pass_invalid_sharding(self):
    batch_size = 16
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
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )

    num_minibatches_per_physical_sparse_core = 1
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr_with_mini_batching,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=256,
        sharding_strategy=2,
        enable_minibatching=True,
    )

  def test_sc_emb_forward_pass(self):
    # Process the input.
    batch_size = 16
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        embedding_ids,
        sample_ids,
        gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        [self.input_tensor],
        [self.input_weights],
        mesh,
        num_sc_per_device=4,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        enable_minibatching=True,
    )

    num_minibatches_per_physical_sparse_core = 1
    logging.debug(
        "num_minibatches_per_physical_sparse_core: %d",
        num_minibatches_per_physical_sparse_core,
    )

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        enable_minibatching=True,
    )

    logging.debug("emb_activations: %s", emb_activations)

    # Check the embedding activations.
    expected_emb_activations = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(emb_activations, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core(self):
    batch_size = 32
    num_minibatches_per_physical_sparse_core = 2

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
    expected_emb_activations = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        enable_minibatching=True,
    )

    logging.debug("emb_activations: %s", emb_activations)

    # Check the embedding activations.
    np.testing.assert_equal(emb_activations, expected_emb_activations)

  def test_sc_emb_forward_pass_2_batches_per_core_with_padded_tensors(self):
    batch_size = 32
    num_minibatches_per_physical_sparse_core = 2

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
    expected_emb_activations = np.array(
        [
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )

    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr_with_mini_batching(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        num_minibatches_per_physical_sparse_core,
        self.emb_table_sharded[0],
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        enable_minibatching=True,
    )

    logging.debug("emb_activations: %s", emb_activations)
    np.testing.assert_equal(emb_activations, expected_emb_activations)


if __name__ == "__main__":
  absltest.main()
