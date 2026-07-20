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
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


class SparseDenseMatmulCsrTest(absltest.TestCase):

  emb_table_sharded: np.ndarray

  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])
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
    self.global_devices = np.array([mock.create_autospec(jax.Device)])

    self.tpu_sparse_dense_matmul_csr = jax.named_call(
        sparse_dense_matmul_csr.tpu_sparse_dense_matmul_csr_primitive.bind,
        name="tpu_sparse_dense_matmul_csr",
    )

  def test_sc_emb_forward_pass_invalid_input_dtypes(self):
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
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    with self.subTest("invalid_row_pointer_type"):
      bad_row_pointers = np.array(lhs_row_pointers, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          bad_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

    with self.subTest("invalid_local_embedding_ids_type"):
      bad_local_embedding_ids = np.array(
          lhs_local_embedding_ids, dtype=np.float32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          lhs_row_pointers,
          bad_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

    with self.subTest("invalid_local_sample_ids_type"):
      bad_local_sample_ids = np.array(lhs_local_sample_ids, dtype=np.float32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_local_sample_ids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

    with self.subTest("invalid_gains_type"):
      bad_gains = np.array(lhs_gains, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          bad_gains,
          1,  # num_minibatches_per_physical_sparse_core
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

    with self.subTest("invalid_emb_table_type"):
      bad_emb_table = np.array(self.emb_table, dtype=np.int32)
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          bad_emb_table,
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

  def test_sc_emb_forward_pass_invalid_input_shapes(self):
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
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    with self.subTest("invalid_sample_id_shape"):
      bad_sample_id = jnp.full(
          (len(lhs_local_sample_ids) - 1,), 2, dtype=np.int32
      )
      self.assertRaises(
          ValueError,
          self.tpu_sparse_dense_matmul_csr,
          lhs_row_pointers,
          lhs_local_embedding_ids,
          bad_sample_id,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
          quantization_config=None,
          enable_minibatching=False,
      )

  def test_sc_emb_forward_pass_invalid_max_ids_per_partition(self):
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
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=0,
        max_unique_ids_per_partition=256,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=0,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )

  def test_sc_emb_forward_pass_invalid_sharding(self):
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
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=256,
        sharding_strategy=2,
        quantization_config=None,
        enable_minibatching=False,
    )

  def test_sc_emb_forward_pass(self):
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
    # Shared the embedding table.
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )

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

  def test_sc_emb_quantization_config_validation(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    lhs_row_pointers, lhs_ids, lhs_sids, lhs_gains = (
        input_preprocessing.preprocess_sparse_dense_matmul_input(
            self.input_tensor,
            self.input_weights,
            mesh,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=64,
            num_sc_per_device=self.num_sc_per_device,
        )
    )
    emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    # num_buckets must be >= 2
    with self.assertRaises(ValueError):
      self.tpu_sparse_dense_matmul_csr(
          lhs_row_pointers,
          lhs_ids,
          lhs_sids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          sharding_strategy=1,
          # num_buckets < 2
          quantization_config=(0.0, 1.0, 1),
          enable_minibatching=False,
      )

    # min must be < max
    with self.assertRaises(ValueError):
      self.tpu_sparse_dense_matmul_csr(
          lhs_row_pointers,
          lhs_ids,
          lhs_sids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          sharding_strategy=1,
          # min < max
          quantization_config=(5.0, 5.0, 4),
          enable_minibatching=False,
      )

  def test_sc_emb_forward_pass_with_quantization_enabled(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    lhs_row_pointers, lhs_ids, lhs_sids, lhs_gains = (
        input_preprocessing.preprocess_sparse_dense_matmul_input(
            self.input_tensor,
            self.input_weights,
            mesh,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=64,
            num_sc_per_device=self.num_sc_per_device,
        )
    )
    emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    activations = self.tpu_sparse_dense_matmul_csr(
        lhs_row_pointers,
        lhs_ids,
        lhs_sids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        # valid config
        quantization_config=(0.0, 15.0, 256),
        enable_minibatching=False,
    )

    # Quantization happens on-device, so numerical values stay identical
    # for the table.
    self.assertEqual(activations.shape, (self.batch_size, self.emb_size))
    self.assertEqual(activations.dtype, jnp.float32)

  def test_sc_emb_forward_pass_dim1(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    # Input tensor with 2 duplicate IDs per sample
    single_batch_2col = np.concatenate(
        [self.input_tensor, self.input_tensor], axis=1
    )
    input_tensor = np.concatenate(
        [single_batch_2col, single_batch_2col], axis=0
    )
    single_weights_2col = np.concatenate(
        [self.input_weights, self.input_weights], axis=1
    )
    input_weights = np.concatenate(
        [single_weights_2col, single_weights_2col], axis=0
    )
    batch_size = self.batch_size * 2
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
    # Define embedding table with dim=1 (1D array) and non-trivial values
    emb_table_dim1 = np.arange(self.vocab_size, dtype=np.float32) + 1.0
    emb_table_sharded = utils.shard_emb_table(
        emb_table_dim1,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    activations = self.tpu_sparse_dense_matmul_csr(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        1,  # num_minibatches_per_physical_sparse_core
        emb_table_sharded[0],  # pyrefly: ignore[bad-index]
        device_batch_size=batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )

    # Each sample looks up 2 duplicate IDs: (table[id]) * 2
    single_expected = (self.input_tensor.squeeze() + 1.0) * 2.0
    expected_activations = np.concatenate([single_expected, single_expected])
    np.testing.assert_allclose(
        activations, expected_activations, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
