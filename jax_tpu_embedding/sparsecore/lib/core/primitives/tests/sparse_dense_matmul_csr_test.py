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
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


class SparseDenseMatmulCsrTest(absltest.TestCase):

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
        num_sc_per_device=self.num_sc_per_device,
    )
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=self.num_sc_per_device,
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
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
          bad_emb_table,
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
        num_sc_per_device=self.num_sc_per_device,
    )
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=self.num_sc_per_device,
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
          self.emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=256,
          max_unique_ids_per_partition=256,
          sharding_strategy=1,
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
        num_sc_per_device=self.num_sc_per_device,
    )
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=self.num_sc_per_device,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=0,
        max_unique_ids_per_partition=256,
        sharding_strategy=1,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=0,
        sharding_strategy=1,
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
        num_sc_per_device=self.num_sc_per_device,
    )
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=self.num_sc_per_device,
    )
    self.assertRaises(
        ValueError,
        self.tpu_sparse_dense_matmul_csr,
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=256,
        max_unique_ids_per_partition=256,
        sharding_strategy=2,
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
        num_sc_per_device=self.num_sc_per_device,
    )
    # Shared the embedding table.
    self.emb_table_sharded = einops.rearrange(
        self.emb_table,
        "(v c s) f -> c (s v) f",
        c=len(self.global_devices),
        s=self.num_sc_per_device,
    )
    # Do the embedding lookup.
    emb_activations = self.tpu_sparse_dense_matmul_csr(
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
        self.emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
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


if __name__ == "__main__":
  absltest.main()
