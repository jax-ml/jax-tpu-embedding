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
from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import local_sparse_dense_matmul
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

jax.config.update("jax_enable_x64", True)


class SparseDenseMatmulCsrTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])
    self.embedding_ids = np.asarray(
        [5, 3, 9, 1, 6, 12, 0, 4, 15, 13, 11, 7, 8, 14, 2, 10],
        dtype=np.int64,
    )
    self.sample_ids = np.arange(self.batch_size, dtype=np.int32)
    self.gains = np.ones_like(self.embedding_ids, dtype=np.float32)
    # Define the embedding table.
    self.emb_table = (
        np.array(
            [[i for _ in range(self.emb_size)] for i in range(self.vocab_size)]
        )
        .reshape(self.vocab_size, self.emb_size)
        .astype(np.float32)
    )

    self.tpu_local_sparse_dense_matmul = jax.named_call(
        local_sparse_dense_matmul.tpu_local_sparse_dense_matmul_primitive.bind,
        name="tpu_local_sparse_dense_matmul",
    )

  def test_sc_emb_forward_pass_invalid_input_dtypes(self):
    with self.subTest("invalid_local_embedding_ids_type"):
      with self.assertRaises(ValueError):
        self.tpu_local_sparse_dense_matmul(
            self.embedding_ids.astype(jnp.float64),
            self.sample_ids,
            self.gains,
            self.emb_table,
            device_batch_size=self.batch_size,
        )

    with self.subTest("invalid_local_sample_ids_type"):
      with self.assertRaises(ValueError):
        self.tpu_local_sparse_dense_matmul(
            self.embedding_ids,
            self.sample_ids.astype(jnp.float32),
            self.gains,
            self.emb_table,
            device_batch_size=self.batch_size,
        )

    with self.subTest("invalid_gains_type"):
      with self.assertRaises(ValueError):
        self.tpu_local_sparse_dense_matmul(
            self.embedding_ids,
            self.sample_ids,
            self.gains.astype(jnp.int32),
            self.emb_table,
            device_batch_size=self.batch_size,
        )

    with self.subTest("invalid_emb_table_type"):
      with self.assertRaises(ValueError):
        self.tpu_local_sparse_dense_matmul(
            self.embedding_ids,
            self.sample_ids,
            self.gains,
            self.emb_table.astype(jnp.int32),
            device_batch_size=self.batch_size,
        )

  def test_sc_emb_forward_pass_invalid_input_shapes(self):
    with self.subTest("invalid_sample_id_shape"):
      ids = self.embedding_ids.reshape(4, 4)
      with self.assertRaises(ValueError):
        self.tpu_local_sparse_dense_matmul(
            ids,
            self.sample_ids,
            self.gains,
            self.emb_table,
            device_batch_size=self.batch_size,
        )

  def test_sc_emb_forward_pass(self):
    # Do the embedding lookup.
    emb_activations = self.tpu_local_sparse_dense_matmul(
        jnp.asarray(self.embedding_ids, dtype=jnp.int32),
        jnp.asarray(self.sample_ids, dtype=jnp.int32),
        jnp.asarray(self.gains, dtype=jnp.float32),
        jnp.asarray(self.emb_table, dtype=jnp.float32),
        device_batch_size=self.batch_size // self.num_chips,
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
