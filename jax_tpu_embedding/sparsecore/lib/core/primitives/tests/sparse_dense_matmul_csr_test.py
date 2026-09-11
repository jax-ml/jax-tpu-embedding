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
import importlib.metadata
import re
from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


def _is_libtpu_at_least(min_version: tuple[int, int, int]) -> bool:
  """Returns True if libtpu is not installed or is at least min_version."""
  try:
    version_str = importlib.metadata.version("libtpu")
  except importlib.metadata.PackageNotFoundError:
    return True
  match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
  if match is None:
    return True
  version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
  return version >= min_version


def _absmax_fake_quant(table: np.ndarray, num_buckets: int = 256) -> np.ndarray:
  """NumPy reference of SparseCore absmax simulated quantization (per row).

  Follows the op order of the SparseCore decomposer so results match in f32:
  scale by `bound / m`, round with `floor(x + 0.5)`, and dequantize by
  `m * (1 / bound)`.

  Args:
    table: A [vocab, dim] float32 table.
    num_buckets: Number of quantization buckets.

  Returns:
    The quantized-then-dequantized table.
  """
  table = np.asarray(table, dtype=np.float32)
  bound = np.float32(np.floor((num_buckets - 1) / 2))
  m = np.maximum(
      np.max(np.abs(table), axis=-1, keepdims=True), np.float32(1e-7)
  )
  q = np.floor(table * (bound / m) + np.float32(0.5))
  return (q * (m * (np.float32(1.0) / bound))).astype(np.float32)


class SparseDenseMatmulCsrTest(parameterized.TestCase):

  emb_table_sharded: jax.Array

  @override
  def setUp(self):
    super().setUp()
    self.num_chips = 1
    self.batch_size = 16
    self.vocab_size = 32
    self.emb_size = 8
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])
    self.sc_simd_width = utils.sparsecore_simd_width(jax.devices()[0])
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
    self.emb_table = test_utils.row_id_initializer(
        (self.vocab_size, self.emb_size)
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
        sc_simd_width=self.sc_simd_width,
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
        sc_simd_width=self.sc_simd_width,
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
        sc_simd_width=self.sc_simd_width,
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
        sc_simd_width=self.sc_simd_width,
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

  @parameterized.named_parameters(
      # 8 floats is the HBM word size. 5 and 21 are not multiples of it, and
      # exercise the sub-word and multi-word-with-remainder cases respectively.
      dict(testcase_name="dim_8", emb_size=8),
      dict(testcase_name="dim_5", emb_size=5),
      dict(testcase_name="dim_21", emb_size=21),
  )
  def test_sc_emb_forward_pass(self, emb_size: int):
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
        sc_simd_width=self.sc_simd_width,
    )
    # Define and shard an embedding table with the parameterized width.
    emb_table = test_utils.row_id_initializer((self.vocab_size, emb_size))
    emb_table_sharded = utils.shard_emb_table(
        emb_table,
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
        emb_table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )

    # Check the embedding activations. Each sample looks up a single id, and
    # row i of the table is filled with i, so the activation for a sample is
    # its id repeated emb_size times.
    expected_emb_activations = np.tile(
        self.input_tensor.astype(np.float32), (1, emb_size)
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
            sc_simd_width=self.sc_simd_width,
        )
    )
    emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    # Fixed quantization requires num_buckets >= 2.
    with self.assertRaises(ValueError):
      embedding_spec.FixedQuantizationConfig(
          min_value=0.0, max_value=1.0, num_buckets=1
      )

    # Absmax quantization requires num_buckets >= 3.
    with self.assertRaises(ValueError):
      embedding_spec.AbsmaxQuantizationConfig(num_buckets=1)

    # min must be < max
    with self.assertRaises(ValueError):
      embedding_spec.FixedQuantizationConfig(
          min_value=5.0, max_value=5.0, num_buckets=4
      )

    # Unsupported quantization configs, including the legacy
    # (min_value, max_value, num_buckets) tuple.
    for bad_config in ("unsupported", (0.0, 15.0, 256)):
      with self.subTest(bad_config=bad_config):
        with self.assertRaisesRegex(
            ValueError, "Unsupported quantization_config"
        ):
          self.tpu_sparse_dense_matmul_csr(
              lhs_row_pointers,
              lhs_ids,
              lhs_sids,
              lhs_gains,
              1,
              emb_table_sharded[0],
              device_batch_size=self.batch_size // self.num_chips,
              max_ids_per_partition=16,
              max_unique_ids_per_partition=16,
              sharding_strategy=1,
              quantization_config=bad_config,
              enable_minibatching=False,
          )

  @absltest.skipIf(
      not _is_libtpu_at_least((0, 0, 49)), "Requires libtpu >= 0.0.49"
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
            sc_simd_width=self.sc_simd_width,
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
        quantization_config=embedding_spec.FixedQuantizationConfig(
            min_value=0.0, max_value=15.0, num_buckets=256
        ),
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
        sc_simd_width=self.sc_simd_width,
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
        emb_table_sharded[0],
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

  def test_sc_emb_forward_pass_dim1_dynamic_quantization_unsupported(self):
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
        sc_simd_width=self.sc_simd_width,
    )
    emb_table_dim1 = np.arange(self.vocab_size, dtype=np.float32) + 1.0
    emb_table_sharded = utils.shard_emb_table(
        emb_table_dim1,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    with self.assertRaisesRegex(
        ValueError,
        "Dynamic quantization.*not supported for 1D scalar embeddings",
    ):
      self.tpu_sparse_dense_matmul_csr(
          lhs_row_pointers,
          lhs_local_embedding_ids,
          lhs_local_sample_ids,
          lhs_gains,
          1,  # num_minibatches_per_physical_sparse_core
          emb_table_sharded[0],
          device_batch_size=self.batch_size // self.num_chips,
          max_ids_per_partition=16,
          max_unique_ids_per_partition=16,
          sharding_strategy=1,
          quantization_config=embedding_spec.AbsmaxQuantizationConfig(
              num_buckets=256
          ),
          enable_minibatching=False,
      )

  @absltest.skipIf(
      not _is_libtpu_at_least((0, 0, 49)), "Requires libtpu >= 0.0.49"
  )
  def test_sc_emb_forward_pass_dynamic_quantization_feature_widths(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_ids,
        lhs_sids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor,
        self.input_weights,
        mesh,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
        sc_simd_width=self.sc_simd_width,
    )
    for feature_width in (8, 16, 32, 64, 128):
      with self.subTest(feature_width=feature_width):
        # Non-uniform rows (see the numerical parity test) so quantization
        # changes the values.
        rows = np.arange(1, self.vocab_size + 1, dtype=np.float32)[:, None]
        cols = (np.arange(feature_width) % 7 + 1).astype(np.float32)[None, :]
        table = rows * cols / np.float32(7.0)
        table_sharded = utils.shard_emb_table(
            table,
            num_devices=len(self.global_devices),
            num_sc_per_device=self.num_sc_per_device,
        )
        activations = self.tpu_sparse_dense_matmul_csr(
            lhs_row_pointers,
            lhs_ids,
            lhs_sids,
            lhs_gains,
            1,
            table_sharded[0],
            device_batch_size=self.batch_size // self.num_chips,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            sharding_strategy=1,
            quantization_config=embedding_spec.AbsmaxQuantizationConfig(
                num_buckets=256
            ),
            enable_minibatching=False,
        )
        self.assertEqual(
            activations.shape,
            (self.batch_size // self.num_chips, feature_width),
        )
        self.assertEqual(activations.dtype, jnp.float32)
        np.testing.assert_allclose(
            activations,
            _absmax_fake_quant(table)[self.input_tensor.squeeze(axis=1)],
            rtol=1e-5,
            atol=1e-5,
        )

  @absltest.skipIf(
      not _is_libtpu_at_least((0, 0, 49)), "Requires libtpu >= 0.0.49"
  )
  def test_sc_emb_forward_pass_dynamic_quantization_numerical_parity(self):
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    (
        lhs_row_pointers,
        lhs_ids,
        lhs_sids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.input_tensor,
        self.input_weights,
        mesh,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=64,
        num_sc_per_device=self.num_sc_per_device,
        sc_simd_width=self.sc_simd_width,
    )
    # Row r is (r + 1) * [1, 2, ..., 7, 1] / 7, so absmax quantization changes
    # it: its absmax is r + 1 and each element scales to bound * k / 7 (bound is
    # 31 or 127 here), which is never within 0.07 of a rounding boundary.
    rows = np.arange(1, self.vocab_size + 1, dtype=np.float32)[:, None]
    cols = (np.arange(self.emb_size) % 7 + 1).astype(np.float32)[None, :]
    table = rows * cols / np.float32(7.0)
    table_sharded = utils.shard_emb_table(
        table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )

    activations_unquant = self.tpu_sparse_dense_matmul_csr(
        lhs_row_pointers,
        lhs_ids,
        lhs_sids,
        lhs_gains,
        1,
        table_sharded[0],
        device_batch_size=self.batch_size // self.num_chips,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        sharding_strategy=1,
        quantization_config=None,
        enable_minibatching=False,
    )

    for num_buckets in (64, 256):
      with self.subTest(num_buckets=num_buckets):
        activations_dynamic = self.tpu_sparse_dense_matmul_csr(
            lhs_row_pointers,
            lhs_ids,
            lhs_sids,
            lhs_gains,
            1,
            table_sharded[0],
            device_batch_size=self.batch_size // self.num_chips,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            sharding_strategy=1,
            quantization_config=embedding_spec.AbsmaxQuantizationConfig(
                num_buckets=num_buckets
            ),
            enable_minibatching=False,
        )

        self.assertEqual(
            activations_dynamic.shape,
            (self.batch_size // self.num_chips, self.emb_size),
        )
        # Each sample looks up a single ID with weight 1.
        expected_activations = _absmax_fake_quant(table, num_buckets)[
            self.input_tensor.squeeze(axis=1)
        ]
        np.testing.assert_allclose(
            activations_dynamic, expected_activations, rtol=1e-5, atol=1e-5
        )
        # Fails if absmax quantization isn't applied.
        self.assertFalse(
            np.allclose(activations_dynamic, activations_unquant, atol=1e-3)
        )


if __name__ == "__main__":
  absltest.main()
