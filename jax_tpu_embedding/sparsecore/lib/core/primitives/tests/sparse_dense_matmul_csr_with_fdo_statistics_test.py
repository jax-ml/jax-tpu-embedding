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
from typing import override
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr_with_fdo_statistics
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_MAX_IDS_PER_PARTITION = 32
_MAX_UNIQUE_IDS_PER_PARTITION = 32


class SparseDenseMatmulCsrWithFdoStatisticsTest(parameterized.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.global_devices = np.array([mock.create_autospec(jax.Device)])
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])
    self.sc_simd_width = utils.sparsecore_simd_width(jax.devices()[0])
    self.hbm_word_size_in_4b = utils.hbm_word_size_in_4b(jax.devices()[0])
    self.fdo_statistics_size = self.num_sc_per_device * self.hbm_word_size_in_4b

    # Each sample looks up two IDs. In the first half of the batch, every sample
    # looks up ID 0, so the partition of ID 0 receives many IDs but few unique
    # IDs, and the two FDO statistics differ. The second half of the batch has
    # fewer repeated IDs, so the FDO statistics also differ across SparseCores.
    half_batch_size = _BATCH_SIZE // 2
    self.input_tensor = np.array(
        [[0, i + 1] for i in range(half_batch_size)]
        + [[i, i + 1] for i in range(half_batch_size, _BATCH_SIZE)],
        dtype=np.int32,
    )
    self.input_weights = np.ones_like(self.input_tensor, dtype=np.float32)

    # Row i of the table is filled with i.
    self.emb_table = test_utils.row_id_initializer((_VOCAB_SIZE, _EMB_SIZE))
    self.emb_table_sharded = utils.shard_emb_table(
        self.emb_table,
        num_devices=len(self.global_devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    # With the sum combiner, the activation of a sample is the sum of its IDs.
    self.expected_activations = np.tile(
        np.sum(self.input_tensor, axis=1, keepdims=True).astype(np.float32),
        (1, _EMB_SIZE),
    )

    self.primitive = (
        sparse_dense_matmul_csr_with_fdo_statistics.tpu_sparse_dense_matmul_csr_with_fdo_statistics_primitive
    )

  def _preprocess_input(self, enable_minibatching: bool):
    """Returns the row pointers, embedding IDs, sample IDs and gains."""
    mesh = jax.sharding.Mesh(self.global_devices, "x")
    preprocess = functools.partial(
        input_preprocessing.preprocess_sparse_dense_matmul_input,
        mesh=mesh,
        max_ids_per_partition=_MAX_IDS_PER_PARTITION,
        max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
        num_sc_per_device=self.num_sc_per_device,
        sc_simd_width=self.sc_simd_width,
        hbm_word_size_in_4b=self.hbm_word_size_in_4b,
    )
    if enable_minibatching:
      # A single minibatch.
      return preprocess(
          [self.input_tensor],
          [self.input_weights],
          minibatching_mode=input_preprocessing.MinibatchingMode.HOST,
      )
    return preprocess(self.input_tensor, self.input_weights)

  def _expected_fdo_statistics(self) -> tuple[np.ndarray, np.ndarray]:
    """Returns the expected `(max_ids, max_unique_ids)` FDO statistics."""
    max_ids = []
    max_unique_ids = []
    # Each SparseCore preprocesses a contiguous slice of the samples, and with
    # MOD sharding it sends an ID to the partition `ID % num_sc_per_device`.
    for sc_ids in np.split(self.input_tensor, self.num_sc_per_device):
      ids_per_partition = [
          sc_ids[sc_ids % self.num_sc_per_device == partition]
          for partition in range(self.num_sc_per_device)
      ]
      max_ids.append(max(ids.size for ids in ids_per_partition))
      max_unique_ids.append(
          max(np.unique(ids).size for ids in ids_per_partition)
      )
    # Every element in a SparseCore's HBM word holds the same value.
    return (
        np.repeat(np.array(max_ids, np.int32), self.hbm_word_size_in_4b),
        np.repeat(np.array(max_unique_ids, np.int32), self.hbm_word_size_in_4b),
    )

  @parameterized.named_parameters(
      dict(testcase_name="without_minibatching", enable_minibatching=False),
      dict(testcase_name="with_minibatching", enable_minibatching=True),
  )
  def test_forward_pass(self, enable_minibatching: bool):
    row_pointers, embedding_ids, sample_ids, gains = self._preprocess_input(
        enable_minibatching
    )
    expected_max_ids, expected_max_unique_ids = self._expected_fdo_statistics()

    activations, max_ids, max_unique_ids = self.primitive.bind(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
        device_batch_size=_BATCH_SIZE,
        max_ids_per_partition=_MAX_IDS_PER_PARTITION,
        max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
        sharding_strategy=1,
        enable_minibatching=enable_minibatching,
        num_sc_per_device=self.num_sc_per_device,
        hbm_word_size_in_4b=self.hbm_word_size_in_4b,
    )

    np.testing.assert_equal(activations, self.expected_activations)
    self.assertEqual(max_ids.dtype, jnp.int32)
    self.assertEqual(max_unique_ids.dtype, jnp.int32)
    np.testing.assert_array_equal(max_ids, expected_max_ids)
    np.testing.assert_array_equal(max_unique_ids, expected_max_unique_ids)
    # The statistics differ, so the test can tell them apart.
    self.assertFalse(np.array_equal(max_ids, max_unique_ids))

  @parameterized.named_parameters(
      dict(testcase_name="without_minibatching", enable_minibatching=False),
      dict(testcase_name="with_minibatching", enable_minibatching=True),
  )
  def test_activations_match_primitive_without_fdo_statistics(
      self, enable_minibatching: bool
  ):
    row_pointers, embedding_ids, sample_ids, gains = self._preprocess_input(
        enable_minibatching
    )
    operands = (
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        1,  # num_minibatches_per_physical_sparse_core
        self.emb_table_sharded[0],
    )
    params = dict(
        device_batch_size=_BATCH_SIZE,
        max_ids_per_partition=_MAX_IDS_PER_PARTITION,
        max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
        sharding_strategy=1,
        enable_minibatching=enable_minibatching,
    )
    expected_activations = (
        sparse_dense_matmul_csr.tpu_sparse_dense_matmul_csr_primitive.bind(
            *operands, **params
        )
    )

    activations, _, _ = self.primitive.bind(
        *operands,
        **params,
        num_sc_per_device=self.num_sc_per_device,
        hbm_word_size_in_4b=self.hbm_word_size_in_4b,
    )

    np.testing.assert_array_equal(activations, expected_activations)

  def _abstract_operands(self) -> tuple[jax.ShapeDtypeStruct | np.int32, ...]:
    ids = jax.ShapeDtypeStruct((16,), jnp.int32)
    gains = jax.ShapeDtypeStruct((16,), jnp.float32)
    table = jax.ShapeDtypeStruct((_VOCAB_SIZE, _EMB_SIZE), jnp.float32)
    return (ids, ids, ids, gains, np.int32(1), table)

  def test_output_shapes(self):
    activations, max_ids, max_unique_ids = jax.eval_shape(
        functools.partial(
            self.primitive.bind,
            device_batch_size=_BATCH_SIZE,
            max_ids_per_partition=_MAX_IDS_PER_PARTITION,
            max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
            num_sc_per_device=4,
            hbm_word_size_in_4b=16,
        ),
        *self._abstract_operands(),
    )

    self.assertEqual(activations.shape, (_BATCH_SIZE, _EMB_SIZE))
    self.assertEqual(activations.dtype, jnp.float32)
    for fdo_statistic in (max_ids, max_unique_ids):
      self.assertEqual(fdo_statistic.shape, (4 * 16,))
      self.assertEqual(fdo_statistic.dtype, jnp.int32)

  @parameterized.named_parameters(
      dict(
          testcase_name="num_sc_per_device",
          num_sc_per_device=0,
          hbm_word_size_in_4b=16,
          expected_regex="`num_sc_per_device` must be positive",
      ),
      dict(
          testcase_name="hbm_word_size_in_4b",
          num_sc_per_device=4,
          hbm_word_size_in_4b=-1,
          expected_regex="`hbm_word_size_in_4b` must be positive",
      ),
  )
  def test_invalid_fdo_statistics_size(
      self,
      num_sc_per_device: int,
      hbm_word_size_in_4b: int,
      expected_regex: str,
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      jax.eval_shape(
          functools.partial(
              self.primitive.bind,
              device_batch_size=_BATCH_SIZE,
              max_ids_per_partition=_MAX_IDS_PER_PARTITION,
              max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
              num_sc_per_device=num_sc_per_device,
              hbm_word_size_in_4b=hbm_word_size_in_4b,
          ),
          *self._abstract_operands(),
      )

  def test_invalid_max_ids_per_partition(self):
    # The parameters of the sparse dense matmul primitive are validated too.
    with self.assertRaises(ValueError):
      jax.eval_shape(
          functools.partial(
              self.primitive.bind,
              device_batch_size=_BATCH_SIZE,
              max_ids_per_partition=0,
              max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
              num_sc_per_device=4,
              hbm_word_size_in_4b=16,
          ),
          *self._abstract_operands(),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="without_minibatching",
          enable_minibatching=False,
          expected_call_target="SparseDenseMatmulOp",
      ),
      dict(
          testcase_name="with_minibatching",
          enable_minibatching=True,
          expected_call_target="SparseDenseMatmulWithMinibatchingOp",
      ),
  )
  def test_lowering_has_fdo_statistics_results(
      self, enable_minibatching: bool, expected_call_target: str
  ):
    lowered = jax.jit(
        functools.partial(
            self.primitive.bind,
            device_batch_size=_BATCH_SIZE,
            max_ids_per_partition=_MAX_IDS_PER_PARTITION,
            max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
            enable_minibatching=enable_minibatching,
            num_sc_per_device=4,
            hbm_word_size_in_4b=16,
        )
    ).lower(*self._abstract_operands())

    module = lowered.compiler_ir()
    assert module is not None
    with module.context:
      custom_calls = []
      for func in module.body.operations:
        for op in func.regions[0].blocks[0].operations:
          if op.operation.name == "stablehlo.custom_call":
            custom_calls.append(op)
      self.assertLen(custom_calls, 1)
      self.assertIn(
          expected_call_target,
          str(custom_calls[0].attributes["call_target_name"]),
      )
      result_types = [str(result.type) for result in custom_calls[0].results]
    self.assertEqual(
        result_types,
        ["tensor<16x8xf32>", "tensor<64xi32>", "tensor<64xi32>"],
    )


if __name__ == "__main__":
  absltest.main()
