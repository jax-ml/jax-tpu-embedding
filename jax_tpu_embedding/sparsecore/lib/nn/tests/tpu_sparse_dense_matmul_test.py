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
import collections
from collections.abc import Sequence
import functools
from typing import override

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

VariableInfo = collections.namedtuple("VariableInfo", ["shape", "offset"])


def _create_embedding_variable_for_pmap(
    var_shapes: list[VariableInfo],
    devices: list[jax.Device],
    mesh: jax.sharding.Mesh,
):
  num_sc_per_device = utils.num_sparsecores_per_device(mesh.devices.item(0))
  dim = var_shapes[0].shape[1]
  assert all(v.shape[1] == dim for v in var_shapes)
  total_vocab = sum(v.shape[0] for v in var_shapes)
  emb_tables = [
      test_utils.row_id_initializer(v.shape, offset=v.offset)
      for v in var_shapes
  ]
  emb_table_sharded = test_utils.create_per_device_sharded_stacked_tables(
      emb_tables,
      num_devices=len(devices),
      num_sparsecore_per_device=num_sc_per_device,
      rotation=num_sc_per_device,
  )
  embedding_variable_shards = [
      jax.device_put(
          emb_table_sharded[i : i + 1],
          device=local_device,
      )
      for i, local_device in enumerate(devices)
  ]
  sharding = NamedSharding(mesh, P("x", None, None))
  return jax.make_array_from_single_device_arrays(
      shape=(len(devices), total_vocab // len(devices), dim),
      sharding=sharding,
      arrays=embedding_variable_shards,
  )


def _create_embedding_variable_for_jit(
    var_shapes: list[VariableInfo],
    devices: list[jax.Device],
    mesh: jax.sharding.Mesh,
    emb_tables: Sequence[jax.Array] | None = None,
):
  """Creates a sharded embedding variable.

  Args:
    var_shapes: Shape and row-id offset of each table in the stack.
    devices: Devices to shard the variable across.
    mesh: Mesh containing `devices`.
    emb_tables: Optional table values, one per entry in `var_shapes`. If None,
      each table is filled by `test_utils.row_id_initializer`.

  Returns:
    The sharded embedding variable.
  """
  num_sc_per_device = utils.num_sparsecores_per_device(mesh.devices.item(0))
  dim = var_shapes[0].shape[1]
  assert all(v.shape[1] == dim for v in var_shapes)
  total_vocab = sum(v.shape[0] for v in var_shapes)
  if emb_tables is None:
    emb_tables = [
        test_utils.row_id_initializer(v.shape, offset=v.offset)
        for v in var_shapes
    ]
  emb_table_sharded = test_utils.create_per_device_sharded_stacked_tables(
      emb_tables,
      num_devices=len(devices),
      num_sparsecore_per_device=num_sc_per_device,
      rotation=num_sc_per_device,
  )
  embedding_variable_shards = [
      jax.device_put(
          emb_table_sharded[i],
          device=local_device,
      )
      for i, local_device in enumerate(devices)
  ]
  sharding = NamedSharding(mesh, P("x", None))
  return jax.make_array_from_single_device_arrays(
      shape=(total_vocab, dim),
      sharding=sharding,
      arrays=embedding_variable_shards,
  )


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


def _absmax_test_table(vocab_size: int, dim: int) -> np.ndarray:
  """Returns a [vocab_size, dim] table whose rows absmax quantization changes.

  Row r is `(r + 1) * [1, 2, ..., 7, 1, 2, ...] / 7`, so its absmax is `r + 1`
  (for dim >= 7) and each element scales to `bound * k / 7` before rounding.
  For 64 or 256 buckets (bound 31 or 127), none of those values is within 0.07
  of a rounding boundary, so the device and NumPy round them the same way.

  Args:
    vocab_size: Number of rows.
    dim: Number of columns. Must be at least 7.
  """
  if dim < 7:
    raise ValueError(f"dim must be >= 7, got {dim}.")
  rows = np.arange(1, vocab_size + 1, dtype=np.float32)[:, None]
  cols = (np.arange(dim) % 7 + 1).astype(np.float32)[None, :]
  return rows * cols / np.float32(7.0)


class ErrorHandlingTest(absltest.TestCase):

  # Tests that even if static buffer size is too small, the matmul can proceed.
  def test_static_buffer_size_was_too_small(self):
    long_feature = np.arange(800, dtype=np.int32).reshape(8, -1)

    long_spec = embedding_spec.TableSpec(
        vocabulary_size=1000,
        embedding_dim=8,
        initializer=lambda *_: np.zeros((1000, 8), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="table",
        max_ids_per_partition=64,
        max_unique_ids_per_partition=64,
        suggested_coo_buffer_size_per_device=64,
        quantization_config=None,
    )
    lf_spec = embedding_spec.FeatureSpec(
        table_spec=long_spec,
        input_shape=[8, 100],
        output_shape=[8, 8],
        name="feature",
    )
    global_devices = jax.devices()
    first_device = global_devices[0]
    num_sc_per_device = utils.num_sparsecores_per_device(first_device)
    mesh = jax.sharding.Mesh([first_device], "x")
    feature_specs = {
        "feature": lf_spec,
    }
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=1,
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature": long_feature,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        batch_number=batch_number,
    )
    np.testing.assert_array_less(
        64, stats.required_buffer_size_per_sc["table"]
    )  # required buffer is bigger than actually suggested
    self.assertNotEmpty(preprocessed_inputs.lhs_row_pointers)
    self.assertNotEmpty(preprocessed_inputs.lhs_embedding_ids)
    self.assertNotEmpty(preprocessed_inputs.lhs_sample_ids)
    self.assertNotEmpty(preprocessed_inputs.lhs_gains)

    # Initialize a table of all 1s.
    emb_table_a = (
        np.array([[1 for _ in range(8)] for _ in range(1000)])
        .reshape(1000, 8)
        .astype(np.float32)
    )
    emb_table_a_sharded = utils.shard_emb_table(
        emb_table_a,
        num_devices=1,
        num_sc_per_device=4,
    )
    table_device_arrays = [
        jax.device_put(
            emb_table_a_sharded[0],
            device=first_device,
        ),
    ]
    sharding = NamedSharding(mesh, P(None, "x", None))
    embedding_variables = {
        "table": embedding.EmbeddingVariables(
            table=jax.make_array_from_single_device_arrays(
                shape=(1000, 8),
                sharding=sharding,
                arrays=table_device_arrays,
            ),
            slot=(),
        )
    }
    tpu_sparse_dense_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        global_device_count=1,
        feature_specs=tuple(jax.tree.leaves(feature_specs)),
        sharding_strategy="MOD",
    )
    sparse_matmul = jax.jit(tpu_sparse_dense_matmul)
    activations = sparse_matmul(
        preprocessed_inputs,
        embedding_variables,
    )
    # The static buffer size is 64.
    # This means we will get only 64 activations back.
    row_sum = 0
    for act in jax.tree.leaves(activations):
      # Each row should have homogenous elements so add up the first element
      # of each row.
      for i in range(act.shape[0]):
        row_sum += act[i][0]
    self.assertEqual(row_sum, 64)


class TpuSparseDenseMatmulTest(parameterized.TestCase, absltest.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=6,
        initializer=lambda *_: jnp.zeros((32, 8), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.table_spec_aa = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=6,
        initializer=lambda *_: jnp.zeros((32, 8), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_aa",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=16,
        initializer=lambda *_: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.feature_spec_a = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=(16, 1),
        output_shape=(
            16,
            self.table_spec_a.embedding_dim,
        ),
        name="feature_spec_a",
    )
    self.feature_spec_aa = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_aa,
        input_shape=(16, 1),
        output_shape=(
            16,
            self.table_spec_a.embedding_dim,
        ),
        name="feature_spec_aa",
    )
    self.feature_spec_b = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_b,
        input_shape=(16, 1),
        output_shape=(
            16,
            self.table_spec_b.embedding_dim,
        ),
        name="feature_spec_b",
    )
    self.input_tensor = np.array(
        [
            np.array([5, 4, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
            np.array([9], dtype=np.int32),
            np.array([1, 9, 16], dtype=np.int32),
            np.array([6, 12, 1, 10], dtype=np.int32),
            np.array([12, 19], dtype=np.int32),
            np.array([0, 15, 2, 25, 25], dtype=np.int32),
            np.array([4, 28, 25], dtype=np.int32),
            np.array([15, 0], dtype=np.int32),
            np.array([13], dtype=np.int32),
            np.array([11], dtype=np.int32),
            np.array([7, 1], dtype=np.int32),
            np.array([8, 9], dtype=np.int32),
            np.array([14, 14, 14], dtype=np.int32),
            np.array([2, 28], dtype=np.int32),
            np.array([10, 16], dtype=np.int32),
        ],
        dtype=object,
    )
    self.input_tensor_table_b = np.array(
        [
            np.array([50, 40, 20], dtype=np.int32),
            np.array([33], dtype=np.int32),
            np.array([59], dtype=np.int32),
            np.array([1, 9, 16], dtype=np.int32),
            np.array([6, 12, 1, 10], dtype=np.int32),
            np.array([12, 19], dtype=np.int32),
            np.array([0, 15, 2, 25, 25], dtype=np.int32),
            np.array([4, 28, 25], dtype=np.int32),
            np.array([50, 51, 0], dtype=np.int32),
            np.array([10, 50, 60], dtype=np.int32),
            np.array([11, 12, 13], dtype=np.int32),
            np.array([61, 60], dtype=np.int32),
            np.array([58, 2], dtype=np.int32),
            np.array([41, 41, 50], dtype=np.int32),
            np.array([2, 58], dtype=np.int32),
            np.array([10, 16], dtype=np.int32),
        ],
        dtype=object,
    )

  @parameterized.product(
      using_pmap=[False, True],
      use_activation_unstack_primitive=[False, True],
  )
  def test_sparse_dense_matmul_two_chips_sharded(
      self, using_pmap, use_activation_unstack_primitive
  ):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_aa": self.feature_spec_aa,
    }
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
            "feature_spec_aa": self.input_tensor,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=2,
        global_device_count=2,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=using_pmap,
        batch_number=batch_number,
    )
    embedding_variables = {}
    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        global_device_count=mesh.size,
        use_activation_unstack_primitive=use_activation_unstack_primitive,
    )
    if using_pmap:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          _create_embedding_variable_for_pmap(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          (),
      )
      embedding_variables["table_aa"] = embedding.EmbeddingVariables(
          _create_embedding_variable_for_pmap(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          (),
      )
      activations = jax.pmap(
          tpu_sparse_dense_matmul_fn,
          static_broadcasted_argnums=[2],
      )(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    else:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          slot=(),
      )
      embedding_variables["table_aa"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          slot=(),
      )
      sharded_matmul = functools.partial(
          tpu_sparse_dense_matmul_fn,
          feature_specs=tuple(jax.tree.leaves(feature_specs)),
      )

      sharded_matmul = jax.shard_map(
          sharded_matmul,
          mesh=mesh,
          in_specs=(
              P(mesh.axis_names[0]),
              P(mesh.axis_names[0], None),
          ),
          out_specs=P(mesh.axis_names[0]),
          check_vma=False,
      )
      sharded_matmul = jax.jit(sharded_matmul)
      activations = sharded_matmul(
          preprocessed_inputs,
          embedding_variables,
      )
    expected_emb_activations = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
    )
    if using_pmap:
      expected_emb_activations = expected_emb_activations.reshape(
          len(devices), 16 // len(devices), 6
      )
    np.testing.assert_equal(activations[0], expected_emb_activations)
    np.testing.assert_equal(activations[1], expected_emb_activations)

  @parameterized.product(
      using_pmap=[False, True],
      use_activation_unstack_primitive=[False, True],
  )
  def test_sparse_dense_matmul_two_chips_sharded_stacked(
      self, using_pmap, use_activation_unstack_primitive
  ):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_aa": self.feature_spec_aa,
    }
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
            "feature_spec_aa": self.input_tensor,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=2,
        global_device_count=2,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=using_pmap,
        batch_number=batch_number,
    )
    embedding_variables = {}
    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        global_device_count=mesh.size,
        use_activation_unstack_primitive=use_activation_unstack_primitive,
    )
    if using_pmap:
      embedding_variables["table_a_table_aa"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_pmap(
              [
                  VariableInfo(shape=(64, 8), offset=0),
                  VariableInfo(shape=(64, 8), offset=100),
              ],
              devices,
              mesh,
          ),
          slot=(),
      )
      activations = jax.pmap(
          tpu_sparse_dense_matmul_fn,
          static_broadcasted_argnums=[2],
      )(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    else:
      embedding_variables["table_a_table_aa"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [
                  VariableInfo(shape=(64, 8), offset=0),
                  VariableInfo(shape=(64, 8), offset=100),
              ],
              devices,
              mesh,
          ),
          slot=(),
      )
      sharded_matmul = functools.partial(
          tpu_sparse_dense_matmul_fn,
          feature_specs=tuple(jax.tree.leaves(feature_specs)),
      )

      sharded_matmul = jax.shard_map(
          sharded_matmul,
          mesh=mesh,
          in_specs=(
              P(mesh.axis_names[0]),
              P(mesh.axis_names[0], None),
          ),
          out_specs=P(mesh.axis_names[0]),
          check_vma=False,
      )
      sharded_matmul = jax.jit(sharded_matmul)
      activations = sharded_matmul(
          preprocessed_inputs,
          embedding_variables,
      )
    expected_emb_activations_a = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
    )
    expected_emb_activations_aa = np.array(
        [
            [311.0, 311.0, 311.0, 311.0, 311.0, 311.0],
            [103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            [109.0, 109.0, 109.0, 109.0, 109.0, 109.0],
            [326.0, 326.0, 326.0, 326.0, 326.0, 326.0],
            [429.0, 429.0, 429.0, 429.0, 429.0, 429.0],
            [231.0, 231.0, 231.0, 231.0, 231.0, 231.0],
            [567.0, 567.0, 567.0, 567.0, 567.0, 567.0],
            [357.0, 357.0, 357.0, 357.0, 357.0, 357.0],
            [215.0, 215.0, 215.0, 215.0, 215.0, 215.0],
            [113.0, 113.0, 113.0, 113.0, 113.0, 113.0],
            [111.0, 111.0, 111.0, 111.0, 111.0, 111.0],
            [208.0, 208.0, 208.0, 208.0, 208.0, 208.0],
            [217.0, 217.0, 217.0, 217.0, 217.0, 217.0],
            [342.0, 342.0, 342.0, 342.0, 342.0, 342.0],
            [230.0, 230.0, 230.0, 230.0, 230.0, 230.0],
            [226.0, 226.0, 226.0, 226.0, 226.0, 226.0],
        ],
        dtype=np.float32,
    )
    if using_pmap:
      expected_emb_activations_a = expected_emb_activations_a.reshape(
          len(devices), 16 // len(devices), 6
      )
      expected_emb_activations_aa = expected_emb_activations_aa.reshape(
          len(devices), 16 // len(devices), 6
      )
    self.assertLen(activations, 2)
    np.testing.assert_equal(activations[0], expected_emb_activations_a)
    np.testing.assert_equal(activations[1], expected_emb_activations_aa)

  @parameterized.product(
      using_pmap=[False, True],
      use_activation_unstack_primitive=[False, True],
  )
  def test_sparse_dense_matmul_single_chip(
      self, using_pmap, use_activation_unstack_primitive
  ):
    global_devices = jax.devices()
    devices = [global_devices[0]]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_b": self.feature_spec_b,
    }
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=1,
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
            "feature_spec_b": self.input_tensor_table_b,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        has_leading_dimension=using_pmap,
        batch_number=batch_number,
    )
    embedding_variables = {}

    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        global_device_count=mesh.size,
        use_activation_unstack_primitive=use_activation_unstack_primitive,
    )
    if using_pmap:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_pmap(
              [VariableInfo(shape=(32, 8), offset=0)], devices, mesh
          ),
          slot=(),
      )
      embedding_variables["table_b"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_pmap(
              [VariableInfo((64, 16), 0)], devices, mesh
          ),
          slot=(),
      )
      activations = jax.pmap(
          tpu_sparse_dense_matmul_fn,
          static_broadcasted_argnums=[2],
      )(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    else:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          slot=(),
      )
      embedding_variables["table_b"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((64, 16), 0)], devices, mesh
          ),
          slot=(),
      )
      sparse_matmul = jax.jit(tpu_sparse_dense_matmul_fn, static_argnums=[2])
      activations = sparse_matmul(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    expected_emb_activations = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
    )
    if using_pmap:
      expected_emb_activations = expected_emb_activations.reshape(1, 16, 6)
    np.testing.assert_equal(activations[0], expected_emb_activations)

  @parameterized.product(
      using_pmap=[False, True],
  )
  def test_sparse_dense_matmul_two_tables(self, using_pmap):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_b": self.feature_spec_b,
    }
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    # Add another table.
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
            "feature_spec_b": self.input_tensor_table_b,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=2,
        global_device_count=2,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=using_pmap,
        batch_number=batch_number,
    )
    embedding_variables = {}
    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        global_device_count=mesh.size,
    )
    if using_pmap:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_pmap(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          slot=(),
      )
      embedding_variables["table_b"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_pmap(
              [VariableInfo((64, 16), 0)], devices, mesh
          ),
          slot=(),
      )
      activations = jax.pmap(
          tpu_sparse_dense_matmul_fn,
          static_broadcasted_argnums=(2),
      )(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    else:
      embedding_variables["table_a"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((32, 8), 0)], devices, mesh
          ),
          slot=(),
      )
      embedding_variables["table_b"] = embedding.EmbeddingVariables(
          table=_create_embedding_variable_for_jit(
              [VariableInfo((64, 16), 0)], devices, mesh
          ),
          slot=(),
      )
      sharded_matmul = functools.partial(
          tpu_sparse_dense_matmul_fn,
          feature_specs=tuple(jax.tree.leaves(feature_specs)),
      )
      sharded_matmul = jax.shard_map(
          sharded_matmul,
          mesh=mesh,
          in_specs=(
              P(mesh.axis_names[0]),
              P(mesh.axis_names[0], None),
          ),
          out_specs=P(mesh.axis_names[0]),
          check_vma=False,
      )
      sharded_matmul = jax.jit(sharded_matmul)
      activations = sharded_matmul(
          preprocessed_inputs,
          embedding_variables,
      )
    expected_emb_activations = {}
    expected_emb_activations["table_a"] = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
    )
    expected_emb_activations["table_b"] = np.array(
        [
            [
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
                110.0,
            ],
            [
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
                33.0,
            ],
            [
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
                59.0,
            ],
            [
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
            ],
            [
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
                29.0,
            ],
            [
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
                31.0,
            ],
            [
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
                67.0,
            ],
            [
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
                57.0,
            ],
            [
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
                101.0,
            ],
            [
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
                120.0,
            ],
            [
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
                36.0,
            ],
            [
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
                121.0,
            ],
            [
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
                132.0,
            ],
            [
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
                60.0,
            ],
            [
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
                26.0,
            ],
        ],
        dtype=np.float32,
    )
    if using_pmap:
      expected_emb_activations["table_a"] = expected_emb_activations[
          "table_a"
      ].reshape(2, 8, 6)
      expected_emb_activations["table_b"] = expected_emb_activations[
          "table_b"
      ].reshape(2, 8, 16)
    np.testing.assert_equal(
        activations,
        (
            expected_emb_activations["table_a"],
            expected_emb_activations["table_b"],
        ),
    )

  @parameterized.product(
      using_pmap=[False, True],
  )
  def test_sparse_dense_matmul_four_chips_complex_stacked(self, using_pmap):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    mesh = jax.sharding.Mesh(devices, "x")
    country_table = embedding_spec.TableSpec(
        vocabulary_size=247,
        embedding_dim=16,
        initializer=lambda *_: jnp.zeros((256, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="country_table",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    language_table = embedding_spec.TableSpec(
        vocabulary_size=316,
        embedding_dim=16,
        initializer=lambda: jnp.zeros((384, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="language_table",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    related_item_table = embedding_spec.TableSpec(
        vocabulary_size=151,
        embedding_dim=16,
        initializer=lambda: jnp.zeros((256, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="related_item_table",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    country_feature = embedding_spec.FeatureSpec(
        table_spec=country_table,
        input_shape=(16, 1),
        output_shape=(
            16,
            country_table.embedding_dim,
        ),
        name="country_feature",
    )
    language_feature = embedding_spec.FeatureSpec(
        table_spec=language_table,
        input_shape=(16, 1),
        output_shape=(
            16,
            language_table.embedding_dim,
        ),
        name="language_feature",
    )
    related_item_feature = embedding_spec.FeatureSpec(
        table_spec=related_item_table,
        input_shape=(16, 1),
        output_shape=(
            16,
            related_item_table.embedding_dim,
        ),
        name="related_item_feature",
    )
    feature_specs = {
        "country": country_feature,  # row_offset=0
        "language": language_feature,  # row_offset=16
        "related_item": related_item_feature,  # row_offset=32
    }
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    input_tensor = np.array(
        [
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([86], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32),
        ],
        dtype=object,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "country": input_tensor,
            "language": input_tensor,
            "related_item": input_tensor,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=mesh.local_mesh.size,
        global_device_count=mesh.size,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        has_leading_dimension=using_pmap,
        batch_number=batch_number,
    )
    embedding_variables = {}
    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        global_device_count=mesh.size,
    )
    if using_pmap:
      embedding_variables["country_table_language_table_related_item_table"] = (
          embedding.EmbeddingVariables(
              table=_create_embedding_variable_for_pmap(
                  [
                      VariableInfo(shape=(256, 16), offset=0),  # country
                      VariableInfo(shape=(384, 16), offset=500),  # language
                      VariableInfo(
                          shape=(256, 16), offset=1000
                      ),  # related_item
                  ],
                  devices,
                  mesh,
              ),
              slot=(),
          )
      )
      activations = jax.pmap(
          tpu_sparse_dense_matmul_fn,
          static_broadcasted_argnums=(2,),
      )(
          preprocessed_inputs,
          embedding_variables,
          tuple(jax.tree.leaves(feature_specs)),
      )
    else:
      embedding_variables["country_table_language_table_related_item_table"] = (
          embedding.EmbeddingVariables(
              table=_create_embedding_variable_for_jit(
                  [
                      VariableInfo(shape=(256, 16), offset=0),  # country
                      VariableInfo(shape=(384, 16), offset=500),  # language
                      VariableInfo(
                          shape=(256, 16), offset=1000
                      ),  # related_item
                  ],
                  devices,
                  mesh,
              ),
              slot=(),
          )
      )
      sharded_matmul = functools.partial(
          tpu_sparse_dense_matmul_fn,
          feature_specs=tuple(jax.tree.leaves(feature_specs)),
      )

      sharded_matmul = jax.shard_map(
          sharded_matmul,
          mesh=mesh,
          in_specs=(
              P(mesh.axis_names[0]),
              P(mesh.axis_names[0], None),
          ),
          out_specs=P(mesh.axis_names[0]),
          check_vma=False,
      )
      sharded_matmul = jax.jit(sharded_matmul)
      activations = sharded_matmul(
          preprocessed_inputs,
          embedding_variables,
      )
    expected_act_country = np.ones((4, 4, 16), np.float32)
    expected_act_country[0][3, :] = 86.0
    if not using_pmap:
      expected_act_country = expected_act_country.reshape(16, 16)
    np.testing.assert_equal(
        activations[0],  # country
        expected_act_country,
        "country activations do not match",
    )
    expected_act_language = np.ones((4, 4, 16), np.float32) * 501
    expected_act_language[0][3, :] = 586.0
    if not using_pmap:
      expected_act_language = expected_act_language.reshape(16, 16)
    np.testing.assert_equal(
        activations[1],  # language
        expected_act_language,
        "language activations do not match",
    )
    expected_act_related_item = np.ones((4, 4, 16), np.float32) * 1001
    expected_act_related_item[0][3, :] = 1086.0
    if not using_pmap:
      expected_act_related_item = expected_act_related_item.reshape(16, 16)
    np.testing.assert_equal(
        activations[2],  # related_item
        expected_act_related_item,
        "related_item activations do not match",
    )

  def test_sparse_dense_matmul_quantized(self):
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, "x")
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])

    quantized_table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=32,
        initializer=lambda *_: jnp.zeros((32, 32), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="quantized_table",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        quantization_config=embedding_spec.FixedQuantizationConfig(
            min_value=0.0, max_value=15.0, num_buckets=256
        ),
    )

    quantized_feature_spec = embedding_spec.FeatureSpec(
        table_spec=quantized_table_spec,
        input_shape=(16, 1),
        output_shape=(16, quantized_table_spec.embedding_dim),
        name="quantized_feature",
    )

    feature_specs = {"quantized_feature": quantized_feature_spec}
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {"quantized_feature": self.input_tensor},
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        batch_number=batch_number,
    )

    embedding_variables = {}
    embedding_variables["quantized_table"] = embedding.EmbeddingVariables(
        table=_create_embedding_variable_for_jit(
            [VariableInfo((32, 32), 0)], devices, mesh
        ),
        slot=(),
    )

    tpu_sparse_dense_matmul_fn = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        sharding_strategy="MOD",
        feature_specs=feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    sparse_matmul = jax.jit(tpu_sparse_dense_matmul_fn)

    lowered = sparse_matmul.lower(preprocessed_inputs, embedding_variables)
    lowered_text = lowered.as_text()
    self.assertIn("quantization_config", lowered_text)
    self.assertIn("fixed", lowered_text)

    # Grab compiled HLO to verify u8 types are used for real quantization.
    compiled_hlo = lowered.compile().as_text()
    self.assertIsNotNone(compiled_hlo)
    self.assertIn("u8", compiled_hlo)

    activations = sparse_matmul(preprocessed_inputs, embedding_variables)

    q_min, q_max = 0.0, 15.0
    expected_activations_flat = [
        np.sum(np.clip(row, q_min, q_max)) for row in self.input_tensor
    ]
    expected_emb_activations = np.array(
        [[val] * 32 for val in expected_activations_flat], dtype=np.float32
    )

    self.assertLen(activations, 1)
    # The quantization and dequantization can introduce small errors.
    np.testing.assert_allclose(
        activations["quantized_feature"], expected_emb_activations, atol=0.5
    )

  def _jit_single_table_matmul(
      self,
      table: np.ndarray,
      quantization_config: (
          embedding_spec.FixedQuantizationConfig
          | embedding_spec.AbsmaxQuantizationConfig
          | None
      ),
      table_name: str,
      feature_name: str,
  ):
    """Builds a jitted sum-combiner matmul of `self.input_tensor` on `table`.

    Args:
      table: The [vocab, dim] embedding table values.
      quantization_config: The table's quantization config.
      table_name: Name of the table.
      feature_name: Name of the feature (and of the output activation).

    Returns:
      A tuple `(jitted_matmul, preprocessed_inputs, embedding_variables)`.
    """
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, "x")
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    vocab_size, dim = table.shape

    table_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size,
        embedding_dim=dim,
        initializer=lambda *_: jnp.zeros((vocab_size, dim), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name=table_name,
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        quantization_config=quantization_config,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(16, 1),
        output_shape=(16, dim),
        name=feature_name,
    )
    feature_specs = {feature_name: feature_spec}
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=len(devices),
        num_sc_per_device=num_sc_per_device,
    )
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {feature_name: self.input_tensor},
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        batch_number=42,
    )
    embedding_variables = {
        table_name: embedding.EmbeddingVariables(
            table=_create_embedding_variable_for_jit(
                [VariableInfo((vocab_size, dim), 0)],
                devices,
                mesh,
                emb_tables=[jnp.asarray(table)],
            ),
            slot=(),
        )
    }
    sparse_matmul = jax.jit(
        functools.partial(
            embedding.tpu_sparse_dense_matmul,
            sharding_strategy="MOD",
            feature_specs=feature_specs,
            global_device_count=len(devices),
            num_sc_per_device=num_sc_per_device,
        )
    )
    return sparse_matmul, preprocessed_inputs, embedding_variables

  def _sum_combine(self, table: np.ndarray) -> np.ndarray:
    """Returns the sum-combiner activations of `self.input_tensor` on `table`."""
    return np.stack(
        [table[sample].sum(axis=0) for sample in self.input_tensor]
    ).astype(np.float32)

  def test_sparse_dense_matmul_dynamic_quantized(self):
    table = _absmax_test_table(32, 32)
    feature_name = "quantized_dynamic_feature"
    sparse_matmul, preprocessed_inputs, embedding_variables = (
        self._jit_single_table_matmul(
            table,
            embedding_spec.AbsmaxQuantizationConfig(num_buckets=256),
            table_name="quantized_dynamic_table",
            feature_name=feature_name,
        )
    )

    lowered = sparse_matmul.lower(preprocessed_inputs, embedding_variables)
    lowered_hlo = lowered.as_text()
    self.assertIn("quantization_config", lowered_hlo)
    self.assertIn("absmax", lowered_hlo)

    activations = sparse_matmul(preprocessed_inputs, embedding_variables)
    self.assertLen(activations, 1)
    self.assertEqual(activations[feature_name].shape, (16, 32))
    self.assertEqual(activations[feature_name].dtype, jnp.float32)

    expected_activations = self._sum_combine(_absmax_fake_quant(table))
    # Fails if the test table stops being changed by absmax quantization.
    self.assertFalse(
        np.allclose(expected_activations, self._sum_combine(table), atol=1e-3)
    )
    np.testing.assert_allclose(
        activations[feature_name], expected_activations, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
  absltest.main()
