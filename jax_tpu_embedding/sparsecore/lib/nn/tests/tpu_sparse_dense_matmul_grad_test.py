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
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

np.set_printoptions(threshold=np.inf, suppress=True)

_VOC_A = 32
_DIM_A = 4
_VOC_B = 64
_DIM_B = 6
_VOC_C = 32
_DIM_C = 18
_BATCH_SIZE = 16


def count_num(arr: embedding.Nested[jnp.ndarray], num: int) -> int:
  """Helper method that counts the number of occurrences of `num` in `arr`."""
  count = 0
  for array in arr:
    for n in array:
      if n == num:
        count += 1
  return count


class LinearLearningRateSchedule:
  """Simple linear learning rate schedule for tests."""

  def __init__(self, initial_learning_rate: float):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step: int | jax.Array):
    return self.initial_learning_rate / (step + 1)


class TpuSparseDenseMatmulGradTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=_VOC_A,
        embedding_dim=_DIM_A,
        initializer=lambda: jnp.zeros((_VOC_A, _DIM_A), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
        combiner="sum",
        name="table_a",
    )
    self.table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=_VOC_B,
        embedding_dim=_DIM_B,
        initializer=lambda: jnp.zeros((_VOC_B, _DIM_B), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
        combiner="sum",
        name="table_b",
    )
    self.table_spec_c = embedding_spec.TableSpec(
        vocabulary_size=_VOC_C,
        embedding_dim=_DIM_C,
        initializer=lambda: jnp.zeros((_VOC_C, _DIM_C), dtype=jnp.float32),
        optimizer=embedding_spec.AdagradOptimizerSpec(learning_rate=0.01),
        combiner="sum",
        name="table_c",
    )
    self.feature_spec_a = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_a.embedding_dim,
        ),
        name="feature_spec_a",
    )
    self.feature_spec_b = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_b,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_b.embedding_dim,
        ),
        name="feature_spec_b",
    )
    self.feature_spec_c = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_c,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_c.embedding_dim,
        ),
        name="feature_spec_c",
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
    self.input_tensor_table_c = np.array(
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

  def test_sparse_dense_matmul_one_chip_unsharded(self):
    devices = jax.devices()[:1]
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_b": self.feature_spec_b,
        "feature_spec_c": self.feature_spec_c,
    }
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        num_sc_per_device=4,
        global_device_count=len(devices),
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
            "feature_spec_b": self.input_tensor_table_b,
            "feature_spec_c": self.input_tensor_table_c,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        batch_number=batch_number,
    )

    table_dim_a = table_stacking._next_largest_multiple(_DIM_A, 8)
    emb_table_a = (
        np.array([[i for _ in range(table_dim_a)] for i in range(_VOC_A)])
        .reshape(_VOC_A, table_dim_a)
        .astype(np.float32)
    )
    emb_table_a_sharded = einops.rearrange(
        emb_table_a,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )
    embedding_variables = {}
    embedding_variables["table_a"] = [
        jax.device_put(
            emb_table_a_sharded[0],
            device=devices[0],
        ),
    ]

    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_a"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_A, table_dim_a),
            sharding=sharding,
            arrays=embedding_variables["table_a"],
        )
    ])

    table_dim_b = table_stacking._next_largest_multiple(_DIM_B, 8)
    emb_table_b = (
        np.array([[i for _ in range(table_dim_b)] for i in range(_VOC_B)])
        .reshape(_VOC_B, table_dim_b)
        .astype(np.float32)
    )
    emb_table_b_sharded = einops.rearrange(
        emb_table_b,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )
    embedding_variables["table_b"] = [
        jax.device_put(
            emb_table_b_sharded[0],
            device=devices[0],
        ),
    ]
    embedding_variables["table_b"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_B, table_dim_b),
            sharding=sharding,
            arrays=embedding_variables["table_b"],
        )
    ])
    table_dim_c = table_stacking._next_largest_multiple(_DIM_C, 8)
    emb_table_c = (
        np.array([[i for _ in range(table_dim_c)] for i in range(_VOC_C)])
        .reshape(_VOC_C, table_dim_c)
        .astype(np.float32)
    )
    emb_table_c_sharded = einops.rearrange(
        emb_table_c,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )
    accumulator_init = jnp.zeros(emb_table_c_sharded[0].shape, np.float32)
    embedding_variables["table_c"] = (
        [
            jax.device_put(
                emb_table_c_sharded[0],
                device=devices[0],
            )
        ],
        [
            jax.device_put(
                accumulator_init,
                device=devices[0],
            )
        ],
    )

    embedding_variables["table_c"] = (
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_C, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][0],
        ),
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_C, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][1],
        ),
    )

    activations_grad = {}
    activations_grad["feature_spec_a"] = jnp.ones(
        (_BATCH_SIZE, _DIM_A),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_b"] = jnp.ones(
        (_BATCH_SIZE, _DIM_B),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_c"] = jnp.ones(
        (_BATCH_SIZE, _DIM_C),
        dtype=jnp.float32,
    )
    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy="MOD",
    )
    sharded_grad_update = jax.jit(sharded_grad_update)
    grad_update = sharded_grad_update(
        activations_grad,
        preprocessed_inputs,
        embedding_variables,
    )
    expected_grad_table_a = np.zeros((_VOC_A, _DIM_A), dtype=np.float32)
    expected_grad_table_b = np.zeros((_VOC_B, _DIM_B), dtype=np.float32)

    # Generate the expected updates.
    # For each col ID, we subtract 0.01 (the learning rate) times the number of
    # times the ID appears in the input.
    for i, array in enumerate(embedding_variables["table_a"][0]):
      col_id = array[0]
      new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      expected_grad_table_a[i] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    for i, array in enumerate(embedding_variables["table_b"][0]):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_b, col_id) * 0.01
      )
      expected_grad_table_b[i] = np.full(
          (1, _DIM_B), new_col_id, dtype=np.float32
      )
    expected_table_c = np.array(
        [
            [-1.000e-02] * _DIM_C,
            [3.990e00] * _DIM_C,
            [7.990e00] * _DIM_C,
            [1.199e01] * _DIM_C,
            [1.600e01] * _DIM_C,
            [2.000e01] * _DIM_C,
            [2.400e01] * _DIM_C,
            [2.800e01] * _DIM_C,
            [9.900e-01] * _DIM_C,
            [4.990e00] * _DIM_C,
            [8.990e00] * _DIM_C,
            [1.299e01] * _DIM_C,
            [1.700e01] * _DIM_C,
            [2.100e01] * _DIM_C,
            [2.500e01] * _DIM_C,
            [2.900e01] * _DIM_C,
            [1.990e00] * _DIM_C,
            [5.990e00] * _DIM_C,
            [9.990e00] * _DIM_C,
            [1.399e01] * _DIM_C,
            [1.800e01] * _DIM_C,
            [2.200e01] * _DIM_C,
            [2.600e01] * _DIM_C,
            [3.000e01] * _DIM_C,
            [2.990e00] * _DIM_C,
            [6.990e00] * _DIM_C,
            [1.099e01] * _DIM_C,
            [1.499e01] * _DIM_C,
            [1.900e01] * _DIM_C,
            [2.300e01] * _DIM_C,
            [2.700e01] * _DIM_C,
            [3.100e01] * _DIM_C,
        ],
        dtype=np.float32,
    )

    expected_accumulator_c = np.array(
        [
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(
        expected_grad_table_a, grad_update["table_a"][0][:, :_DIM_A]
    )
    np.testing.assert_equal(
        expected_grad_table_b, grad_update["table_b"][0][:, :_DIM_B]
    )
    np.testing.assert_equal(
        expected_table_c, grad_update["table_c"][0][:, :_DIM_C]
    )
    np.testing.assert_equal(
        expected_accumulator_c, grad_update["table_c"][1][:, :_DIM_C]
    )

  @parameterized.parameters(
      embedding.StackingStrategy.STACK_THEN_SPLIT,
      embedding.StackingStrategy.SPLIT_THEN_STACK,
  )
  def test_tpu_sparse_dense_matmul_grad_sharded_two_tables(
      self, feature_stacking_strategy
  ):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    num_devices = len(devices)
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_b": self.feature_spec_b,
        "feature_spec_c": self.feature_spec_c,
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
            "feature_spec_c": self.input_tensor_table_c,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=num_devices,
        global_device_count=num_devices,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        batch_number=batch_number,
        feature_stacking_strategy=feature_stacking_strategy,
    )
    table_dim_a = table_stacking._next_largest_multiple(_DIM_A, 8)
    emb_table_a = (
        np.array([[i for _ in range(table_dim_a)] for i in range(_VOC_A)])
        .reshape(_VOC_A, table_dim_a)
        .astype(np.float32)
    )
    emb_table_a_sharded = einops.rearrange(
        emb_table_a,
        "(v c s) f -> c (s v) f",
        c=2,
        s=4,
    )
    embedding_variables = {}
    embedding_variables["table_a"] = [
        jax.device_put(
            emb_table_a_sharded[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_a"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_A, table_dim_a),
            sharding=sharding,
            arrays=embedding_variables["table_a"],
        )
    ])

    table_dim_b = table_stacking._next_largest_multiple(_DIM_B, 8)
    emb_table_b = (
        np.array([[i for _ in range(table_dim_b)] for i in range(_VOC_B)])
        .reshape(_VOC_B, table_dim_b)
        .astype(np.float32)
    )
    emb_table_b_sharded = einops.rearrange(
        emb_table_b,
        "(v c s) f -> c (s v) f",
        c=2,
        s=4,
    )
    embedding_variables["table_b"] = [
        jax.device_put(
            emb_table_b_sharded[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    embedding_variables["table_b"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_B, table_dim_b),
            sharding=sharding,
            arrays=embedding_variables["table_b"],
        )
    ])
    table_dim_c = table_stacking._next_largest_multiple(_DIM_C, 8)
    emb_table_c = (
        np.array([[i for _ in range(table_dim_c)] for i in range(_VOC_C)])
        .reshape(_VOC_C, table_dim_c)
        .astype(np.float32)
    )
    emb_table_c_sharded = einops.rearrange(
        emb_table_c,
        "(v c s) f -> c (s v) f",
        c=2,
        s=4,
    )
    embedding_variables["table_c"] = (
        [
            jax.device_put(
                emb_table_c_sharded[i],
                device=local_device,
            )
            for i, local_device in enumerate(devices)
        ],
        [
            jax.device_put(
                np.zeros((_VOC_C // num_devices, table_dim_c)),
                device=local_device,
            )
            for _, local_device in enumerate(devices)
        ],
    )
    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_c"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_C, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][0],
        ),
        jax.make_array_from_single_device_arrays(
            shape=(_VOC_C, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][1],
        ),
    ])
    activations_grad = {}
    activations_grad["feature_spec_a"] = jnp.ones(
        (_BATCH_SIZE, _DIM_A),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_b"] = jnp.ones(
        (_BATCH_SIZE, _DIM_B),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_c"] = jnp.ones(
        (_BATCH_SIZE, _DIM_C), dtype=jnp.float32
    )
    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy="MOD",
        feature_stacking_strategy=feature_stacking_strategy,
    )
    sharded_grad_update = jax.shard_map(
        sharded_grad_update,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(mesh.axis_names[0], None),
        check_vma=False,
    )
    sharded_grad_update = jax.jit(sharded_grad_update)
    grad_update = sharded_grad_update(
        activations_grad,
        preprocessed_inputs,
        embedding_variables,
    )
    expected_grad_table_a = np.zeros((_VOC_A, _DIM_A), dtype=np.float32)
    expected_grad_table_b = np.zeros((_VOC_B, _DIM_B), dtype=np.float32)

    # Generate the expected updates.
    # For each col ID, we subtract 0.01 (the learning rate) times the number of
    # times the ID appears in the input.
    for i, array in enumerate(embedding_variables["table_a"][0]):
      col_id = array[0]
      new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      expected_grad_table_a[i] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    for i, array in enumerate(embedding_variables["table_b"][0]):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_b, col_id) * 0.01
      )
      expected_grad_table_b[i] = np.full(
          (1, _DIM_B), new_col_id, dtype=np.float32
      )
    expected_table_c = np.array(
        [
            [-1.000e-02] * _DIM_C,
            [7.990e00] * _DIM_C,
            [1.600e01] * _DIM_C,
            [2.400e01] * _DIM_C,
            [9.900e-01] * _DIM_C,
            [8.990e00] * _DIM_C,
            [1.700e01] * _DIM_C,
            [2.500e01] * _DIM_C,
            [1.990e00] * _DIM_C,
            [9.990e00] * _DIM_C,
            [1.800e01] * _DIM_C,
            [2.600e01] * _DIM_C,
            [2.990e00] * _DIM_C,
            [1.099e01] * _DIM_C,
            [1.900e01] * _DIM_C,
            [2.700e01] * _DIM_C,
            [3.990e00] * _DIM_C,
            [1.199e01] * _DIM_C,
            [2.000e01] * _DIM_C,
            [2.800e01] * _DIM_C,
            [4.990e00] * _DIM_C,
            [1.299e01] * _DIM_C,
            [2.100e01] * _DIM_C,
            [2.900e01] * _DIM_C,
            [5.990e00] * _DIM_C,
            [1.399e01] * _DIM_C,
            [2.200e01] * _DIM_C,
            [3.000e01] * _DIM_C,
            [6.990e00] * _DIM_C,
            [1.499e01] * _DIM_C,
            [2.300e01] * _DIM_C,
            [3.100e01] * _DIM_C,
        ],
        dtype=np.float32,
    )

    expected_accumulator_c = np.array(
        [
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(
        expected_grad_table_a, grad_update["table_a"][0][:, :_DIM_A]
    )
    np.testing.assert_equal(
        expected_grad_table_b, grad_update["table_b"][0][:, :_DIM_B]
    )
    np.testing.assert_equal(
        expected_table_c, grad_update["table_c"][0][:, :_DIM_C]
    )
    np.testing.assert_equal(
        expected_accumulator_c, grad_update["table_c"][1][:, :_DIM_C]
    )

  @parameterized.parameters(
      embedding.StackingStrategy.STACK_THEN_SPLIT,
      embedding.StackingStrategy.SPLIT_THEN_STACK,
  )
  def test_tpu_sparse_dense_matmul_grad_sharded_two_tables_stacked(
      self, feature_stacking_strategy
  ):
    devices = jax.devices()[:2]
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    num_devices = len(devices)
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
        "feature_spec_b": self.feature_spec_b,
        "feature_spec_c": self.feature_spec_c,
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
            "feature_spec_b": self.input_tensor_table_b,
            "feature_spec_c": self.input_tensor_table_c,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=num_devices,
        global_device_count=num_devices,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        batch_number=batch_number,
        feature_stacking_strategy=feature_stacking_strategy,
    )
    padded_vocab_a = 64
    padded_vocab_b = _VOC_B
    padded_vocab_c = 64
    table_dim_a = table_stacking._next_largest_multiple(_DIM_A, 8)
    table_dim_b = table_stacking._next_largest_multiple(_DIM_B, 8)
    table_dim_c = table_stacking._next_largest_multiple(_DIM_C, 8)
    emb_table_a = test_utils.row_id_initializer((padded_vocab_a, table_dim_a))
    # The embedding table B values start at 100 for easy testing.
    emb_table_b = test_utils.row_id_initializer(
        (padded_vocab_b, table_dim_b), offset=100
    )
    emb_table_c = test_utils.row_id_initializer((padded_vocab_c, table_dim_c))
    emb_sharded_per_device_ab = (
        test_utils.create_per_device_sharded_stacked_tables(
            [emb_table_a, emb_table_b],
            num_devices=num_devices,
            num_sparsecore_per_device=4,
            rotation=4,
        )
    )
    emb_sharded_per_device_c = (
        test_utils.create_per_device_sharded_stacked_tables(
            [emb_table_c],
            num_devices=num_devices,
            num_sparsecore_per_device=4,
            rotation=0,
        )
    )
    embedding_variables = {}
    embedding_variables["table_a_table_b"] = [
        jax.device_put(
            emb_sharded_per_device_ab[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_a_table_b"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_a + padded_vocab_b, table_dim_a),
            sharding=sharding,
            arrays=embedding_variables["table_a_table_b"],
        )
    ])
    embedding_variables["table_c"] = (
        [
            jax.device_put(
                emb_sharded_per_device_c[i],
                device=local_device,
            )
            for i, local_device in enumerate(devices)
        ],
        [
            jax.device_put(
                np.zeros((padded_vocab_c // num_devices, table_dim_c)),
                device=local_device,
            )
            for _, local_device in enumerate(devices)
        ],
    )
    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_c"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_c, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][0],
        ),
        jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_c, table_dim_c),
            sharding=sharding,
            arrays=embedding_variables["table_c"][1],
        ),
    ])
    activations_grad = {}
    activations_grad["feature_spec_a"] = jnp.ones(
        (_BATCH_SIZE, _DIM_A),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_b"] = jnp.ones(
        (_BATCH_SIZE, _DIM_B),
        dtype=jnp.float32,
    )
    activations_grad["feature_spec_c"] = jnp.ones(
        (_BATCH_SIZE, _DIM_C), dtype=jnp.float32
    )
    sharded_grad_update = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy="MOD",
        feature_stacking_strategy=feature_stacking_strategy,
    )
    sharded_grad_update = jax.shard_map(
        sharded_grad_update,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
        ),
        out_specs=P(mesh.axis_names[0], None),
        check_vma=False,
    )
    sharded_grad_update = jax.jit(sharded_grad_update)
    grad_update = sharded_grad_update(
        activations_grad, preprocessed_inputs, embedding_variables
    )
    expected_grad_table_ab = np.asarray(emb_sharded_per_device_ab).reshape(
        -1, table_dim_a
    )
    expected_grad_table_c = np.zeros(
        (padded_vocab_c, table_dim_c), dtype=np.float32
    )

    # Generate the expected updates.
    # For each col ID, we subtract 0.01 (the learning rate) times the number of
    # times the ID appears in the input.
    for i, array in enumerate(embedding_variables["table_a_table_b"][0]):
      col_id = array[0]
      if col_id < 100:
        # Rows for table A
        new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
        expected_grad_table_ab[i, :_DIM_A] = np.full(
            (1, _DIM_A), new_col_id, dtype=np.float32
        )
      else:
        # Rows for table B
        new_col_id = col_id - (
            count_num(self.input_tensor_table_b, col_id - 100) * 0.01
        )
        expected_grad_table_ab[i, :_DIM_B] = np.full(
            (1, _DIM_B), new_col_id, dtype=np.float32
        )

    for i, array in enumerate(embedding_variables["table_c"][0]):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_c, col_id) * 0.01
      )
      expected_grad_table_c[i, :_DIM_C] = np.full(
          (1, _DIM_C), new_col_id, dtype=np.float32
      )

    expected_accumulator_c = np.array(
        [
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [1.0] * _DIM_C,
            [1.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
            [0.0] * _DIM_C,
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(
        expected_grad_table_ab,
        grad_update["table_a_table_b"][0],
    )
    np.testing.assert_equal(
        expected_grad_table_c[:_VOC_C, :_DIM_C],
        grad_update["table_c"][0][:_VOC_C, :_DIM_C],
    )
    np.testing.assert_equal(
        expected_accumulator_c[:_VOC_C, :_DIM_C],
        grad_update["table_c"][1][:_VOC_C, :_DIM_C],
    )

  @parameterized.named_parameters(
      ("constant", 0.01),
      ("function", lambda step: 0.01 / (step + 1)),
      ("object", LinearLearningRateSchedule(0.01)),
  )
  def test_tpu_sparse_dense_matmul_grad_with_learning_rate(
      self,
      learning_rate: float | Callable[..., float | jax.Array],
  ):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    num_devices = len(devices)
    mesh = jax.sharding.Mesh(devices, "x")
    feature_specs = {
        "feature_spec_a": self.feature_spec_a,
    }

    table_a_optimizer = embedding_spec.SGDOptimizerSpec(
        learning_rate=learning_rate
    )
    feature_specs["feature_spec_a"].table_spec.optimizer = table_a_optimizer

    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=num_devices,
        num_sc_per_device=num_sc_per_device,
    )
    batch_number = 42
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        {
            "feature_spec_a": self.input_tensor,
        },
        features_weights=None,  # uniform weights
        feature_specs=feature_specs,
        local_device_count=num_devices,
        global_device_count=num_devices,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy="MOD",
        batch_number=batch_number,
    )
    padded_vocab_a = feature_specs[
        "feature_spec_a"
    ].table_spec.setting_in_stack.padded_vocab_size
    padded_embedding_dim_a = feature_specs[
        "feature_spec_a"
    ].table_spec.setting_in_stack.padded_embedding_dim

    emb_table_a = np.array([
        [i for _ in range(padded_embedding_dim_a)]
        for i in range(padded_vocab_a)
    ]).astype(np.float32)
    emb_table_a_sharded = einops.rearrange(
        emb_table_a,
        "(v c s) f -> c (s v) f",
        c=num_devices,
        s=num_sc_per_device,
    )
    embedding_variables = {}
    embedding_variables["table_a"] = [
        jax.device_put(
            emb_table_a_sharded[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, P("x", None))
    embedding_variables["table_a"] = tuple([
        jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_a, padded_embedding_dim_a),
            sharding=sharding,
            arrays=embedding_variables["table_a"],
        )
    ])

    activations_grad = {}
    activations_grad["feature_spec_a"] = jnp.ones(
        (_BATCH_SIZE, _DIM_A),
        dtype=jnp.float32,
    )

    def sharded_grad_update(
        activation_gradients,
        preprocessed_inputs,
        embedding_variables,
        step,
    ):
      return embedding.tpu_sparse_dense_matmul_grad(
          activation_gradients,
          preprocessed_inputs,
          embedding_variables,
          feature_specs=feature_specs,
          step=step,
      )

    sharded_grad_update = jax.shard_map(
        sharded_grad_update,
        mesh=mesh,
        in_specs=(
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0]),
            P(mesh.axis_names[0], None),
            P(),  # Step is replicated.
        ),
        out_specs=P(mesh.axis_names[0], None),
        check_vma=False,
    )
    sharded_grad_update = jax.jit(sharded_grad_update)

    for step in range(10):
      # In Keras models, the step counter is a 0-D integer array.
      step_var = jnp.array(step, dtype=jnp.int32)
      grad_update = sharded_grad_update(
          activations_grad,
          preprocessed_inputs,
          embedding_variables,
          step_var,
      )

      # Generate the expected updates.
      # For each col ID, we subtract the learning rate times the number of
      # times the ID appears in the input.
      expected_grad_table_a = np.zeros(
          (padded_vocab_a, _DIM_A), dtype=np.float32
      )
      for i, array in enumerate(embedding_variables["table_a"][0]):
        col_id = array[0]
        new_col_id = col_id - (
            count_num(self.input_tensor, col_id)
        ) * table_a_optimizer.get_learning_rate(step)
        expected_grad_table_a[i] = np.full(
            (1, _DIM_A), new_col_id, dtype=np.float32
        )

      np.testing.assert_allclose(
          expected_grad_table_a, grad_update["table_a"][0][:, :_DIM_A]
      )


if __name__ == "__main__":
  absltest.main()
