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

_BATCH_SIZE = 64


def _create_embedding_variable(
    var_shapes: list[VariableInfo],
    devices: list[jax.Device],
    mesh: jax.sharding.Mesh,
) -> jax.Array:
  """Creates a stacked embedding variable sharded across `devices`."""
  num_sc_per_device = utils.num_sparsecores_per_device(mesh.devices.item(0))
  dim = var_shapes[0].shape[1]
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
  return jax.make_array_from_single_device_arrays(
      shape=(total_vocab, dim),
      sharding=NamedSharding(mesh, P("x", None)),
      arrays=[
          jax.device_put(emb_table_sharded[i], device=device)
          for i, device in enumerate(devices)
      ],
  )


class TpuSparseDenseMatmulFdoStatisticsTest(parameterized.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.devices = jax.devices()[:2]
    self.mesh = jax.sharding.Mesh(self.devices, "x")
    self.num_sc_per_device = utils.num_sparsecores_per_device(self.devices[0])

    def table_spec(name: str) -> embedding_spec.TableSpec:
      return embedding_spec.TableSpec(
          vocabulary_size=64,
          embedding_dim=8,
          initializer=lambda *_: jnp.zeros((64, 8), dtype=jnp.float32),
          optimizer=embedding_spec.SGDOptimizerSpec(),
          combiner="sum",
          name=name,
          # Large enough that no IDs are dropped, so that the host and the
          # device observe the same IDs.
          max_ids_per_partition=64,
          max_unique_ids_per_partition=64,
      )

    self.feature_specs = {
        name: embedding_spec.FeatureSpec(
            table_spec=table_spec(f"table_{name}"),
            input_shape=(_BATCH_SIZE, 2),
            output_shape=(_BATCH_SIZE, 8),
            name=name,
        )
        for name in ("feature_a", "feature_b")
    }
    embedding.prepare_feature_specs_for_training(
        self.feature_specs,
        global_device_count=len(self.devices),
        num_sc_per_device=self.num_sc_per_device,
    )
    # In `feature_a`, every sample looks up ID 0, so the partition of ID 0
    # receives many IDs but few unique IDs, and the maximum number of IDs and of
    # unique IDs per partition differ. `feature_b` repeats a few IDs.
    self.features = {
        "feature_a": np.array(
            [[0, i] for i in range(_BATCH_SIZE)], dtype=np.int32
        ),
        "feature_b": np.array(
            [[i % 4, i % 3] for i in range(_BATCH_SIZE)], dtype=np.int32
        ),
    }
    self.preprocessed_inputs, self.host_stats = (
        embedding.preprocess_sparse_dense_matmul_input(
            self.features,
            features_weights=None,  # uniform weights
            feature_specs=self.feature_specs,
            local_device_count=len(self.devices),
            global_device_count=len(self.devices),
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy="MOD",
            batch_number=42,
        )
    )
    self.embedding_variables = {
        name: embedding.EmbeddingVariables(
            table=_create_embedding_variable(
                [VariableInfo((64, 8), 0)], self.devices, self.mesh
            ),
            slot=(),
        )
        for name in ("table_feature_a", "table_feature_b")
    }

  def _sparse_dense_matmul(
      self, enable_fdo_statistics: bool, enable_minibatching: bool = False
  ):
    matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        feature_specs=tuple(jax.tree.leaves(self.feature_specs)),
        global_device_count=self.mesh.size,
        sharding_strategy="MOD",
        enable_minibatching=enable_minibatching,
        enable_fdo_statistics=enable_fdo_statistics,
    )
    return jax.jit(
        jax.shard_map(
            matmul,
            mesh=self.mesh,
            in_specs=(P("x"), P("x", None)),
            out_specs=P("x"),
            check_vma=False,
        )
    )

  def test_device_statistics_match_host_statistics(self):
    for table_name, drop_count in self.host_stats.id_drop_counters.items():
      self.assertEqual(drop_count, 0, msg=f"IDs dropped for {table_name}")

    _, device_statistics = self._sparse_dense_matmul(
        enable_fdo_statistics=True
    )(self.preprocessed_inputs, self.embedding_variables)

    # One value per device: the maximum over the device's SparseCores.
    for statistic in (
        device_statistics.max_ids_per_partition,
        device_statistics.max_unique_ids_per_partition,
    ):
      self.assertCountEqual(
          statistic.keys(), ["table_feature_a", "table_feature_b"]
      )
      for value in statistic.values():
        self.assertEqual(value.shape, (len(self.devices),))
        self.assertEqual(value.dtype, jnp.int32)

    stats = embedding.device_fdo_statistics_to_input_stats(device_statistics)
    for table_name in ("table_feature_a", "table_feature_b"):
      with self.subTest(table_name):
        np.testing.assert_array_equal(
            stats.max_ids_per_partition[table_name],
            [np.max(self.host_stats.max_ids_per_partition[table_name])],
        )
        np.testing.assert_array_equal(
            stats.max_unique_ids_per_partition[table_name],
            [np.max(self.host_stats.max_unique_ids_per_partition[table_name])],
        )
    # The statistics differ for `feature_a`, so the test can tell them apart.
    self.assertNotEqual(
        stats.max_ids_per_partition["table_feature_a"][0],
        stats.max_unique_ids_per_partition["table_feature_a"][0],
    )

  def test_activations_are_unchanged(self):
    expected_activations = self._sparse_dense_matmul(
        enable_fdo_statistics=False
    )(self.preprocessed_inputs, self.embedding_variables)

    activations, _ = self._sparse_dense_matmul(enable_fdo_statistics=True)(
        self.preprocessed_inputs, self.embedding_variables
    )

    jax.tree.map(
        np.testing.assert_array_equal, activations, expected_activations
    )

  def test_statistics_can_update_preprocessing_parameters(self):
    _, device_statistics = self._sparse_dense_matmul(
        enable_fdo_statistics=True
    )(self.preprocessed_inputs, self.embedding_variables)
    stats = embedding.device_fdo_statistics_to_input_stats(device_statistics)

    embedding.update_preprocessing_parameters(
        self.feature_specs, stats, self.num_sc_per_device
    )

    stacked_table_spec = self.feature_specs[
        "feature_a"
    ].table_spec.stacked_table_spec
    self.assertEqual(
        stacked_table_spec.max_ids_per_partition,
        stats.max_ids_per_partition["table_feature_a"][0],
    )
    self.assertEqual(
        stacked_table_spec.max_unique_ids_per_partition,
        stats.max_unique_ids_per_partition["table_feature_a"][0],
    )

  def test_device_minibatching_statistics_match_host_statistics(self):
    # Device minibatching always produces a single minibatch per step, which
    # the TPU compiler requires for FDO statistics with minibatching.
    preprocessed_inputs, host_stats = (
        embedding.preprocess_sparse_dense_matmul_input(
            self.features,
            features_weights=None,  # uniform weights
            feature_specs=self.feature_specs,
            local_device_count=len(self.devices),
            global_device_count=len(self.devices),
            num_sc_per_device=self.num_sc_per_device,
            sharding_strategy="MOD",
            batch_number=42,
            minibatching_mode="DEVICE",
        )
    )
    np.testing.assert_array_equal(preprocessed_inputs.num_minibatches, 1)
    expected_activations = self._sparse_dense_matmul(
        enable_fdo_statistics=False, enable_minibatching=True
    )(preprocessed_inputs, self.embedding_variables)

    activations, device_statistics = self._sparse_dense_matmul(
        enable_fdo_statistics=True, enable_minibatching=True
    )(preprocessed_inputs, self.embedding_variables)

    jax.tree.map(
        np.testing.assert_array_equal, activations, expected_activations
    )
    stats = embedding.device_fdo_statistics_to_input_stats(device_statistics)
    for table_name in ("table_feature_a", "table_feature_b"):
      with self.subTest(table_name):
        np.testing.assert_array_equal(
            stats.max_ids_per_partition[table_name],
            [np.max(host_stats.max_ids_per_partition[table_name])],
        )
        np.testing.assert_array_equal(
            stats.max_unique_ids_per_partition[table_name],
            [np.max(host_stats.max_unique_ids_per_partition[table_name])],
        )

  def test_device_fdo_statistics_to_input_stats_reduces_over_devices(self):
    sharding = NamedSharding(self.mesh, P("x"))
    device_statistics = embedding.DeviceFdoStatistics(
        max_ids_per_partition={
            "table": jax.device_put(np.array([3, 7], np.int32), sharding)
        },
        max_unique_ids_per_partition={
            "table": jax.device_put(np.array([5, 2], np.int32), sharding)
        },
    )

    stats = embedding.device_fdo_statistics_to_input_stats(device_statistics)

    np.testing.assert_array_equal(stats.max_ids_per_partition["table"], [7])
    np.testing.assert_array_equal(
        stats.max_unique_ids_per_partition["table"], [5]
    )
    self.assertEmpty(stats.required_buffer_size_per_sc)
    self.assertEmpty(stats.id_drop_counters)


if __name__ == "__main__":
  absltest.main()
