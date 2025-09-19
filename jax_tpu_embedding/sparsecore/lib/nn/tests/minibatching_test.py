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
import concurrent
import dataclasses

from absl.testing import absltest
from jax import numpy as jnp
from jax.experimental import shard_map
import jax.sharding
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import portpicker


def _generate_random_inputs(
    feature: embedding_spec.FeatureSpec,
    max_sample_size: int,
    seed: int | None = None,
):
  """Generates random inputs and input weights for testing."""
  rng = np.random.RandomState(seed)
  inputs = []
  inputs_weights = []
  for _ in range(feature.input_shape[0]):
    num_ids = rng.randint(1, max_sample_size + 1)
    ids = rng.randint(
        0, feature.table_spec.vocabulary_size, size=(num_ids,), dtype=np.int32
    )
    inputs.append(ids)
    inputs_weights.append(np.ones_like(ids, dtype=np.float32))
  return np.array(inputs, dtype=object), np.array(inputs_weights, dtype=object)


def _generate_expected_activations(
    feature: embedding_spec.FeatureSpec, inputs: np.ndarray
) -> np.ndarray:
  """Generates expected activations using row-initializer embedding variables."""
  expected_activations_flat = []
  for i in range(feature.input_shape[0]):
    expected_val = 0
    if inputs[i].size > 0:
      expected_val = np.sum(inputs[i])
    expected_activations_flat.append(expected_val)
  expected_activations = np.array(expected_activations_flat, dtype=np.float32)
  expected_activations = np.tile(
      expected_activations[:, np.newaxis],
      (1, feature.output_shape[1]),
  )
  return expected_activations


def _generate_expected_emb_table(
    feature: embedding_spec.FeatureSpec,
    inputs: np.ndarray,
    num_sc_per_device: int,
) -> np.ndarray:
  """Generates expected gradients using all one activation gradients."""
  ids_count = collections.Counter()
  for ids_array in inputs:
    ids_count.update(ids_array)

  vocab_size = feature.table_spec.vocabulary_size
  emb_dim = feature.table_spec.embedding_dim
  expected_table_np = np.zeros((vocab_size, emb_dim), dtype=np.float32)
  num_shards = num_sc_per_device * jax.device_count()
  shard_size = vocab_size // num_shards
  for i in range(vocab_size):
    val = i  # row initializer.
    if i in ids_count:
      val -= feature.table_spec.optimizer.learning_rate * ids_count[i]
    shard_id = i % num_shards
    idx_in_shard = i // num_shards
    physical_row = shard_id * shard_size + idx_in_shard
    expected_table_np[physical_row, :] = val
  return expected_table_np


def _init_embedding_vars(
    table_spec,
    num_sc_per_device,
    devices,
    embedding_var_sharding,
) -> dict[str, embedding.EmbeddingVariables]:
  """Initializes embedding variables for testing."""
  vocab_size = table_spec.vocabulary_size
  emb_dim = table_spec.embedding_dim
  emb_table_np = np.zeros((vocab_size, emb_dim), dtype=np.float32)
  for i in range(vocab_size):
    emb_table_np[i, :] = i
  emb_tables = [jnp.array(emb_table_np)]

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
  emb_var = jax.make_array_from_single_device_arrays(
      shape=(vocab_size, emb_dim),
      sharding=embedding_var_sharding,
      arrays=embedding_variable_shards,
  )
  return {"table_a": embedding.EmbeddingVariables(table=emb_var, slot=())}


class SingleHostMinibatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_sc_per_device = utils.num_sparsecores_per_device()
    self.table_spec = embedding_spec.TableSpec(
        vocabulary_size=2048,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
        suggested_coo_buffer_size_per_device=8192,
    )
    self.feature_spec = embedding_spec.FeatureSpec(
        table_spec=self.table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
    )
    embedding.prepare_feature_specs_for_training(
        [self.feature_spec],
        global_device_count=jax.device_count(),
    )
    self.port = portpicker.pick_unused_port()
    self.all_reduce_interface = embedding.get_all_reduce_interface(
        peer_addresses=[], minibatching_port=self.port
    )
    self.devices = jax.devices()
    self.mesh = jax.sharding.Mesh(np.array(self.devices), axis_names=["device"])
    self.pd = jax.sharding.PartitionSpec("device")
    self.pe = jax.sharding.PartitionSpec("device", None)
    self.embedding_var_sharding = jax.sharding.NamedSharding(self.mesh, self.pe)
    self.data_sharding = jax.sharding.NamedSharding(self.mesh, self.pd)
    self.embedding_vars = _init_embedding_vars(
        self.table_spec,
        self.num_sc_per_device,
        self.devices,
        self.embedding_var_sharding,
    )
    self.sharded_lookup = jax.jit(
        shard_map.shard_map(
            self._lookup,
            mesh=self.mesh,
            in_specs=(self.pd, self.pe),
            out_specs=(self.pd),
            check_rep=False,
        ),
        in_shardings=(self.data_sharding, self.embedding_var_sharding),
    )
    self.sharded_update = jax.jit(
        shard_map.shard_map(
            self._update,
            mesh=self.mesh,
            in_specs=(self.pd, self.pd, self.pe),
            out_specs=(self.pe),
            check_rep=False,
        ),
        in_shardings=(
            self.data_sharding,
            self.data_sharding,
            self.embedding_var_sharding,
        ),
    )

  def test_single_host_minibatching_not_required(self):
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=8
    )
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
    )

    self.assertTrue((preprocessed_input.num_minibatches == 1).all())

  def test_single_host_minibatching_required(self):
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=1, max_unique_ids_per_partition=1
        )
    )
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=20
    )
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
        allow_id_dropping=True,
    )

    self.assertTrue((preprocessed_input.num_minibatches > 1).all())

  def test_single_host_minibatching_id_dropping(self):
    # Set max_ids_per_partition to a small value to trigger id dropping.
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=1, max_unique_ids_per_partition=1
        )
    )
    # Use large batch size to increase chance of dropping
    batch_size = (
        jax.device_count() * 4 * 16
    )  # Multiple of devices * sc_per_device
    self.feature_spec = dataclasses.replace(
        self.feature_spec, input_shape=[batch_size, 1]
    )
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=8
    )

    _, stats = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
        allow_id_dropping=True,
    )
    self.assertIn("table_a", stats.id_drop_counters)
    self.assertGreater(stats.id_drop_counters["table_a"], 0)

  def test_single_host_minibatching_unique_id_dropping(self):
    # Set max_unique_ids_per_partition to a small value to trigger unique id
    # dropping.
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=10, max_unique_ids_per_partition=1
        )
    )
    # Use large batch size to increase chance of dropping
    batch_size = (
        jax.device_count() * 4 * 16
    )  # Multiple of devices * sc_per_device
    self.feature_spec = dataclasses.replace(
        self.feature_spec, input_shape=[batch_size, 1]
    )
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=8
    )

    _, stats = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
        allow_id_dropping=True,
    )
    self.assertIn("table_a", stats.id_drop_counters)
    self.assertGreater(stats.id_drop_counters["table_a"], 0)

  def _lookup(self, preprocessed_input, embedding_vars):
    return embedding.tpu_sparse_dense_matmul(
        preprocessed_input,
        embedding_vars,
        {"feature_a": self.feature_spec},
        global_device_count=self.mesh.size,
        enable_minibatching=True,
    )

  def _update(self, activation_gradients, preprocessed_input, embedding_vars):
    return embedding.tpu_sparse_dense_matmul_grad(
        activation_gradients,
        preprocessed_input,
        embedding_vars,
        {"feature_a": self.feature_spec},
        enable_minibatching=True,
    )

  def test_single_host_minibatching_forward_pass(self):
    # Test forward pass with minibatching
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=2, max_unique_ids_per_partition=2
        )
    )
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=10, seed=2025
    )

    preprocessed_input, stats = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
        allow_id_dropping=True,
    )
    self.assertGreater(preprocessed_input.num_minibatches[0], 1)
    self.assertEqual(stats.id_drop_counters["table_a"], 0)

    activations = self.sharded_lookup(preprocessed_input, self.embedding_vars)

    expected_activations = _generate_expected_activations(
        self.feature_spec, inputs
    )
    self.assertIn("feature_a", activations)
    np.testing.assert_allclose(
        activations["feature_a"], expected_activations, rtol=1e-6
    )

  def test_single_host_minibatching_backward_pass(self):
    # Test backward pass with minibatching
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=2, max_unique_ids_per_partition=2
        )
    )
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=10, seed=2025
    )

    preprocessed_input, stats = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interface,
        allow_id_dropping=True,
    )
    self.assertGreater(preprocessed_input.num_minibatches[0], 1)
    self.assertEqual(stats.id_drop_counters["table_a"], 0)

    # Create dummy gradients
    activation_gradients = {
        "feature_a": jnp.ones(self.feature_spec.output_shape, dtype=jnp.float32)
    }

    updated_vars: dict[str, embedding.EmbeddingVariables] = self.sharded_update(
        activation_gradients, preprocessed_input, self.embedding_vars
    )

    self.assertIn("table_a", updated_vars)

    expected_table_np = _generate_expected_emb_table(
        self.feature_spec, inputs, self.num_sc_per_device
    )

    np.testing.assert_allclose(
        updated_vars["table_a"].table, expected_table_np, rtol=1e-6
    )


class MultiHostMinibatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
        suggested_coo_buffer_size_per_device=16384,
    )
    self.feature_spec = embedding_spec.FeatureSpec(
        table_spec=self.table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
    )
    embedding.prepare_feature_specs_for_training(
        [self.feature_spec],
        global_device_count=jax.device_count(),
    )
    self.num_hosts = 4
    self.ports = [portpicker.pick_unused_port() for _ in range(self.num_hosts)]
    self.all_reduce_interfaces = []
    for host_id in range(self.num_hosts):
      peer_addresses = [
          f"localhost:{self.ports[i]}"
          for i in range(self.num_hosts)
          if i != host_id
      ]
      self.all_reduce_interfaces.append(
          embedding.get_all_reduce_interface(
              host_id=host_id,
              host_count=self.num_hosts,
              peer_addresses=peer_addresses,
              minibatching_port=self.ports[host_id],
          )
      )

  def worker(self, host_id: int):
    inputs, inputs_weights = _generate_random_inputs(
        feature=self.feature_spec, max_sample_size=10, seed=2025
    )
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features=[inputs],
        features_weights=[inputs_weights],
        feature_specs=[self.feature_spec],
        local_device_count=jax.device_count(),
        global_device_count=jax.device_count(),
        batch_number=42,
        enable_minibatching=True,
        all_reduce_interface=self.all_reduce_interfaces[host_id],
    )
    return preprocessed_input

  def test_multi_host_minibatching_not_required(self):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.num_hosts
    ) as executor:
      futures = [
          executor.submit(self.worker, host_id)
          for host_id in range(self.num_hosts)
      ]
      for future in futures:
        self.assertEqual(future.result().num_minibatches[0], 1)

  def test_multi_host_minibatching_required(self):
    self.feature_spec.table_spec.stacked_table_spec = (
        self.feature_spec.table_spec.stacked_table_spec.replace(
            max_ids_per_partition=1, max_unique_ids_per_partition=1
        )
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.num_hosts
    ) as executor:
      futures = [
          executor.submit(self.worker, host_id)
          for host_id in range(self.num_hosts)
      ]
      for future in futures:
        self.assertGreater(future.result().num_minibatches[0], 1)


if __name__ == "__main__":
  absltest.main()
