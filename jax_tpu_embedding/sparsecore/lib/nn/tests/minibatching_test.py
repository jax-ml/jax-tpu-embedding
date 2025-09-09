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
import concurrent

from absl.testing import absltest
import jax
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import portpicker


def _generate_random_inputs(
    feature: embedding_spec.FeatureSpec, max_sample_size: int
):
  """Generates random inputs and input weights for testing."""
  inputs = []
  inputs_weights = []
  for _ in range(feature.input_shape[0]):
    num_ids = np.random.randint(1, max_sample_size + 1)
    ids = np.random.randint(
        0, feature.table_spec.vocabulary_size, size=(num_ids,), dtype=np.int32
    )
    inputs.append(ids)
    inputs_weights.append(np.ones_like(ids, dtype=np.float32))
  return np.array(inputs, dtype=object), np.array(inputs_weights, dtype=object)


class SingleHostMinibatchingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
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
    )

    self.assertTrue((preprocessed_input.num_minibatches > 1).all())


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
        feature=self.feature_spec, max_sample_size=10
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
        self.assertTrue((future.result().num_minibatches == 1).all())

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
        self.assertTrue((future.result().num_minibatches > 1).all())


if __name__ == "__main__":
  absltest.main()
