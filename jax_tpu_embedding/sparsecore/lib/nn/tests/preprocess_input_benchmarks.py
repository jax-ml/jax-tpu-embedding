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
"""Simple benchmarks for preprocessing input for sparse-dense matmul.

Example usage:

On perflab comparing against HEAD:
benchy --perflab --runs=10 --reference=srcfs --benchmark_filter=all
:preprocess_input_benchmarks

Or locally:
bazel run -c opt --dynamic_mode=off --copt=-gmlt :preprocess_input_benchmarks --
--benchmark_filter=all --cpu_profile=/tmp/preprocess.prof
"""

import google_benchmark
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


def generate_feature_specs(num_features, num_samples):
  """Generates feature specs for the given number of features."""
  feature_specs = []
  for i in range(num_features):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=1024,
        embedding_dim=16,
        initializer=lambda: np.zeros((1024, 16), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="table_{}".format(i),
        max_ids_per_partition=1024,
        max_unique_ids_per_partition=1024,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_{}".format(i),
            padded_vocab_size=1024,
            padded_embedding_dim=16,
            row_offset_in_shard=0,
            shard_rotation=0,
        ),
        stacked_table_spec=embedding_spec.StackedTableSpec(
            stack_name="table_{}".format(i),
            stack_vocab_size=1024,
            stack_embedding_dim=16,
            optimizer=embedding_spec.SGDOptimizerSpec(
                learning_rate=0.001,
            ),
            combiner="sum",
            total_sample_count=num_samples,
            max_ids_per_partition=1024,
            max_unique_ids_per_partition=1024,
        ),
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[num_samples, 1],
        output_shape=[num_samples, 16],
        name="feature_spec_{}".format(i),
    )
    feature_specs.append(feature_spec)
  return feature_specs


def generate_samples_for_feature_spec(feature_specs, num_samples, ragged=False):
  """Generates random samples for a given feature spec."""
  all_features = []
  all_feature_weights = []
  for feature_spec in feature_specs:
    table_spec = feature_spec.table_spec
    if not ragged:
      features = np.random.randint(
          table_spec.vocabulary_size,
          size=(num_samples, 16),
          dtype=np.int32,
      )
      feature_weights = np.ones(
          (num_samples, table_spec.embedding_dim), dtype=np.float32
      )
      all_features.append(features)
      all_feature_weights.append(feature_weights)
    else:
      features = []
      feature_weights = []
      for _ in range(num_samples):
        num_ids = np.random.randint(1, 32)
        ids = np.random.randint(
            table_spec.vocabulary_size,
            size=(num_ids,),
            dtype=np.int32,
        )
        features.append(ids)
        feature_weights.append(np.ones((num_ids,), dtype=np.float32))
      all_features.append(np.array(features, dtype=object))
      all_feature_weights.append(np.array(feature_weights, dtype=object))
  return all_features, all_feature_weights

# Total local batch size that is measured is 16000x100 = 1,600,000.
_GLOBAL_SPECS = generate_feature_specs(num_features=100, num_samples=16000)
_GLOBAL_RAGGED_FEATURES, _GLOBAL_RAGGED_FEATURE_WEIGHTS = (
    generate_samples_for_feature_spec(
        _GLOBAL_SPECS, num_samples=16000, ragged=True
    )
)
_GLOBAL_2D_FEATURES, _GLOBAL_2D_FEATURE_WEIGHTS = (
    generate_samples_for_feature_spec(
        _GLOBAL_SPECS, num_samples=16000, ragged=False
    )
)


@google_benchmark.register
def preprocess_input_benchmark_ragged_tensor_jit(state):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        _GLOBAL_RAGGED_FEATURES,
        _GLOBAL_RAGGED_FEATURE_WEIGHTS,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        sharding_strategy=1,
        has_leading_dimension=False,
        allow_id_dropping=False,
    )


@google_benchmark.register
def preprocess_input_benchmark_ragged_tensor_pmap(state):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        _GLOBAL_RAGGED_FEATURES,
        _GLOBAL_RAGGED_FEATURE_WEIGHTS,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        sharding_strategy=1,
        has_leading_dimension=True,
        allow_id_dropping=False,
    )


@google_benchmark.register
def preprocess_input_benchmark_dense_tensor_jit(state):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        _GLOBAL_2D_FEATURES,
        _GLOBAL_2D_FEATURE_WEIGHTS,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        sharding_strategy=1,
        has_leading_dimension=False,
        allow_id_dropping=False,
    )


@google_benchmark.register
def preprocess_input_benchmark_dense_tensor_pmap(state):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        _GLOBAL_2D_FEATURES,
        _GLOBAL_2D_FEATURE_WEIGHTS,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        sharding_strategy=1,
        has_leading_dimension=True,
        allow_id_dropping=False,
    )


if __name__ == "__main__":
  google_benchmark.main()
