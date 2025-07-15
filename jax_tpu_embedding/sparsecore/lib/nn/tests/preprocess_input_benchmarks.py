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

import sys
from absl import app
from absl import flags
import google_benchmark
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np

_NUM_FEATURES = flags.DEFINE_integer("num_features", 100, "Number of features.")
_NUM_SAMPLES = flags.DEFINE_integer("num_samples", 16000, "Number of samples.")

_GLOBAL_SPECS = None
_GLOBAL_RAGGED_FEATURES = None
_GLOBAL_RAGGED_WEIGHTS = None
_GLOBAL_DENSE_FEATURES = None
_GLOBAL_DENSE_WEIGHTS = None
_GLOBAL_RAGGED_INDICES = None
_GLOBAL_RAGGED_VALUES = None
_GLOBAL_RAGGED_DENSE_SHAPES = None


def generate_feature_specs(num_features: int, num_samples: int):
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
        _stacked_table_spec=embedding_spec.StackedTableSpec(
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


def generate_sparse_coo_inputs_for_feature_spec(
    feature_specs, num_samples, vocab_size
):
  """Generates random samples for a given feature spec."""
  all_indices_tensors = []
  all_values_tensors = []
  all_dense_shape_tensors = []

  for feature_spec in feature_specs:
    table_spec = feature_spec.table_spec
    indices_tensors = []
    values_tensors = []
    for i in range(num_samples):
      num_ids = np.random.randint(1, 32)
      for j in range(num_ids):
        indices_tensors.append([i, j])
      for _ in range(num_ids):
        values_tensors.append(np.random.randint(table_spec.vocabulary_size))
    all_indices_tensors.append(np.array(indices_tensors, dtype=np.int64))
    all_values_tensors.append(np.array(values_tensors, dtype=np.int32))
    all_dense_shape_tensors.append(
        np.array([num_samples, vocab_size], dtype=np.int64)
    )
  return all_indices_tensors, all_values_tensors, all_dense_shape_tensors


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["ragged", "has_leading_dimension"])
@google_benchmark.option.args_product([[False, True], [False, True]])
@google_benchmark.option.iterations(100)
def preprocess_input_benchmark(state: google_benchmark.State):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  ragged, has_leading_dimension = state.range(0), state.range(1)
  if ragged:
    features, feature_weights = _GLOBAL_RAGGED_FEATURES, _GLOBAL_RAGGED_WEIGHTS
  else:
    features, feature_weights = _GLOBAL_DENSE_FEATURES, _GLOBAL_DENSE_WEIGHTS
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        features,
        feature_weights,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        has_leading_dimension=has_leading_dimension,
    )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_name("has_leading_dimension")
@google_benchmark.option.args_product([[False, True]])
@google_benchmark.option.iterations(100)
def preprocess_input_benchmark_sparse_coo(state: google_benchmark.State):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  has_leading_dimension = state.range(0)
  while state:
    _ = pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput(
        _GLOBAL_RAGGED_INDICES,
        _GLOBAL_RAGGED_VALUES,
        _GLOBAL_RAGGED_DENSE_SHAPES,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        has_leading_dimension=has_leading_dimension,
    )


def main(_):
  global _GLOBAL_SPECS
  global _GLOBAL_RAGGED_FEATURES
  global _GLOBAL_RAGGED_WEIGHTS
  global _GLOBAL_DENSE_FEATURES
  global _GLOBAL_DENSE_WEIGHTS
  global _GLOBAL_RAGGED_INDICES
  global _GLOBAL_RAGGED_VALUES
  global _GLOBAL_RAGGED_DENSE_SHAPES

  # Total local batch size that is measured is 16000x100 = 1,600,000.
  _GLOBAL_SPECS = generate_feature_specs(
      _NUM_FEATURES.value, _NUM_SAMPLES.value
  )
  _GLOBAL_RAGGED_FEATURES, _GLOBAL_RAGGED_WEIGHTS = (
      generate_samples_for_feature_spec(
          _GLOBAL_SPECS, _NUM_SAMPLES.value, ragged=True
      )
  )
  _GLOBAL_DENSE_FEATURES, _GLOBAL_DENSE_WEIGHTS = (
      generate_samples_for_feature_spec(
          _GLOBAL_SPECS, _NUM_SAMPLES.value, ragged=False
      )
  )
  _GLOBAL_RAGGED_INDICES, _GLOBAL_RAGGED_VALUES, _GLOBAL_RAGGED_DENSE_SHAPES = (
      generate_sparse_coo_inputs_for_feature_spec(
          _GLOBAL_SPECS, _NUM_SAMPLES.value, 1024
      )
  )
  google_benchmark.run_benchmarks()


if __name__ == "__main__":
  sys.argv = google_benchmark.initialize(sys.argv)
  app.run(main)
