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
r"""Simple benchmarks for preprocessing input for sparse-dense matmul.

Example usage:

On perflab comparing against HEAD:
benchy --perflab --runs=10 --reference=srcfs --benchmark_filter=all \
 //jax_tpu_embedding/sparsecore/lib/nn/tests:preprocess_input_benchmarks.par

Or locally:
bazel run -c opt --dynamic_mode=off --copt=-gmlt \
 //jax_tpu_embedding/sparsecore/lib/nn/tests:preprocess_input_benchmarks \
 -- --benchmark_filter=all --cpu_profile=/tmp/preprocess.prof

The --benchmark_filter flag uses a regex to select benchmarks. For parameterized
benchmarks, the name is typically formatted as:
Boolean parameters are often represented as 0 for False and 1 for True.
`[benchmark_name]/[param1]:[value1]`.

For example, to run only the `sparse_coo` benchmarks:
`--benchmark_filter=preprocess_input_benchmark_sparse_coo`

To run only the ragged benchmark with ragged=True:
`--benchmark_filter='preprocess_input_benchmark/ragged:1'`

To upload the profile to pprof:
pprof -flame /tmp/preprocess.prof

To only view profiles for C++ preprocessing and functions it calls, use
`-show_from=jax_sc_embedding::PreprocessSparseDenseMatmulInput` and for
extraction use `-show_from='jax_sc_embedding::ExtractCooTensors'` and so on.
(See https://github.com/google/pprof/blob/main/doc/README.md for more details.)
"""

import concurrent
import sys
from typing import Optional

from absl import app
from absl import flags
import google_benchmark
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import portpicker


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
  for feature_spec in feature_specs:
    table_spec = feature_spec.table_spec
    if not ragged:
      features = np.random.randint(
          table_spec.vocabulary_size,
          size=(num_samples, 16),
          dtype=np.int32,
      )
      all_features.append(features)
    else:
      counts = np.random.randint(1, 32, size=num_samples)
      total_ids = np.sum(counts)
      ids_flat = np.random.randint(
          table_spec.vocabulary_size,
          size=(total_ids,),
          dtype=np.int32,
      )
      split_indices = np.cumsum(counts)[:-1]
      features = np.split(ids_flat, split_indices)
      all_features.append(np.array(features, dtype=object))
  return all_features, None


def generate_sparse_coo_inputs_for_feature_spec(
    feature_specs, num_samples, vocab_size
):
  """Generates random samples for a given feature spec."""
  all_indices_tensors = []
  all_values_tensors = []
  all_dense_shape_tensors = []

  for feature_spec in feature_specs:
    table_spec = feature_spec.table_spec
    counts = np.random.randint(1, 32, size=num_samples)
    total_ids = np.sum(counts)
    values = np.random.randint(
        table_spec.vocabulary_size, size=total_ids, dtype=np.int32
    )
    row_indices = np.repeat(np.arange(num_samples), counts)
    col_indices = np.concatenate([np.arange(c) for c in counts])
    indices = np.stack([row_indices, col_indices], axis=1)

    all_indices_tensors.append(indices.astype(np.int64))
    all_values_tensors.append(values)
    all_dense_shape_tensors.append(
        np.array([num_samples, vocab_size], dtype=np.int64)
    )
  return all_indices_tensors, all_values_tensors, all_dense_shape_tensors


def apply_fdo_stats(
    stats_cc: embedding.SparseDenseMatmulInputStats,
    fdo_headroom: float = 1.0,
    buffer_size_headroom: Optional[float] = None,
):
  """Applies FDO adjustment to benchmark specs from stats.

  Args:
    stats_cc: The C++ SparseDenseMatmulInputStats object.
    fdo_headroom: The headroom to apply to the FDO stats for max_ids and
      max_unique_ids. Defaults to 1.0 because we process the same input
      repeatedly.
    buffer_size_headroom: The headroom to apply to FDO stats for
      required_buffer_size_per_sc. If None, fdo_headroom is used. This may need
      to be larger than fdo_headroom if minibatching is forced, as minibatching
      might increase buffer size requirements.
  """
  stats = embedding.SparseDenseMatmulInputStats.from_cc(stats_cc)
  if buffer_size_headroom is None:
    buffer_size_headroom = fdo_headroom

  for stat_dict in [
      stats.max_ids_per_partition,
      stats.max_unique_ids_per_partition,
  ]:
    for table_name, value in stat_dict.items():
      stat_dict[table_name] = np.array(
          [int(np.ceil(np.max(value) * fdo_headroom))]
      )
  for table_name, value in stats.required_buffer_size_per_sc.items():
    stats.required_buffer_size_per_sc[table_name] = np.array(
        [int(np.ceil(np.max(value) * buffer_size_headroom))]
    )
  assert _GLOBAL_SPECS is not None
  embedding.update_preprocessing_parameters(
      _GLOBAL_SPECS, stats, num_sc_per_device=4
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["ragged"])
@google_benchmark.option.args_product([[False, True]])
@google_benchmark.option.iterations(100)
def preprocess_input_benchmark(state: google_benchmark.State):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  ragged = state.range(0)
  if ragged:
    features, feature_weights = _GLOBAL_RAGGED_FEATURES, _GLOBAL_RAGGED_WEIGHTS
  else:
    features, feature_weights = _GLOBAL_DENSE_FEATURES, _GLOBAL_DENSE_WEIGHTS
  batch_num = 0
  while state:
    if batch_num == 0:
      state.pause_timing()
    *_, stats_cc = pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        features,
        feature_weights,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        batch_number=batch_num,
        allow_id_dropping=batch_num == 0,
    )
    if batch_num == 0:
      apply_fdo_stats(stats_cc)
      state.resume_timing()
    batch_num += 1


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.iterations(100)
def preprocess_input_benchmark_sparse_coo(state: google_benchmark.State):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  batch_num = 0
  while state:
    if batch_num == 0:
      state.pause_timing()
    *_, stats_cc = (
        pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput(
            _GLOBAL_RAGGED_INDICES,
            _GLOBAL_RAGGED_VALUES,
            _GLOBAL_RAGGED_DENSE_SHAPES,
            _GLOBAL_SPECS,
            local_device_count=4,
            global_device_count=16,
            num_sc_per_device=4,
            batch_number=batch_num,
            allow_id_dropping=batch_num == 0,
        )
    )
    if batch_num == 0:
      apply_fdo_stats(stats_cc)
      state.resume_timing()
    batch_num += 1


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["force_minibatching"])
@google_benchmark.option.args_product([[False, True]])
@google_benchmark.option.iterations(25)
def preprocess_input_benchmark_minibatching_enabled(
    state: google_benchmark.State,
):
  """Benchmark for preprocessing input for sparse-dense matmul with minibatching enabled."""
  force_minibatching = state.range(0)
  num_hosts = 4
  ports = [portpicker.pick_unused_port() for _ in range(num_hosts)]
  all_reduce_interfaces = []
  for host_id in range(num_hosts):
    peer_addresses = [
        f"localhost:{ports[i]}" for i in range(num_hosts) if i != host_id
    ]
    all_reduce_interfaces.append(
        embedding.get_all_reduce_interface(
            host_id=host_id,
            host_count=num_hosts,
            peer_addresses=peer_addresses,
            minibatching_port=ports[host_id],
        )
    )

  def worker(host_id: int, batch_number: int):
    return pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
        _GLOBAL_RAGGED_FEATURES,
        _GLOBAL_RAGGED_WEIGHTS,
        _GLOBAL_SPECS,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        enable_minibatching=True,
        all_reduce_interface=all_reduce_interfaces[host_id],
        batch_number=batch_number,
        allow_id_dropping=batch_number == 0,
    )

  with concurrent.futures.ThreadPoolExecutor(max_workers=num_hosts) as executor:
    batch_num = 0
    while state:
      state.items_processed = (
          num_hosts  # Since we use the same CPU for all hosts.
      )
      if batch_num == 0:
        state.pause_timing()
      futures = [
          (host_id, executor.submit(worker, host_id, batch_num))
          for host_id in range(num_hosts)
      ]
      num_minibatches = None
      stats_cc = None
      for host_id, future in futures:
        try:
          # All inputs are the same, so just use the last updated stats.
          *_, num_minibatches, stats_cc = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
          print(f"Host {host_id} timed out.")
          return
      if batch_num == 0:
        assert stats_cc is not None
        if force_minibatching:
          apply_fdo_stats(stats_cc, fdo_headroom=0.5, buffer_size_headroom=1.0)
        else:
          apply_fdo_stats(stats_cc, fdo_headroom=1.0)
        state.resume_timing()
      else:
        if force_minibatching:
          # Make sure we are benchmarking multiple minibatches (guaranteed by
          # initial seed).
          assert num_minibatches >= 5, num_minibatches
        else:
          # Make sure we are benchmarking only one minibatch.
          assert num_minibatches == 1, num_minibatches
      batch_num += 1


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
  np.random.seed(0)
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
