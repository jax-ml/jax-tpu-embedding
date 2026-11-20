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

Locally:
bazel run -c opt --dynamic_mode=off --copt=-gmlt \
 //jax_tpu_embedding/sparsecore/lib/nn/tests:preprocess_input_benchmarks \
 -- --benchmark_filter=all --cpu_profile=/tmp/preprocess.prof

The --benchmark_filter flag uses a regex to select benchmarks. For parameterized
benchmarks, the name is typically formatted as:
Boolean parameters are often represented as 0 for False and 1 for True.
`[benchmark_name]/[param1]:[value1]`.

For example, to run only the `sparse_coo` benchmarks:
`--benchmark_filter=preprocess_sparse_coo`

To run only the unweighted single-table ragged benchmark:
`--benchmark_filter='preprocess_numpy_ragged/weighted:0/stacked:0'`

To upload the profile to pprof:
pprof -flame /tmp/preprocess.prof

To only view profiles for C++ preprocessing and functions it calls, use
`-show_from=jax_sc_embedding::PreprocessSparseDenseMatmulInput` and for
extraction use `-show_from='jax_sc_embedding::ExtractCooTensors'` and so on.
(See https://github.com/google/pprof/blob/main/doc/README.md for more details.)
"""

from __future__ import annotations

from collections.abc import Sequence
import concurrent.futures
import functools
import sys

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


def generate_feature_specs(
    num_features: int,
    num_samples: int,
    *,
    local_device_count: int = 4,
    global_device_count: int = 16,
    num_sc_per_device: int = 4,
    vocab_size: int = 1024,
    combiner: str = "sum",
    stacked: bool = False,
) -> list[embedding_spec.FeatureSpec]:
  """Generates feature specs for the given number of features."""
  global_num_samples = num_samples * global_device_count // local_device_count
  feature_specs = []
  for i in range(num_features):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size,
        embedding_dim=16,
        initializer=lambda: np.zeros((1, 16), dtype=np.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner=combiner,
        name="table_{}".format(i),
        max_ids_per_partition=1024,
        max_unique_ids_per_partition=1024,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[global_num_samples, 1],
        output_shape=[global_num_samples, 16],
        name="feature_spec_{}".format(i),
    )
    feature_specs.append(feature_spec)
  if stacked:
    limit = 1024 * num_features
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        stack_to_max_ids_per_partition=lambda _name, _batch: limit,
        stack_to_max_unique_ids_per_partition=lambda _name, _batch: limit,
        activation_mem_bytes_limit=64 * 1024 * 1024,
    )
  else:
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
    )
  return feature_specs


@functools.cache
def generate_samples(
    num_features: int,
    num_samples: int,
    *,
    vocab_size: int = 1024,
    ragged: bool = False,
    weighted: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
  """Generates random samples for the given dimensions."""
  np.random.seed(0)
  all_features = []
  all_weights = [] if weighted else None
  dtype = np.int64 if vocab_size > np.iinfo(np.int32).max else np.int32
  for _ in range(num_features):
    if not ragged:
      features = np.random.randint(
          vocab_size,
          size=(num_samples, 16),
          dtype=dtype,
      )
      all_features.append(features)
    else:
      counts = np.random.randint(1, 32, size=num_samples)
      total_ids = int(np.sum(counts))
      ids_flat = np.random.randint(
          vocab_size,
          size=(total_ids,),
          dtype=np.int32,
      )
      split_indices = np.cumsum(counts)[:-1]
      features = np.split(ids_flat, split_indices)
      all_features.append(np.array(features, dtype=object))
      if all_weights is not None:
        weights_flat = np.random.uniform(0.1, 1.0, size=total_ids).astype(
            np.float32
        )
        all_weights.append(
            np.array(np.split(weights_flat, split_indices), dtype=object)
        )
  return all_features, all_weights


@functools.cache
def generate_sparse_coo_inputs(
    num_features: int, num_samples: int, vocab_size: int = 1024
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
  """Generates random sparse COO inputs for the given dimensions."""
  np.random.seed(0)
  all_indices_tensors = []
  all_values_tensors = []
  all_dense_shape_tensors = []

  for _ in range(num_features):
    counts = np.random.randint(1, 32, size=num_samples)
    total_ids = int(np.sum(counts))
    values = np.random.randint(vocab_size, size=total_ids, dtype=np.int32)
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
    feature_specs: Sequence[embedding_spec.FeatureSpec],
    stats_cc: pybind_input_preprocessing.SparseDenseMatmulInputStats,
    fdo_headroom: float = 1.0,
    buffer_size_headroom: float | None = None,
):
  """Applies FDO adjustment to benchmark specs from stats.

  Args:
    feature_specs: Feature specs to update in place.
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
  embedding.update_preprocessing_parameters(
      feature_specs, stats, num_sc_per_device=4
  )


def _run_numpy_benchmark(
    state: google_benchmark.State,
    features: list[np.ndarray],
    feature_weights: list[np.ndarray] | None,
    feature_specs: list[embedding_spec.FeatureSpec],
) -> None:
  """Runs the benchmark loop for NumPy input preprocessing."""
  batch_num = 0
  while state:
    if batch_num == 0:
      state.pause_timing()
    *_, stats_cc = (
        pybind_input_preprocessing.preprocess_sparse_dense_matmul_input(
            features,
            feature_weights,
            feature_specs,
            local_device_count=4,
            global_device_count=16,
            num_sc_per_device=4,
            sc_simd_width=8,
            hbm_word_size_in_4b=8,
            batch_number=batch_num,
            allow_id_dropping=batch_num == 0,
        )
    )
    if batch_num == 0:
      apply_fdo_stats(feature_specs, stats_cc)
      state.resume_timing()
    batch_num += 1


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["vocab", "combiner"])
@google_benchmark.option.args([1024, 0])
@google_benchmark.option.args([1024, 1])
@google_benchmark.option.args([1024, 2])
@google_benchmark.option.args([5_000_000_000, 0])
@google_benchmark.option.iterations(100)
def preprocess_numpy_dense(state: google_benchmark.State):
  """Benchmark for preprocessing dense NumPy inputs."""
  vocab_size = state.range(0)
  combiner = ("sum", "mean", "sqrtn")[state.range(1)]
  state.set_label(combiner)
  specs = generate_feature_specs(
      _NUM_FEATURES.value,
      _NUM_SAMPLES.value,
      vocab_size=vocab_size,
      combiner=combiner,
  )
  features, feature_weights = generate_samples(
      _NUM_FEATURES.value, _NUM_SAMPLES.value, vocab_size=vocab_size
  )

  _run_numpy_benchmark(state, features, feature_weights, specs)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["weighted", "stacked"])
@google_benchmark.option.args([0, 0])
@google_benchmark.option.args([1, 0])
@google_benchmark.option.args([0, 1])
@google_benchmark.option.iterations(100)
def preprocess_numpy_ragged(state: google_benchmark.State):
  """Benchmark for preprocessing ragged NumPy inputs."""
  weighted, stacked = bool(state.range(0)), bool(state.range(1))
  specs = generate_feature_specs(
      _NUM_FEATURES.value, _NUM_SAMPLES.value, stacked=stacked
  )
  features, feature_weights = generate_samples(
      _NUM_FEATURES.value,
      _NUM_SAMPLES.value,
      ragged=True,
      weighted=weighted,
  )

  _run_numpy_benchmark(state, features, feature_weights, specs)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["device_minibatching"])
@google_benchmark.option.args_product([[False, True]])
@google_benchmark.option.iterations(100)
def preprocess_sparse_coo(state: google_benchmark.State):
  """Benchmark for preprocessing input for sparse-dense matmul."""
  device_minibatching = bool(state.range(0))
  minibatching_mode = (
      pybind_input_preprocessing.MinibatchingMode.DEVICE
      if device_minibatching
      else pybind_input_preprocessing.MinibatchingMode.DISABLED
  )
  specs = generate_feature_specs(_NUM_FEATURES.value, _NUM_SAMPLES.value)
  indices, values, dense_shapes = generate_sparse_coo_inputs(
      _NUM_FEATURES.value, _NUM_SAMPLES.value
  )
  batch_num = 0
  while state:
    if batch_num == 0:
      state.pause_timing()
    *_, stats_cc = (
        pybind_input_preprocessing.preprocess_sparse_dense_matmul_sparse_coo_input(
            indices,
            values,
            dense_shapes,
            specs,
            local_device_count=4,
            global_device_count=16,
            num_sc_per_device=4,
            sc_simd_width=8,
            hbm_word_size_in_4b=8,
            minibatching_mode=minibatching_mode,
            batch_number=batch_num,
            allow_id_dropping=device_minibatching or batch_num == 0,
        )
    )
    if batch_num == 0:
      apply_fdo_stats(specs, stats_cc)
      state.resume_timing()
    batch_num += 1


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
@google_benchmark.option.arg_names(["force_minibatching"])
@google_benchmark.option.args_product([[False, True]])
@google_benchmark.option.iterations(25)
def preprocess_minibatching(
    state: google_benchmark.State,
):
  """Benchmark for preprocessing input for sparse-dense matmul with minibatching enabled."""
  force_minibatching = state.range(0)
  specs = generate_feature_specs(_NUM_FEATURES.value, _NUM_SAMPLES.value)
  indices, values, dense_shapes = generate_sparse_coo_inputs(
      _NUM_FEATURES.value, _NUM_SAMPLES.value
  )
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
    return pybind_input_preprocessing.preprocess_sparse_dense_matmul_sparse_coo_input(
        indices,
        values,
        dense_shapes,
        specs,
        local_device_count=4,
        global_device_count=16,
        num_sc_per_device=4,
        sc_simd_width=8,
        hbm_word_size_in_4b=8,
        minibatching_mode=pybind_input_preprocessing.MinibatchingMode.HOST,
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
          apply_fdo_stats(
              specs, stats_cc, fdo_headroom=0.5, buffer_size_headroom=1.0
          )
        else:
          apply_fdo_stats(specs, stats_cc, fdo_headroom=1.0)
        state.resume_timing()
      else:
        assert num_minibatches is not None
        if force_minibatching:
          # Make sure we are benchmarking multiple minibatches (guaranteed by
          # initial seed).
          assert num_minibatches >= 5, num_minibatches
        else:
          # Make sure we are benchmarking only one minibatch.
          assert num_minibatches == 1, num_minibatches
      batch_num += 1


def main(_):
  google_benchmark.run_benchmarks()


if __name__ == "__main__":
  sys.argv = google_benchmark.initialize(sys.argv)
  app.run(main)
