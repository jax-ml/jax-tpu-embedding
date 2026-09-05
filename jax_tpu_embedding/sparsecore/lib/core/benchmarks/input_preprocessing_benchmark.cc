// Copyright 2024 The JAX SC Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "absl/random/distributions.h"  // from @com_google_absl
#include "absl/random/random.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"

namespace jax_sc_embedding {

namespace {

std::string CombinerToString(RowCombiner combiner) {
  switch (combiner) {
    case RowCombiner::kSum:
      return "kSum";
    case RowCombiner::kMean:
      return "kMean";
    case RowCombiner::kSqrtn:
      return "kSqrtn";
  }
}

template <typename Derived>
void LogStats(const Eigen::MatrixBase<Derived>& data, absl::string_view name) {
  if (data.size() == 0) {
    fprintf(stderr, "%s: data is empty.\n", name.data());
    return;
  }
  double mean = data.template cast<double>().mean();
  double stdev =
      std::sqrt((data.template cast<double>().array() - mean).square().sum() /
                data.size());
  fprintf(stderr, "%s: mean=%f, stdev=%f, min=%d, max=%d\n", name.data(), mean,
          stdev, data.minCoeff(), data.maxCoeff());
}

// The following parameters are derived from an empirical analysis of a
// production model workload, and tuned to benchmark
// SparseCore input processing performance under realistic conditions, including
// vocabulary size, average number of IDs per sample, and ID distribution skew.
// FDO parameters for reference: max_ids_per_partition=4456,
// max_unique_ids_per_partition=792.
constexpr int kNumScPerDevice = 4;
constexpr int kGlobalDeviceCount = 128;
constexpr int kBatchSizePerSc = 16384;
constexpr int kSeed = 31337;

enum class WorkloadProfile {
  // Sparse vocabulary with large key space (100M), average valency ~27,
  // Zipf skewed distribution. Key collision / deduplication rate is ~0%.
  kLargeVocab = 0,
  // High multivalency with compact vocabulary (33 keys), high valency (~300).
  // Exhibits high duplicate rates (~85-90%), exercising combiner-specific
  // deduplication fast paths.
  kHighMultivalency = 1,
  // Moderate multivalency with medium vocabulary (1,000 keys), average valency
  // ~50. Exhibits moderate deduplication rates (~40-50%).
  kModerateMultivalency = 2,
};

std::string WorkloadProfileToString(WorkloadProfile profile) {
  switch (profile) {
    case WorkloadProfile::kLargeVocab:
      return "LargeVocab";
    case WorkloadProfile::kHighMultivalency:
      return "HighMultivalency";
    case WorkloadProfile::kModerateMultivalency:
      return "ModerateMultivalency";
  }
}

struct WorkloadConfig {
  int vocab_size;
  float lognormal_mean;
  float lognormal_stddev;
  double zipf_q;
  double skew_probability;
};

WorkloadConfig GetWorkloadConfig(WorkloadProfile profile) {
  switch (profile) {
    case WorkloadProfile::kLargeVocab:
      return {
          .vocab_size = 100000000,
          .lognormal_mean = 1.5f,
          .lognormal_stddev = 1.9f,
          .zipf_q = 1.00278,
          .skew_probability = 0.42430,
      };
    case WorkloadProfile::kHighMultivalency:
      return {
          .vocab_size = 33,
          .lognormal_mean = 5.7f,
          .lognormal_stddev = 0.2f,
          .zipf_q = 1.00278,
          .skew_probability = 0.5,
      };
    case WorkloadProfile::kModerateMultivalency:
      return {
          .vocab_size = 1000,
          .lognormal_mean = 3.9f,
          .lognormal_stddev = 0.3f,
          .zipf_q = 1.00278,
          .skew_probability = 0.5,
      };
  }
}

std::vector<int> GenerateEmbeddingIdsForRow(absl::BitGen& gen,
                                            const WorkloadConfig& config) {
  std::vector<int> ids_out;
  int sample_size = static_cast<int>(std::round(std::exp(absl::Gaussian<float>(
      gen, config.lognormal_mean, config.lognormal_stddev))));
  sample_size = std::max(1, sample_size);
  ids_out.reserve(sample_size);
  for (int i = 0; i < sample_size; ++i) {
    int embedding_id;
    if (config.vocab_size > 1 &&
        absl::Bernoulli(gen, config.skew_probability)) {
      embedding_id = absl::Zipf<int>(gen, config.vocab_size - 1, config.zipf_q);
    } else {
      embedding_id = absl::Uniform<int>(gen, 0, config.vocab_size);
    }
    ids_out.push_back(embedding_id);
  }
  return ids_out;
}

std::vector<std::unique_ptr<AbstractInputBatch>>
GenerateSkewedRaggedTensorInputBatches(int num_sc_per_device,
                                       int batch_size_per_sc,
                                       const WorkloadConfig& config,
                                       int num_features) {
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches;
  input_batches.reserve(num_features);
  absl::BitGen gen(std::seed_seq{kSeed});  // seed for reproducibility

  const int batch_size_for_device = num_sc_per_device * batch_size_per_sc;

  for (int f = 0; f < num_features; ++f) {
    std::vector<int64_t> values;
    std::vector<int32_t> row_splits;
    row_splits.push_back(0);

    for (int row = 0; row < batch_size_for_device; ++row) {
      std::vector<int> embedding_ids = GenerateEmbeddingIdsForRow(gen, config);
      for (int embedding_id : embedding_ids) {
        values.push_back(embedding_id);
      }
      row_splits.push_back(values.size());
    }

    input_batches.push_back(
        std::make_unique<RaggedTensorInputBatchWithOwnedData<int64_t, int32_t>>(
            std::move(values), std::move(row_splits)));
  }
  return input_batches;
}

void BM_ExtractCooTensors(benchmark::State& state) {
  const int num_features = state.range(0);
  const RowCombiner combiner = static_cast<RowCombiner>(state.range(1));
  const WorkloadProfile profile = static_cast<WorkloadProfile>(state.range(2));
  state.SetLabel(absl::StrCat(CombinerToString(combiner), "/",
                              WorkloadProfileToString(profile)));
  const WorkloadConfig workload_config = GetWorkloadConfig(profile);
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      GenerateSkewedRaggedTensorInputBatches(kNumScPerDevice, kBatchSizePerSc,
                                             workload_config, num_features);

  std::vector<FeatureMetadataInStack> stacked_table_metadata;
  stacked_table_metadata.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    stacked_table_metadata.push_back(FeatureMetadataInStack(
        absl::StrCat("table_", i), /*feature_index=*/i,
        /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*row_offset=*/0,
        /*col_offset=*/0,
        /*col_shift=*/0, /*batch_size=*/kBatchSizePerSc,
        /*suggested_coo_buffer_size_per_device=*/std::nullopt,
        /*row_combiner=*/combiner));
  }

  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .allow_id_dropping = false,
      .enable_minibatching = true,
  };

  for (auto s : state) {
    ExtractedCooTensors extracted_coo_tensors =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
            stacked_table_metadata, absl::MakeSpan(input_batches),
            /*local_device_id=*/0, options);
    extracted_coo_tensors.BlockUntilReady();
  }
}
BENCHMARK(BM_ExtractCooTensors)
    // Args: {num_features, combiner, workload_profile}
    ->Args({20, 0, 0})  // kSum, LargeVocab
    ->Args({20, 1, 0})  // kMean, LargeVocab
    ->Args({20, 2, 0})  // kSqrtn, LargeVocab
    ->Args({1, 0, 1})   // kSum, HighMultivalency
    ->Args({1, 1, 1})   // kMean, HighMultivalency
    ->Args({1, 2, 1})   // kSqrtn, HighMultivalency
    ->Args({5, 1, 2})   // kMean, ModerateMultivalency
    ->Threads(8)
    ->UseRealTime();

void BM_SortAndGroup_Phase1(benchmark::State& state) {
  const int num_features = state.range(0);
  const RowCombiner combiner = static_cast<RowCombiner>(state.range(1));
  const WorkloadProfile profile = static_cast<WorkloadProfile>(state.range(2));
  state.SetLabel(absl::StrCat(CombinerToString(combiner), "/",
                              WorkloadProfileToString(profile)));
  const WorkloadConfig workload_config = GetWorkloadConfig(profile);
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      GenerateSkewedRaggedTensorInputBatches(kNumScPerDevice, kBatchSizePerSc,
                                             workload_config, num_features);

  std::vector<FeatureMetadataInStack> stacked_table_metadata_list;
  stacked_table_metadata_list.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    // Set to INT_MAX to avoid ID dropping and observe the actual statistics of
    // the generated data. This doesn't affect performance of grouping itself.
    stacked_table_metadata_list.push_back(FeatureMetadataInStack(
        absl::StrCat("table_", i), /*feature_index=*/i,
        /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*row_offset=*/0,
        /*col_offset=*/0,
        /*col_shift=*/0, /*batch_size=*/kBatchSizePerSc,
        /*suggested_coo_buffer_size_per_device=*/std::nullopt,
        /*row_combiner=*/combiner));
  }

  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .allow_id_dropping = false,
      .enable_minibatching = true,
  };

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_list, absl::MakeSpan(input_batches),
          /*local_device_id=*/0, options);

  bool minibatching_required = false;
  StatsPerHost stats_per_host(
      /*local_device_count=*/1,
      /*global_sc_count=*/kNumScPerDevice * kGlobalDeviceCount,
      /*num_sc_per_device=*/kNumScPerDevice);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);

  if (state.thread_index() == 0) {
    SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
        extracted_coo_tensors, "table_0", stacked_table_metadata_list[0],
        options, stats_per_device, minibatching_required);
    LogStats(stats_per_device.max_ids_per_partition,
             "Max ids per partition across all global SCs");
    LogStats(stats_per_device.max_unique_ids_per_partition,
             "Max unique ids per partition across all global SCs");
  }

  for (auto s : state) {
    SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
        extracted_coo_tensors, "table_0", stacked_table_metadata_list[0],
        options, stats_per_device, minibatching_required);
  }
}
BENCHMARK(BM_SortAndGroup_Phase1)
    // Args: {num_features, combiner, workload_profile}
    ->Args({20, 0, 0})  // kSum, LargeVocab
    ->Args({20, 1, 0})  // kMean, LargeVocab
    ->Args({20, 2, 0})  // kSqrtn, LargeVocab
    ->Args({1, 0, 1})   // kSum, HighMultivalency
    ->Args({1, 1, 1})   // kMean, HighMultivalency
    ->Args({1, 2, 1})   // kSqrtn, HighMultivalency
    ->Args({1, 1, 2})   // kMean, ModerateMultivalency
    ->Threads(8)
    ->UseRealTime();

void BM_FillBuffer(benchmark::State& state) {
  const int num_features = state.range(0);
  const WorkloadProfile profile = static_cast<WorkloadProfile>(state.range(1));
  state.SetLabel(WorkloadProfileToString(profile));
  const WorkloadConfig workload_config = GetWorkloadConfig(profile);
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      GenerateSkewedRaggedTensorInputBatches(kNumScPerDevice, kBatchSizePerSc,
                                             workload_config, num_features);

  std::vector<FeatureMetadataInStack> stacked_table_metadata_list;
  stacked_table_metadata_list.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    stacked_table_metadata_list.push_back(FeatureMetadataInStack(
        absl::StrCat("table_", i), /*feature_index=*/i,
        /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*row_offset=*/0,
        /*col_offset=*/0,
        /*col_shift=*/0, /*batch_size=*/kBatchSizePerSc,
        /*suggested_coo_buffer_size_per_device=*/std::nullopt,
        /*row_combiner=*/RowCombiner::kSum));
  }

  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .allow_id_dropping = false,
      .enable_minibatching = true,
  };

  ExtractedCooTensors extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
          stacked_table_metadata_list, absl::MakeSpan(input_batches),
          /*local_device_id=*/0, options);

  bool minibatching_required = false;
  StatsPerHost stats_per_host(
      /*local_device_count=*/1,
      /*global_sc_count=*/kNumScPerDevice * kGlobalDeviceCount,
      /*num_sc_per_device=*/kNumScPerDevice);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  DevicePartitionedCooTensors grouped_coo_tensors =
      SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
          extracted_coo_tensors, "table_0", stacked_table_metadata_list[0],
          options, stats_per_device, minibatching_required);

  const int coo_buffer_size_for_device =
      stats_per_device.required_buffer_size.sum();
  const int row_pointers_size =
      options.GetRowPointersSizePerBucket() * kNumScPerDevice;

  MatrixXi row_pointers(1, row_pointers_size);
  MatrixXi embedding_ids(1, coo_buffer_size_for_device);
  MatrixXi sample_ids(1, coo_buffer_size_for_device);
  MatrixXf gains(1, coo_buffer_size_for_device);

  CsrArraysPerHost csr_arrays_per_host(row_pointers, embedding_ids, sample_ids,
                                       gains);
  internal::CsrArraysRefPerDevice csr_arrays =
      csr_arrays_per_host.GetCsrArraysRefForDevice(0);
  int dropped_id_count_static_bound = 0;

  for (auto s : state) {
    FillLocalDeviceBuffer(grouped_coo_tensors, kBatchSizePerSc,
                          stats_per_device.required_buffer_size, options,
                          stacked_table_metadata_list[0].name, csr_arrays,
                          dropped_id_count_static_bound);
  }
}

// Buffer filling is independent of the row combiner.
BENCHMARK(BM_FillBuffer)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->UseRealTime();
BENCHMARK(BM_FillBuffer)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->Threads(8)
    ->UseRealTime();

}  // namespace
}  // namespace jax_sc_embedding
