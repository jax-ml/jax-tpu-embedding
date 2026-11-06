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
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "jax_tpu_embedding/sparsecore/lib/core/ragged_tensor_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla

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
constexpr int kScSimdWidth = 8;
constexpr int kHbmWordSizeIn4b = 8;
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
  // Sparse 64-bit vocabulary exceeding UINT32_MAX (5B keys), exercising the
  // 64-bit global column ID path while emitting int32 shard-local IDs.
  kLargeVocab64 = 3,
};

std::string WorkloadProfileToString(WorkloadProfile profile) {
  switch (profile) {
    case WorkloadProfile::kLargeVocab:
      return "LargeVocab";
    case WorkloadProfile::kHighMultivalency:
      return "HighMultivalency";
    case WorkloadProfile::kModerateMultivalency:
      return "ModerateMultivalency";
    case WorkloadProfile::kLargeVocab64:
      return "LargeVocab64";
  }
}

struct WorkloadConfig {
  int64_t vocab_size;
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
    case WorkloadProfile::kLargeVocab64:
      return {
          .vocab_size = 5'000'000'000LL,
          .lognormal_mean = 1.5f,
          .lognormal_stddev = 1.9f,
          .zipf_q = 1.00278,
          .skew_probability = 0.42430,
      };
  }
}

std::vector<int64_t> GenerateEmbeddingIdsForRow(absl::BitGen& gen,
                                                const WorkloadConfig& config) {
  std::vector<int64_t> ids_out;
  int sample_size = static_cast<int>(std::round(std::exp(absl::Gaussian<float>(
      gen, config.lognormal_mean, config.lognormal_stddev))));
  sample_size = std::max(1, sample_size);
  ids_out.reserve(sample_size);
  const bool fits_int32 = config.vocab_size <= std::numeric_limits<int>::max();
  for (int i = 0; i < sample_size; ++i) {
    int64_t embedding_id;
    if (fits_int32) {
      const int vocab_size_32 = static_cast<int>(config.vocab_size);
      if (vocab_size_32 > 1 && absl::Bernoulli(gen, config.skew_probability)) {
        embedding_id = absl::Zipf<int>(gen, vocab_size_32 - 1, config.zipf_q);
      } else {
        embedding_id = absl::Uniform<int>(gen, 0, vocab_size_32);
      }
    } else {
      if (absl::Bernoulli(gen, config.skew_probability)) {
        embedding_id =
            absl::Zipf<int64_t>(gen, config.vocab_size - 1, config.zipf_q);
      } else {
        embedding_id =
            absl::Uniform<int64_t>(gen, int64_t{0}, config.vocab_size);
      }
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
      std::vector<int64_t> embedding_ids =
          GenerateEmbeddingIdsForRow(gen, config);
      for (int64_t embedding_id : embedding_ids) {
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

std::vector<FeatureMetadataInStack> BuildStackedTableMetadata(
    int num_features, RowCombiner combiner, int64_t vocab_size) {
  const int64_t num_scs = int64_t{kGlobalDeviceCount} * kNumScPerDevice;
  const int64_t rows_per_sc =
      std::max<int64_t>(1, (vocab_size + num_scs - 1) / num_scs);
  const std::optional<int64_t> max_col_id =
      vocab_size > std::numeric_limits<int32_t>::max()
          ? std::optional<int64_t>(rows_per_sc - 1)
          : std::nullopt;
  std::vector<FeatureMetadataInStack> metadata;
  metadata.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    // Set partition limits to INT_MAX to avoid ID dropping and observe the
    // actual statistics of the generated data.
    metadata.push_back(FeatureMetadataInStack(
        absl::StrCat("table_", i), /*feature_index=*/i,
        /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
        /*row_offset=*/0,
        /*col_offset=*/0,
        /*col_shift=*/0, /*batch_size=*/kBatchSizePerSc,
        /*suggested_coo_buffer_size_per_device=*/std::nullopt,
        /*row_combiner=*/combiner,
        /*max_col_id=*/max_col_id));
  }
  return metadata;
}

template <typename ColIdT>
void RunExtractBenchmark(
    benchmark::State& state,
    absl::Span<const FeatureMetadataInStack> stacked_table_metadata,
    absl::Span<const std::unique_ptr<AbstractInputBatch>> input_batches,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  for (auto s : state) {
    ExtractedCooTensorsT<ColIdT> extracted_coo_tensors =
        internal::ExtractCooTensorsForAllFeaturesPerLocalDevice<ColIdT>(
            stacked_table_metadata, input_batches,
            /*local_device_id=*/0, options, /*has_variable_weights=*/false);
    extracted_coo_tensors.BlockUntilReady();
  }
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
  const std::vector<FeatureMetadataInStack> stacked_table_metadata =
      BuildStackedTableMetadata(num_features, combiner,
                                workload_config.vocab_size);

  const PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .sc_simd_width = kScSimdWidth,
      .hbm_word_size_in_4b = kHbmWordSizeIn4b,
      .allow_id_dropping = false,
      .minibatching_mode = MinibatchingMode::kHost,
  };

  if (workload_config.vocab_size > std::numeric_limits<int32_t>::max()) {
    RunExtractBenchmark<int64_t>(state, stacked_table_metadata,
                                 absl::MakeSpan(input_batches), options);
  } else {
    RunExtractBenchmark<int32_t>(state, stacked_table_metadata,
                                 absl::MakeSpan(input_batches), options);
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
    ->Args({20, 0, 3})  // kSum, LargeVocab64
    ->Args({20, 1, 3})  // kMean, LargeVocab64
    ->Args({20, 2, 3})  // kSqrtn, LargeVocab64
    ->Threads(8)
    ->UseRealTime();

template <typename ColIdT>
void RunSortAndGroupBenchmark(
    benchmark::State& state,
    absl::Span<const FeatureMetadataInStack> stacked_table_metadata_list,
    absl::Span<const std::unique_ptr<AbstractInputBatch>> input_batches,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  ExtractedCooTensorsT<ColIdT> extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice<ColIdT>(
          stacked_table_metadata_list, input_batches,
          /*local_device_id=*/0, options, /*has_variable_weights=*/false);

  // Sorting permutes the extracted keys in place, so keep a pristine copy and
  // restore it before every iteration. Otherwise, every iteration after the
  // first would sort already-sorted keys and understate the sorting cost.
  std::vector<std::vector<uint64_t>> unsorted_keys_per_sc;
  unsorted_keys_per_sc.reserve(extracted_coo_tensors.per_sc_tensors_av.size());
  for (const auto& sc_tensor_av : extracted_coo_tensors.per_sc_tensors_av) {
    tsl::BlockUntilReady(sc_tensor_av);
    unsorted_keys_per_sc.push_back(sc_tensor_av->keys);
  }

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
    state.PauseTiming();
    for (size_t sc_id = 0; sc_id < unsorted_keys_per_sc.size(); ++sc_id) {
      extracted_coo_tensors.per_sc_tensors_av[sc_id]->keys =
          unsorted_keys_per_sc[sc_id];
    }
    state.ResumeTiming();
    SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
        extracted_coo_tensors, "table_0", stacked_table_metadata_list[0],
        options, stats_per_device, minibatching_required);
  }
}

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
  const std::vector<FeatureMetadataInStack> stacked_table_metadata_list =
      BuildStackedTableMetadata(num_features, combiner,
                                workload_config.vocab_size);

  const PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .sc_simd_width = kScSimdWidth,
      .hbm_word_size_in_4b = kHbmWordSizeIn4b,
      .allow_id_dropping = false,
      .minibatching_mode = MinibatchingMode::kHost,
  };

  if (workload_config.vocab_size > std::numeric_limits<int32_t>::max()) {
    RunSortAndGroupBenchmark<int64_t>(state, stacked_table_metadata_list,
                                      absl::MakeSpan(input_batches), options);
  } else {
    RunSortAndGroupBenchmark<int32_t>(state, stacked_table_metadata_list,
                                      absl::MakeSpan(input_batches), options);
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
    ->Args({20, 0, 3})  // kSum, LargeVocab64
    ->Args({20, 1, 3})  // kMean, LargeVocab64
    ->Args({20, 2, 3})  // kSqrtn, LargeVocab64
    ->Threads(8)
    ->UseRealTime();

template <typename ColIdT, bool kEnableMegacoreCsr>
void RunFillBufferBenchmark(
    benchmark::State& state, const WorkloadConfig& workload_config,
    absl::Span<const FeatureMetadataInStack> stacked_table_metadata_list,
    absl::Span<const std::unique_ptr<AbstractInputBatch>> input_batches,
    const PreprocessSparseDenseMatmulInputOptions& options) {
  ExtractedCooTensorsT<ColIdT> extracted_coo_tensors =
      internal::ExtractCooTensorsForAllFeaturesPerLocalDevice<ColIdT>(
          stacked_table_metadata_list, input_batches,
          /*local_device_id=*/0, options, /*has_variable_weights=*/false);

  bool minibatching_required = false;
  StatsPerHost stats_per_host(
      /*local_device_count=*/1,
      /*global_sc_count=*/kNumScPerDevice * kGlobalDeviceCount,
      /*num_sc_per_device=*/kNumScPerDevice);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);
  DevicePartitionedCooTensorsT<CooFormatT<ColIdT>> grouped_coo_tensors =
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

  std::optional<MegacoreCsrOptions> megacore_csr;
  if constexpr (kEnableMegacoreCsr) {
    const int64_t num_scs = int64_t{kGlobalDeviceCount} * kNumScPerDevice;
    megacore_csr = MegacoreCsrOptions{
        .rows_per_sc = std::max(
            1, static_cast<int>(workload_config.vocab_size / num_scs))};
  }

  for (auto s : state) {
    FillLocalDeviceBuffer(grouped_coo_tensors, kBatchSizePerSc,
                          stats_per_device.required_buffer_size, options,
                          stacked_table_metadata_list[0].name, megacore_csr,
                          csr_arrays, dropped_id_count_static_bound);
  }
}

template <bool kEnableMegacoreCsr>
void BM_FillBufferImpl(benchmark::State& state,
                       MinibatchingMode minibatching_mode) {
  const int num_features = state.range(0);
  const WorkloadProfile profile = static_cast<WorkloadProfile>(state.range(1));
  state.SetLabel(WorkloadProfileToString(profile));
  const WorkloadConfig workload_config = GetWorkloadConfig(profile);
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      GenerateSkewedRaggedTensorInputBatches(kNumScPerDevice, kBatchSizePerSc,
                                             workload_config, num_features);
  const std::vector<FeatureMetadataInStack> stacked_table_metadata_list =
      BuildStackedTableMetadata(num_features, RowCombiner::kSum,
                                workload_config.vocab_size);

  const PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .sc_simd_width = kScSimdWidth,
      .hbm_word_size_in_4b = kHbmWordSizeIn4b,
      .allow_id_dropping = false,
      .minibatching_mode = minibatching_mode,
  };

  if (workload_config.vocab_size > std::numeric_limits<int32_t>::max()) {
    RunFillBufferBenchmark<int64_t, kEnableMegacoreCsr>(
        state, workload_config, stacked_table_metadata_list,
        absl::MakeSpan(input_batches), options);
  } else {
    RunFillBufferBenchmark<int32_t, kEnableMegacoreCsr>(
        state, workload_config, stacked_table_metadata_list,
        absl::MakeSpan(input_batches), options);
  }
}

void BM_FillBuffer(benchmark::State& state) {
  BM_FillBufferImpl</*kEnableMegacoreCsr=*/false>(state,
                                                  MinibatchingMode::kHost);
}

void BM_FillBuffer_Megacore(benchmark::State& state) {
  // Megacore CSR only supports device minibatching.
  BM_FillBufferImpl</*kEnableMegacoreCsr=*/true>(state,
                                                 MinibatchingMode::kDevice);
}

// Buffer filling is independent of the row combiner.
BENCHMARK(BM_FillBuffer)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->Args({20, 3})  // LargeVocab64
    ->UseRealTime();
BENCHMARK(BM_FillBuffer)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->Args({20, 3})  // LargeVocab64
    ->Threads(8)
    ->UseRealTime();
BENCHMARK(BM_FillBuffer_Megacore)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->Args({20, 3})  // LargeVocab64
    ->UseRealTime();
BENCHMARK(BM_FillBuffer_Megacore)
    // Args: {num_features, profile}
    ->Args({20, 0})  // LargeVocab
    ->Args({1, 1})   // HighMultivalency
    ->Args({20, 3})  // LargeVocab64
    ->Threads(8)
    ->UseRealTime();

}  // namespace
}  // namespace jax_sc_embedding
