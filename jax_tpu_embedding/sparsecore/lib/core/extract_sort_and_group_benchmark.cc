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
constexpr int kVocabSize = 100000000;
constexpr int kBatchSizePerSc = 16384;
constexpr int kSeed = 31337;
constexpr float kLognormalMean = 1.5f;
constexpr float kLognormalStddev = 1.9f;
constexpr double kZipfQ = 1.00278;
constexpr double kSkewProbability = 0.42430;

std::vector<int> GenerateEmbeddingIdsForRow(absl::BitGen& gen, int vocab_size) {
  std::vector<int> ids_out;
  int sample_size = static_cast<int>(std::round(
      std::exp(absl::Gaussian<float>(gen, kLognormalMean, kLognormalStddev))));
  ids_out.reserve(sample_size);
  for (int i = 0; i < sample_size; ++i) {
    int embedding_id;
    if (absl::Bernoulli(gen, kSkewProbability)) {
      embedding_id = absl::Zipf<int>(gen, vocab_size - 1, kZipfQ);
    } else {
      embedding_id = absl::Uniform<int>(gen, 0, vocab_size);
    }
    ids_out.push_back(embedding_id);
  }
  return ids_out;
}

ExtractedCooTensors GenerateSkewedCooTensors(int num_sc_per_device,
                                             int batch_size_per_sc,
                                             int vocab_size) {
  const int batch_size_for_device = num_sc_per_device * batch_size_per_sc;

  absl::BitGen gen(std::seed_seq{kSeed});  // seed for reproducibility

  ExtractedCooTensors extracted_coo_tensors(num_sc_per_device,
                                            batch_size_for_device);

  // For each sample in the batch:
  // 1. Draw a sample size from a Lognormal distribution.
  // 2. Draw `sample_size` IDs. Each ID is drawn from a Zipf distribution
  //    with probability `kSkewProbability`, or uniformly from [0, kVocabSize)
  //    otherwise.
  for (int row = 0; row < batch_size_for_device; ++row) {
    std::vector<int> embedding_ids =
        GenerateEmbeddingIdsForRow(gen, vocab_size);
    int sc_id = row / batch_size_per_sc;
    extracted_coo_tensors.coo_tensors_per_sc[sc_id] += embedding_ids.size();
    for (int embedding_id : embedding_ids) {
      extracted_coo_tensors.coo_tensors.push_back(
          CooFormat(row, embedding_id, 1.0));
    }
  }
  return extracted_coo_tensors;
}

std::vector<std::unique_ptr<AbstractInputBatch>>
GenerateSkewedRaggedTensorInputBatches(int num_sc_per_device,
                                       int batch_size_per_sc, int vocab_size,
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
      std::vector<int> embedding_ids =
          GenerateEmbeddingIdsForRow(gen, vocab_size);
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
  state.SetLabel(CombinerToString(combiner));
  std::vector<std::unique_ptr<AbstractInputBatch>> input_batches =
      GenerateSkewedRaggedTensorInputBatches(kNumScPerDevice, kBatchSizePerSc,
                                             kVocabSize, num_features);

  std::vector<StackedTableMetadata> stacked_table_metadata;
  stacked_table_metadata.reserve(num_features);
  for (int i = 0; i < num_features; ++i) {
    stacked_table_metadata.push_back(StackedTableMetadata(
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
      .feature_stacking_strategy = FeatureStackingStrategy::kSplitThenStack,
  };

  for (auto s : state) {
    internal::ExtractCooTensorsForAllFeaturesPerLocalDevice(
        stacked_table_metadata, absl::MakeSpan(input_batches),
        /*local_device_id=*/0, options);
  }
}
BENCHMARK(BM_ExtractCooTensors)
    // Args: {num_features, combiner}
    ->Args({4, 0})  // kSum
    ->Args({4, 1})  // kMean
    ->Args({4, 2})  // kSqrtn
    ->Threads(8)
    ->UseRealTime();

void BM_SortAndGroup_Phase1(benchmark::State& state) {
  const RowCombiner combiner = static_cast<RowCombiner>(state.range(0));
  ExtractedCooTensors extracted_coo_tensors =
      GenerateSkewedCooTensors(kNumScPerDevice, kBatchSizePerSc, kVocabSize);
  state.SetLabel(CombinerToString(combiner));

  // Set to INT_MAX to avoid ID dropping and observe the actual statistics of
  // the generated data. This doesn't affect performance of grouping itself.
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0,
      /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
      /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
      /*row_offset=*/0,
      /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0,
      /*suggested_coo_buffer_size_per_device=*/std::nullopt,
      /*row_combiner=*/combiner);
  PreprocessSparseDenseMatmulInputOptions options = {
      .local_device_count = 1,
      .global_device_count = kGlobalDeviceCount,
      .num_sc_per_device = kNumScPerDevice,
      .allow_id_dropping = false,
  };
  bool minibatching_required = false;
  StatsPerHost stats_per_host(
      /*local_device_count=*/1,
      /*global_sc_count=*/kNumScPerDevice * kGlobalDeviceCount,
      /*num_sc_per_device=*/kNumScPerDevice);
  internal::StatsPerDevice stats_per_device =
      stats_per_host.GetStatsPerDevice(0);

  if (state.thread_index() == 0) {
    SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
        extracted_coo_tensors, stacked_table_metadata, options,
        stats_per_device, minibatching_required);
    LogStats(stats_per_device.max_ids_per_partition,
             "Max ids per partition across all global SCs");
    LogStats(stats_per_device.max_unique_ids_per_partition,
             "Max unique ids per partition across all global SCs");
  }

  for (auto s : state) {
    SortAndGroupCooTensorsPerLocalDevice</*kHasVariableWeights=*/false>(
        extracted_coo_tensors, stacked_table_metadata, options,
        stats_per_device, minibatching_required);
  }
}
BENCHMARK(BM_SortAndGroup_Phase1)
    ->Arg(0)  // kSum
    ->Arg(1)  // kMean
    ->Arg(2)  // kSqrtn
    ->Threads(8)
    ->UseRealTime();

}  // namespace
}  // namespace jax_sc_embedding
