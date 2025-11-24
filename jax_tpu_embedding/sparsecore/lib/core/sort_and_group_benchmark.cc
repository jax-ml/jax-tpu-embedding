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
#include <cstdio>
#include <limits>
#include <random>

#include "benchmark/benchmark.h"
#include "absl/random/distributions.h"  // from @com_google_absl
#include "absl/random/random.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/sort_and_group_coo_tensors_impl.h"

namespace jax_sc_embedding {

namespace {
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
    // Determine sample size for this row.
    int sample_size = static_cast<int>(std::round(std::exp(
        absl::Gaussian<float>(gen, kLognormalMean, kLognormalStddev))));
    int sc_id = row / batch_size_per_sc;
    extracted_coo_tensors.coo_tensors_per_sc[sc_id] += sample_size;

    // Generate sample_size IDs.
    for (int i = 0; i < sample_size; ++i) {
      int col_id;
      if (absl::Bernoulli(gen, kSkewProbability)) {
        // Skewed distribution.
        col_id = absl::Zipf<int>(gen, vocab_size - 1, kZipfQ);
      } else {
        // Uniform distribution.
        col_id = absl::Uniform<int>(gen, 0, vocab_size);
      }
      extracted_coo_tensors.coo_tensors.push_back(CooFormat(row, col_id, 1.0));
    }
  }
  return extracted_coo_tensors;
}

void BM_SortAndGroup_Phase1(benchmark::State& state) {
  ExtractedCooTensors extracted_coo_tensors =
      GenerateSkewedCooTensors(kNumScPerDevice, kBatchSizePerSc, kVocabSize);

  // Set to INT_MAX to avoid ID dropping and observe the actual statistics of
  // the generated data. This doesn't affect performance of grouping itself.
  StackedTableMetadata stacked_table_metadata(
      "stacked_table", /*feature_index=*/0,
      /*max_ids_per_partition=*/std::numeric_limits<int>::max(),
      /*max_unique_ids_per_partition=*/std::numeric_limits<int>::max(),
      /*row_offset=*/0,
      /*col_offset=*/0,
      /*col_shift=*/0, /*batch_size=*/0);
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
    SortAndGroupCooTensorsPerLocalDevice(
        extracted_coo_tensors, stacked_table_metadata, options,
        stats_per_device, minibatching_required);
    LogStats(stats_per_device.max_ids_per_partition,
             "Max ids per partition across all global SCs");
    LogStats(stats_per_device.max_unique_ids_per_partition,
             "Max unique ids per partition across all global SCs");
  }

  for (auto s : state) {
    SortAndGroupCooTensorsPerLocalDevice(
        extracted_coo_tensors, stacked_table_metadata, options,
        stats_per_device, minibatching_required);
  }
}
BENCHMARK(BM_SortAndGroup_Phase1)
    ->Threads(8)
    ->UseRealTime();

}  // namespace
}  // namespace jax_sc_embedding
