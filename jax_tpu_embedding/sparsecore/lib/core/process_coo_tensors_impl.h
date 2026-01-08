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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PROCESS_COO_TENSORS_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PROCESS_COO_TENSORS_IMPL_H_

#include <cmath>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/unity_weights_stream_impl.h"

namespace jax_sc_embedding {

// Helper to check if a type is an instantiation of UnityWeightsStream.
template <typename>
struct is_unity_weights_stream : std::false_type {};
template <typename T>
struct is_unity_weights_stream<UnityWeightsStream<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_unity_weights_stream_v =
    is_unity_weights_stream<T>::value;

// Overload for UnityWeightsStream.
template <typename ValuesStream>
float ComputeWeightDivisor(RowCombiner combiner,
                           UnityWeightsStream<ValuesStream>& weights_stream) {
  const int num_cols = weights_stream.cols();
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean:
      return static_cast<float>(num_cols);
    case RowCombiner::kSqrtn:
      return std::sqrt(static_cast<float>(num_cols));
  }
}

// Generic implementation for any weight stream that is not UnityWeightsStream.
template <typename WeightsStreamT>
float ComputeWeightDivisor(RowCombiner combiner,
                           WeightsStreamT& weights_stream) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      // Sum of elements.
      float sum = 0.0f;
      for (const float weight : weights_stream.getRowSpan()) {
        sum += weight;
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (const float weight : weights_stream.getRowSpan()) {
        sum += std::pow(weight, 2);
      }
      return std::sqrt(sum);
    }
  }
}

template <typename ValuesStreamT, typename WeightsStreamT>
void ProcessCooTensors(
    const AbstractInputBatch::ExtractCooTensorsOptions& options,
    ValuesStreamT& values_stream, WeightsStreamT& weights_stream,
    ExtractedCooTensors& extracted_coo_tensors) {
  CHECK(options.num_scs > 0 && (options.num_scs & (options.num_scs - 1)) == 0);
  CHECK_GT(extracted_coo_tensors.batch_size_for_device, 0);
  CHECK_GT(options.num_sc_per_device, 0);

  const int num_scs_bit = std::log2(options.num_scs);
  const int num_scs_mod = (1 << num_scs_bit) - 1;
  DCHECK_EQ(
      extracted_coo_tensors.batch_size_for_device % options.num_sc_per_device,
      0);
  const int batch_size_per_sc =
      extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
  CHECK_GT(batch_size_per_sc, 0);

  for (; values_stream.row() < options.slice_end;
       values_stream.NextRow(), weights_stream.NextRow()) {
    DCHECK_LT(weights_stream.row(), options.slice_end);
    DCHECK_EQ(values_stream.cols(), weights_stream.cols());
    DCHECK_EQ(values_stream.row(), weights_stream.row());

    const int sample_id =
        values_stream.row() - options.slice_start + options.row_offset;
    const int num_cols = values_stream.cols();
    float divisor = 1.0f;
    if constexpr (is_unity_weights_stream_v<WeightsStreamT>) {
      if (options.combiner == RowCombiner::kMean) {
        extracted_coo_tensors.row_token_counts[sample_id] = num_cols;
      } else if (options.combiner == RowCombiner::kSqrtn) {
        divisor = ComputeWeightDivisor(options.combiner, weights_stream);
        extracted_coo_tensors.row_gains[sample_id] = 1.0f / divisor;
      }
    } else {
      divisor = ComputeWeightDivisor(options.combiner, weights_stream);
    }

    extracted_coo_tensors.coo_tensors_per_sc[sample_id / batch_size_per_sc] +=
        num_cols;
    DCHECK_LT(sample_id, batch_size_per_sc * options.num_sc_per_device);

    const auto values = values_stream.getRowSpan();  // template-dependent type.
    if constexpr (!is_unity_weights_stream_v<WeightsStreamT>) {
      const absl::Span<const float> weights = weights_stream.getRowSpan();
      for (int i = 0; i < num_cols; ++i) {
        const int embedding_id = values[i];
        const float gain = weights[i] / divisor;
        DCHECK_GE(embedding_id, 0);

        extracted_coo_tensors.emplace_back_with_gain(
            sample_id, embedding_id, gain, options.col_shift,
            options.col_offset, num_scs_mod);
      }
    } else {
      for (int i = 0; i < num_cols; ++i) {
        const int embedding_id = values[i];
        DCHECK_GE(embedding_id, 0);

        extracted_coo_tensors.emplace_back(sample_id, embedding_id,
                                           options.col_shift,
                                           options.col_offset, num_scs_mod);
      }
    }
  }
}
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PROCESS_COO_TENSORS_IMPL_H_
