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
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

template <typename WeightsStreamT>
float ComputeWeightDivisor(RowCombiner combiner,
                           WeightsStreamT& weights_stream) {
  switch (combiner) {
    case RowCombiner::kSum:
      return 1.0f;
    case RowCombiner::kMean: {
      // Sum of elements.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.NextCol()) {
        sum += weights_stream.get();
      }
      return sum;
    }
    case RowCombiner::kSqrtn: {
      // Sqrt of sum of squares.
      float sum = 0.0f;
      for (; weights_stream.col() < weights_stream.cols();
           weights_stream.NextCol()) {
        sum += std::pow(weights_stream.get(), 2);
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
  const int num_scs_mod_inv = ~num_scs_mod;
  const int batch_size_per_sc =
      extracted_coo_tensors.batch_size_for_device / options.num_sc_per_device;
  CHECK_GT(batch_size_per_sc, 0);

  extracted_coo_tensors.coo_tensors.reserve(values_stream.size());

  DCHECK_EQ(values_stream.size(), weights_stream.size());

  for (; values_stream.row() < options.slice_end &&
         weights_stream.row() < options.slice_end;
       values_stream.NextRow(), weights_stream.NextRow()) {
    DCHECK_EQ(values_stream.cols(), weights_stream.cols());
    DCHECK_EQ(values_stream.row(), weights_stream.row());
    DCHECK_EQ(values_stream.col(), weights_stream.col());
    DCHECK_EQ(values_stream.col(), 0);

    const int sample_id =
        values_stream.row() - options.slice_start + options.row_offset;
    const float divisor =
        ComputeWeightDivisor(options.combiner, weights_stream);

    for (weights_stream.SeekCol(0); values_stream.col() < values_stream.cols();
         values_stream.NextCol(), weights_stream.NextCol()) {
      const int embedding_id = values_stream.get();
      const float gain = weights_stream.get() / divisor;
      DCHECK_GE(embedding_id, 0);
      DCHECK_LT(sample_id, batch_size_per_sc * options.num_sc_per_device);

      // Compute per-SC coo tensor size.
      extracted_coo_tensors.coo_tensors_per_sc[sample_id / batch_size_per_sc]++;

      extracted_coo_tensors.coo_tensors.emplace_back(
          sample_id,
          GetColId(embedding_id, options.col_shift, options.col_offset,
                   num_scs_mod, num_scs_mod_inv),
          gain);
    }
  }
}
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PROCESS_COO_TENSORS_IMPL_H_
