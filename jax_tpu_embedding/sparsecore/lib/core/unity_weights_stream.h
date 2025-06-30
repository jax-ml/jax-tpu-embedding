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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_UNITY_WEIGHTS_STREAM_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_UNITY_WEIGHTS_STREAM_H_

#include "jax_tpu_embedding/sparsecore/lib/core/abstract_input_batch.h"

namespace jax_sc_embedding {

// Class to iterate over a sparse CSR array, providing unity weights for each
// value. This class takes an existing `ValuesStream` (e.g.,
// `SparseCsrInputBatchStream`) and provides an interface to iterate over the
// same structure, but returning a weight of 1.0 for each value instead of the
// actual value. This is useful when the input does not have associated weights
// but the processing logic expects a weight stream.
template <typename ValuesStream>
class UnityWeightsStream
    : public AbstractInputBatchStream<float, UnityWeightsStream<ValuesStream>> {
 public:
  UnityWeightsStream(const ValuesStream& value_stream)
      : value_stream_(value_stream), curr_col_(0) {}

  int size() const { return value_stream_.size(); }

  int cols() const { return value_stream_.cols(); }

  void next_row() { curr_col_ = 0; }

  void next_col() { ++curr_col_; }

  void seek_col(int col) { curr_col_ = col; }

  int row() const { return value_stream_.row(); }

  int col() const { return curr_col_; }

  float get() const { return 1.0f; }

 private:
  const ValuesStream& value_stream_;
  int curr_col_;
};

template <typename T>
UnityWeightsStream(T) -> UnityWeightsStream<T>;

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_UNITY_WEIGHTS_STREAM_H_
