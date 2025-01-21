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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_WITH_MINI_BATCHING_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_WITH_MINI_BATCHING_H_

namespace jax_sc_embedding {

// The mode of mini-batching operation.
enum class MiniBatchingMode {
  // No mini-batching, essentially the same as
  // MINI_BATCHING_EXPERIMENTAL_FORCE_VOCABULARY_DIV with mini_batch_size = 1.
  kNone = 1,

  // If there is no need to mini-batch in any of the tasks for this input batch,
  // this is essentially
  // the same as MINI_BATCHING_NONE.
  // First hash the embedding IDs into 2^64 domain, and then modulo into
  // 2^max_division_level buckets. Finally optimize the number of buckets
  // necessary to a minimum through merging neighboring buckets and
  // communication among all tasks.
  kVocabularyDimension = 2,

  kSampleDimension = 3,

  // Linearly divide the vocabulary dimension into specified mini_batch_size
  // slices.
  kExperimentalForceVocabularyDiv = 200,

  // Split the vocabulary dimension into specified mini_batch_size slices
  // through simple modulo operations.
  kExperimentalForceVocabularyMod = 201,
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_INPUT_PREPROCESSING_WITH_MINI_BATCHING_H_
