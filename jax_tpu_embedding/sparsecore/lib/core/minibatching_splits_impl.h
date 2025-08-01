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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SPLITS_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SPLITS_IMPL_H_

#include <bitset>
#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"

namespace jax_sc_embedding {
namespace internal {

// Computes the minibatching split for given unique ids per bucket where a set
// bit represents a split.
// WARNING: Modifies the original unique id count per bucket with
// combined counts for efficiency.
// Combines values bottom up starting with subtree of size 2.
// For e.g.
// subtree_size = 2: {0,1} {2,3} {4,5}, ...
// subtree_size = 4: {0,2} {4,6} {8,10}, ...
// ...
// subtree_size = N: {0, N/2}
// If two buckets are merged, the value is combined in the left subtree and
// compared up.
template <size_t N = CooFormat::kMaxMinibatchingBuckets>
std::bitset<N - 1> ComputeMinibatchingSplit(
    absl::Span<int32_t> unique_ids_per_bucket,
    int max_unique_ids_per_partition) {
  static_assert(absl::has_single_bit(N));
  DCHECK_EQ(unique_ids_per_bucket.size(), N);
  std::bitset<N - 1> split = 0;
  const int tree_size = unique_ids_per_bucket.size();
  int split_index = 0;
  for (int subtree_size = 2; subtree_size <= tree_size; subtree_size *= 2) {
    for (int i = 0; i < tree_size; i += subtree_size, ++split_index) {
      const int val_left = unique_ids_per_bucket[i];
      const int val_right = unique_ids_per_bucket[i + subtree_size / 2];
      if (val_left + val_right > max_unique_ids_per_partition) {
        split.set(split_index);
      } else {
        unique_ids_per_bucket[i] += val_right;
      }
    }
  }
  return split;
}

// Converts a binary tree splits into split positions. For example 0b1100011
// (split index {0,1,5,6}) represents a split between {0,1}, {2,3}, {6,7}, {3,4}
// resulting in split positions as {0,2,3,5}.
//
// pos         0 1 2 3 4 5 6 7
// split index  0   1   2   3
// split index    4       5
// split index        6
template <size_t N>
std::bitset<N - 1> GetSplitPos(std::bitset<N - 1> split) {
  static_assert(absl::has_single_bit(N));
  int split_index = 0;
  std::bitset<N - 1> splitpos;
  for (int subtree_size = 2; subtree_size <= N; subtree_size *= 2) {
    for (int i = 0; i < N; i += subtree_size, ++split_index) {
      if (split.test(split_index)) {
        const int right_index = i + subtree_size / 2;
        // Split between {right_index -1, right_index}.
        splitpos.set(right_index - 1);
      }
    }
  }
  return splitpos;
}

}  // namespace internal
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SPLITS_IMPL_H_
