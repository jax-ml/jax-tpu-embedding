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

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"

namespace jax_sc_embedding {
namespace internal {

// Computes a minibatching split indicator for a given set of unique IDs per
// bucket.
//
// Example:
//   Suppose we have 8 buckets and `max_unique_ids_per_partition` is 7.
//   unique_ids_per_bucket = {2, 5, 1, 6, 3, 4, 2, 0}
//
//   The binary tree computation proceeds as follows:
//     - Level 1:
//       - {2, 5} -> 7 <= 7, no split, unique_ids_per_bucket[0] = 7
//       - {1, 6} -> 7 <= 7, no split, unique_ids_per_bucket[2] = 7
//       - {3, 4} -> 7 <= 7, no split, unique_ids_per_bucket[4] = 7
//       - {2, 0} -> 2 <= 7, no split, unique_ids_per_bucket[6] = 2
//     - Level 2:
//       - {7, 7} -> 14 > 7, split, split_index = 4
//       - {7, 2} -> 9 > 7, split, split_index = 5
//     - Level 3:
//       - {14, 9} -> 23 > 7, split, split_index = 6
//
//   The resulting `split` bitset is 0b1110000.
//
// WARNING: This function modifies the input `unique_ids_per_bucket` by
// combining counts of sibling buckets for efficiency.
//
// Example for subtree_size:
//   - subtree_size = 2: {0,1} {2,3} {4,5}, ...
//   - subtree_size = 4: {0,2} {4,6} {8,10}, ...
//   - ...
//   - subtree_size = N: {0, N/2}
// If two buckets are merged, the count is combined into the left sibling and
// propagated up the tree.
template <size_t N = CooFormat::kMaxMinibatchingBuckets>
std::bitset<N - 1> ComputeMinibatchingSplit(
    absl::Span<int32_t> unique_ids_per_bucket,
    int max_unique_ids_per_partition) {
  static_assert(absl::has_single_bit(N));
  DCHECK_EQ(unique_ids_per_bucket.size(), N);
  std::bitset<N - 1> split = 0;
  int split_index = 0;
  for (int subtree_size = 2; subtree_size <= N; subtree_size *= 2) {
    for (int i = 0; i < N; i += subtree_size, ++split_index) {
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

using MergeFn = absl::FunctionRef<void(int, int)>;
struct NoOpMerge {
  void operator()(int, int) const {}
};

// Merges buckets based on a binary tree split indicator.
// The `split` bitset indicates which nodes of the binary tree should be split.
// If a node is not split, its left and right children are merged using the
// provided `merge_fun`.
//
// Example:
//   Input split: 0b1100011 (split index {0,1,5,6})
//   This represents splits between:
//     - {0,1}
//     - {2,3}
//     - {5,6}
//     - {3,4}
//   This function will call `merge_fun` for the following pairs of buckets:
//     - index = 2: {4,5}
//     - index = 3: {6,7}
//     - index = 4: {1,2}
//
// pos         0 1 2 3 4 5 6 7
// split index  0   1   2   3
// split index    4       5
// split index        6
//
// The `merge_fun` is called with the indices of the buckets to be merged.
template <size_t N>
void MergeBuckets(std::bitset<N - 1> split, MergeFn merge_fun = NoOpMerge()) {
  static_assert(absl::has_single_bit(N));
  int split_index = 0;
  for (int subtree_size = 2; subtree_size <= N; subtree_size *= 2) {
    for (int i = 0; i < N; i += subtree_size, ++split_index) {
      const int right_index = i + subtree_size / 2;
      const int left_index = i;
      // Merge/Split left subtree with right subtree.
      // Implementations decide how to do the merging, for input preprocessing,
      // we merge the last unmerged node in left subtree with first unmerged
      // node in right subtree.
      if (!split.test(split_index)) {
        merge_fun(left_index, right_index);
      }
    }
  }
}

}  // namespace internal
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SPLITS_IMPL_H_
