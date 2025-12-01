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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PARTITIONED_COO_TENSORS_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PARTITIONED_COO_TENSORS_H_

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_splits_impl.h"

namespace jax_sc_embedding {

class PartitionedCooTensors {
 public:
  PartitionedCooTensors() : PartitionedCooTensors(0, 0, 0, 1) {}
  PartitionedCooTensors(int reserve_count, int num_sc_per_device,
                        uint32_t global_sc_count, int bucket_count_per_sc = 1)
      : coo_tensors_(),
        bucket_count_per_sc_(bucket_count_per_sc),
        num_sc_per_device_(num_sc_per_device),
        global_sc_count_(global_sc_count),
        bucket_offsets_(),
        curr_sc_id_(0),
        curr_bucket_id_(0),
        merged_(false),
        dedup_col_id_(std::numeric_limits<uint32_t>::max()),
        dedup_row_id_(std::numeric_limits<uint32_t>::max()) {
    coo_tensors_.reserve(reserve_count);
    bucket_offsets_.reserve(1 + num_sc_per_device * bucket_count_per_sc_);
    bucket_offsets_.push_back(0);
  }

  inline void MergeWithLastCoo(const CooFormat& coo_tensor) {
    DCHECK_GT(coo_tensors_.size(), 0);
    CooFormat& last = coo_tensors_.back();
    DCHECK_EQ(last.row_id, coo_tensor.row_id);
    DCHECK_EQ(last.col_id, coo_tensor.col_id);
    last.gain += coo_tensor.gain;
  }

  inline bool MaybeMerge(const CooFormat& coo_tensor) {
    // If col_id is the same, bucket_id must also be the same.
    // For fastest short-circuiting, check row_id first, as it's the last
    // component of the sort key and thus most likely to differ between
    // consecutive non-identical elements.
    if (coo_tensor.row_id == dedup_row_id_ &&
        coo_tensor.col_id == dedup_col_id_) {
      MergeWithLastCoo(coo_tensor);
      return true;
    }
    return false;
  }

  void ResetDedupState() {
    dedup_col_id_ = std::numeric_limits<uint32_t>::max();
    dedup_row_id_ = std::numeric_limits<uint32_t>::max();
  }

  // Add Coo tensor for given SC and bucket. Similar to std::vector::push_back.
  void Add(int target_sc_id, int target_bucket_id,
           const CooFormat& coo_tensor) {
    AdvanceBucketOffsets(target_sc_id, target_bucket_id);

    coo_tensors_.push_back(coo_tensor);
    dedup_col_id_ = coo_tensor.col_id;
    dedup_row_id_ = coo_tensor.row_id;
  }

  // Get size of COO tensors for given SC.
  int Size(const int local_sc_id) const {
    DCHECK_GE(curr_sc_id_, local_sc_id);

    const int start_index = bucket_count_per_sc_ * local_sc_id;
    const int end_index = start_index + bucket_count_per_sc_;

    DCHECK_LT(end_index, bucket_offsets_.size());

    return bucket_offsets_[end_index] - bucket_offsets_[start_index];
  }

  // Get COO tensors for given SC and bucket.
  absl::Span<const CooFormat> operator()(int local_sc_id, int bucket_id) const {
    const int start_index = local_sc_id * bucket_count_per_sc_ + bucket_id;
    const int start_offset = bucket_offsets_[start_index];
    DCHECK_LT(start_index + 1, bucket_offsets_.size());
    const int end_offset = bucket_offsets_[start_index + 1];
    return absl::MakeSpan(coo_tensors_)
        .subspan(/*pos=*/start_offset,
                 /*len=*/end_offset - start_offset);
  }

  // Fill remaining buckets for current SC.
  void FillRemainingScBuckets() {
    AdvanceBucketOffsets(/* target_sc_id= */ curr_sc_id_ + 1,
                         /* target_bucket_id= */ 0);
  }

  // Merge two given buckets.
  template <size_t N = CooFormat::kMaxMinibatchingBuckets>
  void MergeBucketCoos(int left, int right, uint32_t global_sc_count,
                       std::bitset<N>& is_bucket_merged) {
    DCHECK(absl::has_single_bit(N));
    DCHECK(absl::has_single_bit(global_sc_count));
    const int subtree_size = right - left;
    DCHECK_GT(subtree_size, 0);
    int i = right - 1, j = right;
    while (i > left && is_bucket_merged.test(i)) {
      i--;  // Find last non-merged node in left subtree.
    }
    while (j < right + subtree_size && is_bucket_merged.test(j)) {
      j++;  // Find first non-merged node in right subtree.
    }
    for (int sc_id = 0; sc_id < num_sc_per_device_; ++sc_id) {
      const int ptr_left = bucket_offsets_[sc_id * bucket_count_per_sc_ + i];
      const int ptr_left_end =
          bucket_offsets_[sc_id * bucket_count_per_sc_ + i + 1];
      const int ptr_right = bucket_offsets_[sc_id * bucket_count_per_sc_ + j];
      const int ptr_right_end =
          bucket_offsets_[sc_id * bucket_count_per_sc_ + j + 1];
      DCHECK_EQ(ptr_left_end, ptr_right) << "Must be contiguos buckets.";
      // COOs already have same local SC and bucket, only need to compare global
      // SC.
      std::inplace_merge(
          coo_tensors_.begin() + ptr_left, coo_tensors_.begin() + ptr_right,
          coo_tensors_.begin() + ptr_right_end,
          [&global_sc_count](const CooFormat& a, const CooFormat& b) {
            return (a.col_id & (global_sc_count - 1)) <
                   (b.col_id & (global_sc_count - 1));
          });
      // Combine into the right bucket (arbitrary choice).
      bucket_offsets_[sc_id * bucket_count_per_sc_ + i + 1] = ptr_left;
      bucket_offsets_[sc_id * bucket_count_per_sc_ + j] = ptr_left;
      is_bucket_merged.set(i);
    }
  }

  // Merges all buckets per SparseCore.
  template <size_t N = CooFormat::kMaxMinibatchingBuckets>
  void MergeAll() {
    Merge<N>(0);
  }

  // Merges bucket contents as well as updating the offsets.
  template <size_t N = CooFormat::kMaxMinibatchingBuckets>
  void Merge(std::bitset<N - 1> split) {
    CHECK(!merged_) << "Merge can only be called once.";
    AdvanceBucketOffsets(/* target_sc_id= */ num_sc_per_device_,
                         /* target_bucket_id= */ 0);
    DCHECK_EQ(bucket_count_per_sc_, N);

    DCHECK(absl::c_is_sorted(bucket_offsets_));

    std::bitset<N> is_bucket_merged = 0;
    internal::MergeBuckets<N>(split, [&](int i, int j) {
      MergeBucketCoos(i, j, global_sc_count_, is_bucket_merged);
    });

    const int minibatches = split.count() + 1;  // N splits -> N+1 minibatches.

    for (int sc_id = 0; sc_id < num_sc_per_device_; ++sc_id) {
      const int start_pos = 1 + sc_id * bucket_count_per_sc_;
      const int dest_pos = 1 + sc_id * minibatches;
      for (int bucket_id = 0, minibatch_id = 0; bucket_id < N - 1;
           ++bucket_id) {
        if (!is_bucket_merged.test(bucket_id)) {
          bucket_offsets_[dest_pos + minibatch_id] =
              bucket_offsets_[start_pos + bucket_id];
          ++minibatch_id;
        }
      }
      // SparseCore partition.
      bucket_offsets_[dest_pos + minibatches - 1] =
          bucket_offsets_[start_pos + bucket_count_per_sc_ - 1];
    }

    bucket_offsets_.resize(1 + num_sc_per_device_ * minibatches);

    DCHECK(absl::c_is_sorted(bucket_offsets_));

    bucket_count_per_sc_ = minibatches;
    merged_ = true;
  }

  // Minibatches (after merging) or Max buckets (before merging).
  int GetNumMinibatches() const { return bucket_count_per_sc_; }

 private:
  // Advance bucket offsets to the given `target_sc_id` and `target_bucket_id`.
  void AdvanceBucketOffsets(int target_sc_id, int target_bucket_id) {
    // Every SC should have at least one ID, therefore SC Id will either be same
    // or be the successor of current one.
    DCHECK(target_sc_id == curr_sc_id_ + 1 || target_sc_id == curr_sc_id_);

    while (curr_sc_id_ < target_sc_id || curr_bucket_id_ < target_bucket_id) {
      DCHECK_LT(curr_sc_id_, num_sc_per_device_);
      DCHECK_LT(curr_bucket_id_, bucket_count_per_sc_);
      bucket_offsets_.push_back(coo_tensors_.size());
      if (curr_bucket_id_ == bucket_count_per_sc_ - 1) {
        ++curr_sc_id_;
        curr_bucket_id_ = 0;
      } else {
        ++curr_bucket_id_;
      }
    }
    DCHECK_EQ(curr_sc_id_, target_sc_id);
    DCHECK_EQ(curr_bucket_id_, target_bucket_id);
  }

  // Flattened list of COO Tensors for all SCs.
  std::vector<CooFormat> coo_tensors_;
  // Minibatching buckets per SC.
  int bucket_count_per_sc_;
  // Number of SCs per device.
  int num_sc_per_device_;
  // Number of global SCs.
  uint32_t global_sc_count_;
  // Minibatching bucket offsets for all SCs (=1 for non-minibatching per SC).
  std::vector<int> bucket_offsets_;
  // Internal counters of the current SC and bucket being populated.
  int curr_sc_id_;
  int curr_bucket_id_;
  // Merged into minibatches?
  bool merged_;
  // For deduplication, we track the properties of the last ID that was *not*
  // dropped.
  uint32_t dedup_col_id_;
  uint32_t dedup_row_id_;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PARTITIONED_COO_TENSORS_H_
