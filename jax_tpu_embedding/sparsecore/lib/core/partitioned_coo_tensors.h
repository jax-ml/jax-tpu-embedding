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
#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_splits_impl.h"

namespace jax_sc_embedding {

class PartitionedCooTensors {
 public:
  PartitionedCooTensors() : PartitionedCooTensors(0, 1) {}
  PartitionedCooTensors(uint32_t global_sc_count, int bucket_count_per_sc = 1)
      : coo_tensors_(),
        bucket_count_per_sc_(bucket_count_per_sc),
        global_sc_count_(global_sc_count),
        bucket_offsets_(),
        curr_bucket_id_(0),
        merged_(false),
        pending_col_id_(std::numeric_limits<uint32_t>::max()),
        pending_row_id_(std::numeric_limits<uint32_t>::max()),
        gain_for_pending_(0.0f),
        pending_count_(0),
        pending_bucket_id_(0) {
    bucket_offsets_.reserve(1 + bucket_count_per_sc_);
    bucket_offsets_.push_back(0);
  }

  void Reserve(int reserve_count) { coo_tensors_.reserve(reserve_count); }

  // Add Coo tensor for given SC and bucket. Similar to std::vector::push_back.
  ABSL_ATTRIBUTE_ALWAYS_INLINE void Add(int target_bucket_id, uint32_t row_id,
                                        uint32_t col_id, float gain) {
    if (row_id == pending_row_id_ && col_id == pending_col_id_) {
      DCHECK_EQ(target_bucket_id, pending_bucket_id_);
      pending_count_++;
      return;
    }
    // If we reach here, it's a new tensor group or bucket. Flush previous if
    // any.
    if (pending_count_ > 0) {
      AdvanceBucketOffsets(pending_bucket_id_);
      coo_tensors_.emplace_back(pending_row_id_, pending_col_id_,
                                gain_for_pending_ * pending_count_);
    }
    // Start new pending group.
    pending_bucket_id_ = target_bucket_id;
    pending_col_id_ = col_id;
    pending_row_id_ = row_id;
    gain_for_pending_ = gain;
    pending_count_ = 1;
  }

  // Get size of COO tensors for given SC.
  int Size() const {
    const int start_index = 0;
    const int end_index = start_index + bucket_count_per_sc_;

    DCHECK_LT(end_index, bucket_offsets_.size());

    return bucket_offsets_[end_index] - bucket_offsets_[start_index];
  }

  // Get COO tensors for given SC and bucket.
  absl::Span<const CooFormat> operator()(int bucket_id) const {
    const int start_index = bucket_id;
    const int start_offset = bucket_offsets_[start_index];
    DCHECK_LT(start_index + 1, bucket_offsets_.size());
    const int end_offset = bucket_offsets_[start_index + 1];
    return absl::MakeSpan(coo_tensors_)
        .subspan(/*pos=*/start_offset,
                 /*len=*/end_offset - start_offset);
  }

  // Fill remaining buckets for current SC.
  void FillRemainingScBuckets() {
    FlushPending();
    AdvanceBucketOffsets(bucket_count_per_sc_);
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
    const int ptr_left = bucket_offsets_[i];
    const int ptr_left_end = bucket_offsets_[i + 1];
    const int ptr_right = bucket_offsets_[j];
    const int ptr_right_end = bucket_offsets_[j + 1];
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
    bucket_offsets_[i + 1] = ptr_left;
    bucket_offsets_[j] = ptr_left;
    is_bucket_merged.set(i);
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
    CHECK_EQ(curr_bucket_id_, bucket_count_per_sc_)
        << "FillRemainingScBuckets() must be called before Merge().";
    DCHECK_EQ(bucket_count_per_sc_, N);

    DCHECK(absl::c_is_sorted(bucket_offsets_));

    std::bitset<N> is_bucket_merged = 0;
    internal::MergeBuckets<N>(split, [&](int i, int j) {
      MergeBucketCoos(i, j, global_sc_count_, is_bucket_merged);
    });

    const int minibatches = split.count() + 1;  // N splits -> N+1 minibatches.

    const int start_pos = 1;
    const int dest_pos = 1;
    for (int bucket_id = 0, minibatch_id = 0; bucket_id < N - 1; ++bucket_id) {
      if (!is_bucket_merged.test(bucket_id)) {
        bucket_offsets_[dest_pos + minibatch_id] =
            bucket_offsets_[start_pos + bucket_id];
        ++minibatch_id;
      }
    }
    // SparseCore partition.
    bucket_offsets_[dest_pos + minibatches - 1] =
        bucket_offsets_[start_pos + bucket_count_per_sc_ - 1];

    bucket_offsets_.resize(1 + minibatches);

    DCHECK(absl::c_is_sorted(bucket_offsets_));

    bucket_count_per_sc_ = minibatches;
    merged_ = true;
  }

  // Minibatches (after merging) or Max buckets (before merging).
  int GetNumMinibatches() const { return bucket_count_per_sc_; }

 private:
  void FlushPending() {
    if (pending_count_ > 0) {
      AdvanceBucketOffsets(pending_bucket_id_);
      coo_tensors_.emplace_back(pending_row_id_, pending_col_id_,
                                gain_for_pending_ * pending_count_);
      pending_count_ = 0;  // Mark as flushed
    }
  }

  // Advance bucket offsets to the given `target_bucket_id`.
  void AdvanceBucketOffsets(int target_bucket_id) {
    DCHECK_LT(curr_bucket_id_, bucket_count_per_sc_)
        << "Bucket offsets already finalized.";
    DCHECK_LE(target_bucket_id, bucket_count_per_sc_);

    while (curr_bucket_id_ < target_bucket_id) {
      bucket_offsets_.push_back(coo_tensors_.size());
      curr_bucket_id_++;
    }
  }

  // Flattened list of COO Tensors for all SCs.
  std::vector<CooFormat> coo_tensors_;
  // Minibatching buckets per SC.
  int bucket_count_per_sc_;
  // Number of global SCs.
  uint32_t global_sc_count_;
  // Minibatching bucket offsets for all SCs (=1 for non-minibatching per SC).
  std::vector<int> bucket_offsets_;
  // Internal counters of the current SC and bucket being populated.
  int curr_bucket_id_;
  // Merged into minibatches?
  bool merged_;
  // For deduplication, we track the properties of the pending ID.
  uint32_t pending_col_id_;
  uint32_t pending_row_id_;
  float gain_for_pending_;
  int pending_count_;
  int pending_bucket_id_;
};

struct DevicePartitionedCooTensors {
  std::vector<PartitionedCooTensors> grouped_coo_tensors;

  void FillRemainingScBuckets() {
    for (auto& grouped_coo_tensor : grouped_coo_tensors) {
      grouped_coo_tensor.FillRemainingScBuckets();
    }
  }

  template <size_t N = CooFormat::kMaxMinibatchingBuckets>
  void Merge(const std::bitset<N - 1> split) {
    for (auto& grouped_coo_tensor : grouped_coo_tensors) {
      grouped_coo_tensor.Merge<N>(split);
    }
  }

  void MergeAll() {
    for (auto& grouped_coo_tensor : grouped_coo_tensors) {
      grouped_coo_tensor.MergeAll();
    }
  }

  absl::Span<const CooFormat> operator()(int local_sc_id, int bucket_id) const {
    return grouped_coo_tensors[local_sc_id](bucket_id);
  }

  void Add(int local_sc_id, int bucket_id, const CooFormat& coo_tensor) {
    grouped_coo_tensors[local_sc_id].Add(bucket_id, coo_tensor.row_id,
                                         coo_tensor.col_id, coo_tensor.gain);
  }

  int GetNumMinibatches() const {
    for (const auto& grouped_coo_tensor : grouped_coo_tensors) {
      DCHECK_EQ(grouped_coo_tensor.GetNumMinibatches(),
                grouped_coo_tensors[0].GetNumMinibatches());
    }
    return grouped_coo_tensors[0].GetNumMinibatches();
  }
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PARTITIONED_COO_TENSORS_H_
