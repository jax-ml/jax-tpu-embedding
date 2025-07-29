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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_COO_FORMAT_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_COO_FORMAT_H_

#include <cstdint>
#include <optional>
#include <ostream>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl

namespace jax_sc_embedding {

struct CooFormat {
  // Maximum buckets that can be formed during minibatching.
  static constexpr uint32_t kMaxMinibatchingBuckets = 64;
  // Bits taken by minibatching bucket ID.
  static constexpr uint32_t kMinibatchingBucketBits =
      absl::bit_width(kMaxMinibatchingBuckets - 1);
  // Bits for Index
  static constexpr uint32_t kIndexBits = 32 - kMinibatchingBucketBits;
  // Index Mask
  static constexpr uint32_t kIndexMask = (1 << kIndexBits) - 1;

  // A deterministic hash function eventually used to compute mini-batching
  // bucket id as `hash(col_id) % bucket_count`.
  using HashFn = absl::FunctionRef<uint32_t(int col_id)>;

  // C++17-compatible replacement for std::identity.
  struct Identity {
    uint32_t operator()(int col_id) const { return col_id; }
  };

  CooFormat(int row_id, int embedding_id, float gain, int col_shift,
            int col_offset, int num_scs_mod)
      : CooFormat(row_id,
                  GetColId(embedding_id, col_shift, col_offset, num_scs_mod),
                  gain) {}

  CooFormat(int row_id, int col_id, float gain)
      : row_id(row_id), col_id(col_id), gain(gain) {}

  // Defines the Row ID as the sum of the Sample ID and the Row Offset,
  // where Row Offset accounts for table stacking.
  //
  // - Row Offset: Specifies the offset of the sample within the feature input,
  //               facilitating addressing in stacked tables.
  int row_id;
  // Represents a packed structure where a single integer word encodes:
  // Col ID (Embedding ID), Col Shift, and Col Offset.
  //
  // - Col Shift: Defines table rotation.
  // - Col Offset: Specifies the embedding's offset in a stacked embedding
  // table.
  //
  // This packing allows for efficient storage and extractions using bitwise
  // masks (assuming number of sparsecores (SC) is a power of 2).
  //
  // Shard ID = col_id % num_scs (global SC id)
  // Local Embedding ID = col_id / num_scs
  //
  // Col ID = [Local Embedding ID(w/ Col Offset), Shard ID(w/ Col Shift)]
  int col_id;
  // Combiner weight for this COO tensor.
  float gain;

  bool operator==(const CooFormat& other) const = default;

  friend std::ostream& operator<<(std::ostream& os, const CooFormat& coo) {
    os << absl::StrFormat("(%d, %d, %2.2f)", coo.row_id, coo.col_id, coo.gain);
    return os;
  }

  // Computes the ColID from the given Embedding ID using Col Shift and Offset.
  // The Offset is assumed to be a multiple of number of SC shards.
  static int GetColId(const int embedding_id, const int col_shift,
                      const int col_offset, const int num_scs_mod) {
    // This is equivalent to:
    // (embedding_id + col_shift) % num_sc_shards +
    //    (embedding_id // num_sc_shards * num_sc_shards) + col_offset

    // This calculation remaps the embedding ID to a new sparse core (SC). It
    // uses the low bits from embedding_id + col_shift to determine the new SC,
    // while shifting the high bits by given col_offset (which is already
    // multiplied by num_scs).

    DCHECK_EQ(col_offset & num_scs_mod, 0);

    return ((embedding_id + col_shift) & num_scs_mod) +
           (embedding_id & ~num_scs_mod) + col_offset;
  }

  // TODO: b/428790659 - Update hash function.
  uint32_t GetBucketId(HashFn hash_fn = Identity()) const {
    // Checks that kMaxMinibatchingBuckets is a power of 2.
    static_assert(absl::has_single_bit(kMaxMinibatchingBuckets));

    return hash_fn(col_id) & (kMaxMinibatchingBuckets - 1);
  }

  // Computes a 64-bit sorting key with the following layout:
  // [63:58] bucket_id (6 bits)
  // [57:26] {global_sc_id, local_embedding_id} (32 bits) <- rotated col_id
  // [25:0]  index (26 bits)
  // The key is used to group and sort COO tensors for efficient processing.
  // TODO: b/428790659 - Update hash function.
  uint64_t GetGroupingKey(const uint32_t num_scs_bit, const int index,
                          const bool enable_minibatching = false,
                          HashFn hash_fn = Identity()) const {
    // This structure ensures tensors are sorted first by bucket_id, then by
    // sparse core, and finally by embedding ID.
    const uint32_t bucket_id = enable_minibatching ? GetBucketId(hash_fn) : 0;

    DCHECK_LE(index, kIndexMask);

    // [global_sc_id, local_embedding_id]
    uint32_t rotated_col_id =
        absl::rotr(static_cast<uint32_t>(col_id), num_scs_bit);

    return (uint64_t{bucket_id} << (64 - kMinibatchingBucketBits)) |
           (uint64_t{rotated_col_id} << (32 - kMinibatchingBucketBits)) |
           static_cast<uint64_t>(index);
  }
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_COO_FORMAT_H_
