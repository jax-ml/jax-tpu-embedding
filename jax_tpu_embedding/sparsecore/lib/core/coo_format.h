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
#include <limits>
#include <ostream>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "highwayhash/arch_specific.h"  // from @highwayhash
#include "highwayhash/hh_types.h"  // from @highwayhash
#include "highwayhash/highwayhash.h"  // from @highwayhash

namespace jax_sc_embedding {

// Arbitrarily chosen, forever-unchanging hash key required by HighwayHash.
inline constexpr highwayhash::HHKey kHighwayHashKey = {
    0x4451e30f87db9609ULL,
    0xca7358a1fd2737f8ULL,
    0x4b2c991fcee4fdeaULL,
    0x0b2658e18326f6baULL,
};

// HighwayHash is a fast, strong (well-distributed and unpredictable)
// pseudo-random-function. Its output is stable, meaning it won't change over
// time for the same key and input. This makes it suitable for deterministic
// random selection.
inline uint64_t HighwayHash(int col_id) {
  highwayhash::HHResult64 result;
  highwayhash::HighwayHashCatT<HH_TARGET> hash(kHighwayHashKey);
  hash.Append(reinterpret_cast<const char*>(&col_id), sizeof(col_id));
  hash.Finalize(&result);
  return result;
}

struct CooFormat {
  // Maximum buckets that can be formed during minibatching.
  static constexpr uint32_t kMaxMinibatchingBuckets = 64;
  // The minibatching split will have `num buckets - 1` split points (bits)
  // which should fit in uint64_t.
  static_assert(kMaxMinibatchingBuckets - 1 <=
                std::numeric_limits<uint64_t>::digits);
  // Bits taken by minibatching bucket ID.
  static constexpr uint32_t kMinibatchingBucketBits =
      absl::bit_width(kMaxMinibatchingBuckets - 1);
  // Bits for variable data (index or row_id).
  static constexpr uint32_t kDataBits = 32 - kMinibatchingBucketBits;
  // Mask for variable data (index or row_id).
  static constexpr uint32_t kDataMask = (1 << kDataBits) - 1;
  // Bit offset for rotated_col_id in grouping key.
  static constexpr uint32_t kRotatedColIdOffset = kDataBits;
  // Bit offset for bucket_id in grouping key.
  static constexpr uint32_t kBucketIdOffset = kRotatedColIdOffset + 32;

  // A deterministic hash function eventually used to compute mini-batching
  // bucket id as `hash(col_id) % bucket_count`.
  using HashFn = absl::FunctionRef<uint32_t(int col_id)>;

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

  // Static method to avoid creating a CooFormat object when only col_id is
  // available, which is common when iterating over Struct-of-Arrays data
  // structures for performance.
  static uint32_t GetBucketId(int col_id, HashFn hash_fn = HighwayHash) {
    // Checks that kMaxMinibatchingBuckets is a power of 2.
    static_assert(absl::has_single_bit(kMaxMinibatchingBuckets));

    return hash_fn(col_id) & (kMaxMinibatchingBuckets - 1);
  }

  uint32_t GetBucketId(HashFn hash_fn = HighwayHash) const {
    return GetBucketId(col_id, hash_fn);
  }

  // Static method to avoid creating a CooFormat object when only col_id and
  // data are available, which is common when iterating over Struct-of-Arrays
  // data structures for performance.
  static uint64_t GetGroupingKey(uint32_t col_id, uint32_t data,
                                 const uint32_t num_scs_bit,
                                 const bool create_buckets,
                                 HashFn hash_fn = HighwayHash) {
    // This structure ensures tensors are sorted first by bucket_id, then by
    // sparse core, and finally by embedding ID.
    const uint32_t bucket_id =
        create_buckets ? GetBucketId(col_id, hash_fn) : 0;

    DCHECK_LE(data, kDataMask);

    // [global_sc_id, local_embedding_id]
    uint32_t rotated_col_id =
        absl::rotr(static_cast<uint32_t>(col_id), num_scs_bit);

    return (uint64_t{bucket_id} << kBucketIdOffset) |
           (uint64_t{rotated_col_id} << kRotatedColIdOffset) |
           static_cast<uint64_t>(data);
  }

  // Computes a 64-bit sorting key with the following layout:
  // [63:58] bucket_id (6 bits)
  // [57:26] {global_sc_id, local_embedding_id} (32 bits) <- rotated col_id
  // [25:0]  index or row_id (26 bits)
  // The key is used to group and sort COO tensors for efficient processing.
  uint64_t GetGroupingKey(const uint32_t num_scs_bit, const int index,
                          const bool create_buckets,
                          HashFn hash_fn = HighwayHash,
                          const bool has_variable_weights = true) const {
    return GetGroupingKey(col_id, has_variable_weights ? index : row_id,
                          num_scs_bit, create_buckets, hash_fn);
  }

  static uint32_t GetDataFromKey(uint64_t key) { return key & kDataMask; }

  static uint32_t GetRotatedColIdFromKey(uint64_t key) {
    return (key >> kRotatedColIdOffset) & 0xFFFFFFFF;
  }

  static uint32_t GetBucketIdFromKey(uint64_t key) {
    return key >> kBucketIdOffset;
  }
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_COO_FORMAT_H_
