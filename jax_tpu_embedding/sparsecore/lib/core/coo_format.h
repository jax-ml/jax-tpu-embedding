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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

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
template <typename ColIdT>
inline uint64_t HighwayHash(ColIdT col_id) {
  static_assert(std::is_same_v<ColIdT, int32_t> ||
                std::is_same_v<ColIdT, int64_t>);
  highwayhash::HHResult64 result;
  highwayhash::HighwayHashCatT<HH_TARGET> hash(kHighwayHashKey);
  hash.Append(reinterpret_cast<const char*>(&col_id), sizeof(col_id));
  hash.Finalize(&result);
  return result;
}

template <typename ColIdT = int32_t>
struct CooFormatT {
  using ColIdType = ColIdT;
  using UColIdT = std::make_unsigned_t<ColIdT>;

  // Maximum buckets that can be formed during minibatching.
  static constexpr uint32_t kMaxMinibatchingBuckets = 64;
  // The minibatching split will have `num buckets - 1` split points (bits)
  // which should fit in uint64_t.
  static_assert(kMaxMinibatchingBuckets - 1 <=
                std::numeric_limits<uint64_t>::digits);

  // A deterministic hash function eventually used to compute mini-batching
  // bucket id as `hash(col_id) % bucket_count`.
  using HashFn = absl::FunctionRef<uint32_t(ColIdT col_id)>;

  // Default constructor. Note that using `= default` leaves members
  // uninitialized, which is more performant when creating arrays with `new[]`.
  CooFormatT() = default;

  CooFormatT(int row_id, ColIdT embedding_id, float gain, int col_shift,
             ColIdT col_offset, int num_scs_mod)
      : CooFormatT(row_id,
                   GetColId(embedding_id, col_shift, col_offset, num_scs_mod),
                   gain) {}

  CooFormatT(int row_id, ColIdT col_id, float gain)
      : col_id(col_id), row_id(row_id), gain(gain) {}

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
  ColIdT col_id;
  // Defines the Row ID as the sum of the Sample ID and the Row Offset,
  // where Row Offset accounts for table stacking.
  //
  // - Row Offset: Specifies the offset of the sample within the feature input,
  //               facilitating addressing in stacked tables.
  int row_id;
  // Combiner weight for this COO tensor.
  float gain;

  bool operator==(const CooFormatT& other) const = default;

  friend std::ostream& operator<<(std::ostream& os, const CooFormatT& coo) {
    os << absl::StrFormat("(%d, %d, %2.2f)", coo.row_id, coo.col_id, coo.gain);
    return os;
  }

  // Computes the ColID from the given Embedding ID using Col Shift and Offset.
  // The Offset is assumed to be a multiple of number of SC shards.
  static ColIdT GetColId(const ColIdT embedding_id, const int col_shift,
                         const ColIdT col_offset, const int num_scs_mod) {
    // This is equivalent to:
    // (embedding_id + col_shift) % num_sc_shards +
    //    (embedding_id // num_sc_shards * num_sc_shards) + col_offset

    // This calculation remaps the embedding ID to a new sparse core (SC). It
    // uses the low bits from embedding_id + col_shift to determine the new SC,
    // while shifting the high bits by given col_offset (which is already
    // multiplied by num_scs).
    const UColIdT u_mask = static_cast<UColIdT>(num_scs_mod);
    DCHECK_EQ(static_cast<UColIdT>(col_offset) & u_mask, 0);

    const UColIdT u_emb = static_cast<UColIdT>(embedding_id);
    const UColIdT u_shift =
        static_cast<UColIdT>(static_cast<ColIdT>(col_shift));
    const UColIdT u_offset = static_cast<UColIdT>(col_offset);
    return static_cast<ColIdT>(((u_emb + u_shift) & u_mask) +
                               (u_emb & ~u_mask) + u_offset);
  }

  // Static method to avoid creating a CooFormatT object when only col_id is
  // available, which is common when iterating over Struct-of-Arrays data
  // structures for performance.
  static uint32_t GetBucketId(ColIdT column_id,
                              HashFn hash_fn = HighwayHash<ColIdT>) {
    // Checks that kMaxMinibatchingBuckets is a power of 2.
    static_assert(absl::has_single_bit(kMaxMinibatchingBuckets));

    return hash_fn(column_id) & (kMaxMinibatchingBuckets - 1);
  }

  uint32_t GetBucketId(HashFn hash_fn = HighwayHash<ColIdT>) const {
    return GetBucketId(this->col_id, hash_fn);
  }
};

using CooFormat = CooFormatT<int32_t>;
static_assert(sizeof(CooFormat) == 12);
static_assert(sizeof(CooFormatT<int64_t>) == 16);
static_assert(std::is_trivially_default_constructible_v<CooFormat>);
static_assert(std::is_trivially_default_constructible_v<CooFormatT<int64_t>>);
static_assert(std::is_trivially_copyable_v<CooFormat>);
static_assert(std::is_trivially_copyable_v<CooFormatT<int64_t>>);

namespace internal {

struct KeyLayoutBase {
  // Bits taken by minibatching bucket ID.
  static constexpr uint32_t kMinibatchingBucketBits =
      absl::bit_width(CooFormat::kMaxMinibatchingBuckets - 1);
  // Bit offset for bucket_id in grouping key.
  static constexpr uint32_t kBucketIdOffset = 64 - kMinibatchingBucketBits;

  static uint32_t GetBucketIdFromKey(uint64_t key) {
    return key >> kBucketIdOffset;
  }

  static uint64_t SetBucketIdInKey(uint64_t key, uint32_t bucket_id) {
    DCHECK_EQ(GetBucketIdFromKey(key), 0);
    DCHECK_LE(bucket_id, CooFormat::kMaxMinibatchingBuckets - 1);
    return key | (uint64_t{bucket_id} << kBucketIdOffset);
  }
};

}  // namespace internal

// Compile-time-selected 64-bit VQSort key layout parameterized on ColIdT.
// `KeyLayout<int32_t>` uses fixed constant bit shifts for 32-bit column IDs,
// whereas `KeyLayout<int64_t>` dynamically splits the 58 non-bucket bits
// `[57:0]` across `{global_sc_id, local_embedding_id, data}`.
template <typename ColIdT = int32_t>
class KeyLayout;

// Fixed 64-bit VQSort key layout for 32-bit column IDs:
//   [63:58] bucket_id (6 bits)
//   [57:26] {global_sc_id, local_embedding_id} (32 bits) <- rotated col_id
//   [25:0]  data: COO index or row_id (26 bits)
template <>
class KeyLayout<int32_t> : public internal::KeyLayoutBase {
 public:
  // Bits for variable data (index or row_id).
  static constexpr uint32_t kDataBits = 32 - kMinibatchingBucketBits;
  // Mask for variable data (index or row_id).
  static constexpr uint32_t kDataMask = (uint32_t{1} << kDataBits) - 1;
  // Bit offset for rotated_col_id in grouping key.
  static constexpr uint32_t kRotatedColIdOffset = kDataBits;
  static_assert(kRotatedColIdOffset + 32 == kBucketIdOffset);

  constexpr KeyLayout() = default;

  static constexpr KeyLayout Make(uint32_t num_scs_bit,
                                  int64_t /*max_local_emb_id*/ = 0) {
    return KeyLayout(num_scs_bit);
  }

  constexpr uint32_t num_scs_bit() const { return num_scs_bit_; }
  constexpr uint32_t data_mask() const { return kDataMask; }

  uint64_t GetGroupingKey(uint32_t col_id, uint32_t data, bool create_buckets,
                          CooFormat::HashFn hash_fn = HighwayHash<int32_t>)
      const {
    // Ensures tensors are sorted first by bucket_id, then by SparseCore, and
    // finally by local embedding ID.
    const uint32_t bucket_id =
        create_buckets
            ? CooFormat::GetBucketId(static_cast<int32_t>(col_id), hash_fn)
            : 0;

    DCHECK_LE(data, kDataMask);

    // [global_sc_id, local_embedding_id]
    const uint32_t rotated_col_id = absl::rotr(col_id, num_scs_bit_);

    return (uint64_t{bucket_id} << kBucketIdOffset) |
           (uint64_t{rotated_col_id} << kRotatedColIdOffset) |
           static_cast<uint64_t>(data);
  }

  static uint32_t GetDataFromKey(uint64_t key) { return key & kDataMask; }

  static uint32_t GetRotatedColIdFromKey(uint64_t key) {
    return (key >> kRotatedColIdOffset) & 0xFFFFFFFF;
  }

  uint32_t GetColIdFromKey(uint64_t key) const {
    return absl::rotl(GetRotatedColIdFromKey(key), num_scs_bit_);
  }

 private:
  explicit constexpr KeyLayout(uint32_t num_scs_bit)
      : num_scs_bit_(num_scs_bit) {}

  uint32_t num_scs_bit_ = 0;
};

// Dynamic 64-bit VQSort key layout for stacked tables whose global column ID
// space exceeds INT32_MAX:
//   [63:58]             bucket_id (6 bits)
//   [57 : 58 - s]       global_sc_id (s = num_scs_bit bits)
//   [57 - s : d]        local_embedding_id (local_id_bits bits)
//   [d - 1 : 0]         data: COO index or row_id (d = data_bits bits)
//
// Bit budget (`s + local_id_bits + data_bits <= 58`, with per-shard CSR
// addressing capping `local_id_bits <= 31`, i.e., `2^31 - 1` rows/shard):
//   - 256 SparseCores (`s = 8`), `d = 19` (524K IDs/SC): up to `2^39` (~550B)
//     global rows.
//   - 2048 SparseCores (`s = 11`), `d = 16` (65K IDs/SC): up to `2^42` (~4.4T)
//     global rows.
template <>
class KeyLayout<int64_t> : public internal::KeyLayoutBase {
 public:
  constexpr KeyLayout() = default;

  static KeyLayout Make(
      uint32_t num_scs_bit,
      int64_t max_local_emb_id = std::numeric_limits<int32_t>::max()) {
    const uint32_t local_bits = std::max<uint32_t>(
        1, absl::bit_width(
               static_cast<uint64_t>(std::max<int64_t>(0, max_local_emb_id))));
    DCHECK_LT(num_scs_bit + local_bits, kBucketIdOffset);
    const uint32_t d_bits = kBucketIdOffset - num_scs_bit - local_bits;
    return KeyLayout(num_scs_bit, local_bits, d_bits);
  }

  constexpr uint32_t num_scs_bit() const { return num_scs_bit_; }
  constexpr uint64_t local_id_mask() const { return local_id_mask_; }
  constexpr uint64_t data_mask() const { return data_mask_val_; }

  uint64_t GetGroupingKey(
      uint64_t col_id, uint64_t data, bool create_buckets,
      CooFormatT<int64_t>::HashFn hash_fn = HighwayHash<int64_t>) const {
    const uint32_t bucket_id =
        create_buckets ? CooFormatT<int64_t>::GetBucketId(
                             static_cast<int64_t>(col_id), hash_fn)
                       : 0;
    DCHECK_LE(data, data_mask_val_);
    const uint64_t global_sc_id = col_id & num_scs_mask_;
    const uint64_t local_emb_id = col_id >> num_scs_bit_;
    DCHECK_LE(local_emb_id, local_id_mask_);
    return (uint64_t{bucket_id} << kBucketIdOffset) |
           (global_sc_id << sc_id_offset_) |
           ((local_emb_id & local_id_mask_) << data_bits_) |
           (data & data_mask_val_);
  }

  uint64_t GetDataFromKey(uint64_t key) const { return key & data_mask_val_; }

  uint64_t GetColIdFromKey(uint64_t key) const {
    const uint64_t local_emb_id = (key >> data_bits_) & local_id_mask_;
    const uint64_t global_sc_id = (key >> sc_id_offset_) & num_scs_mask_;
    return (local_emb_id << num_scs_bit_) | global_sc_id;
  }

 private:
  constexpr KeyLayout(uint32_t num_scs_bit, uint32_t local_id_bits,
                      uint32_t data_bits)
      : num_scs_bit_(num_scs_bit),
        data_bits_(data_bits),
        sc_id_offset_(kBucketIdOffset - num_scs_bit),
        num_scs_mask_((uint64_t{1} << num_scs_bit) - 1),
        local_id_mask_((uint64_t{1} << local_id_bits) - 1),
        data_mask_val_((uint64_t{1} << data_bits) - 1) {}

  uint32_t num_scs_bit_ = 0;
  uint32_t data_bits_ = 27;
  uint32_t sc_id_offset_ = kBucketIdOffset;
  uint64_t num_scs_mask_ = 0;
  uint64_t local_id_mask_ = (uint64_t{1} << 31) - 1;
  uint64_t data_mask_val_ = (uint64_t{1} << 27) - 1;
};

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_COO_FORMAT_H_
