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
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace jax_sc_embedding {
namespace {

TEST(CooFormatTest, HighwayHashInt64UsesUpperBits) {
  constexpr int64_t kLowColId = 156;
  constexpr int64_t kHighColId = (int64_t{1} << 35) | kLowColId;

  EXPECT_NE(HighwayHash<int64_t>(kLowColId), HighwayHash<int64_t>(kHighColId));
}

TEST(CooFormatTest, GetBucketIdMasksToBucketRange) {
  const CooFormat coo(/*row_id=*/3, /*col_id=*/156, /*gain=*/1.0f);
  const auto hash_fn = [](int32_t) -> uint32_t { return (3U << 6) | 21U; };

  EXPECT_EQ(coo.GetBucketId(hash_fn), 21U);
  EXPECT_LT(coo.GetBucketId(), CooFormat::kMaxMinibatchingBuckets);
}

TEST(CooFormatTest, KeyLayoutInt32EmbedsBucketColAndData) {
  const KeyLayout<int32_t> layout = KeyLayout<int32_t>::Make(/*num_scs_bit=*/4);

  const uint64_t key = layout.GetGroupingKey(
      /*col_id=*/156U, /*data=*/7U, /*create_buckets=*/true);

  EXPECT_EQ(KeyLayout<int32_t>::GetBucketIdFromKey(key),
            CooFormat::GetBucketId(156));
  EXPECT_EQ(layout.GetColIdFromKey(key), 156U);
  EXPECT_EQ(KeyLayout<int32_t>::GetDataFromKey(key), 7U);
}

TEST(CooFormatTest, KeyLayoutInt64EmbedsBucketColAndData) {
  const KeyLayout<int64_t> layout =
      KeyLayout<int64_t>::Make(/*num_scs_bit=*/8, /*max_local_emb_id=*/1 << 24);
  constexpr uint64_t kColId = 4'999'999'996ULL;

  const uint64_t key = layout.GetGroupingKey(
      /*col_id=*/kColId, /*data=*/19U, /*create_buckets=*/true);

  EXPECT_EQ(KeyLayout<int64_t>::GetBucketIdFromKey(key),
            CooFormatT<int64_t>::GetBucketId(static_cast<int64_t>(kColId)));
  EXPECT_EQ(layout.GetColIdFromKey(key), kColId);
  EXPECT_EQ(layout.GetDataFromKey(key), 19U);
}

}  // namespace
}  // namespace jax_sc_embedding
