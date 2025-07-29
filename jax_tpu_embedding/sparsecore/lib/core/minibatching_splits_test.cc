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
#include <bitset>
#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_splits_impl.h"

namespace jax_sc_embedding {
namespace {

using ::testing::Values;

class ComputeMinibatchingSplitTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<int32_t>, int32_t, std::bitset<7>>> {};

TEST_P(ComputeMinibatchingSplitTest, TestSplits) {
  std::vector<int32_t> unique_ids_per_bucket;
  int32_t max_ids_per_partition;
  std::bitset<7> expected_split;
  std::tie(unique_ids_per_bucket, max_ids_per_partition, expected_split) =
      GetParam();

  std::bitset<7> split = internal::ComputeMinibatchingSplit<8>(
      unique_ids_per_bucket, max_ids_per_partition);
  EXPECT_EQ(split, expected_split);
}

INSTANTIATE_TEST_SUITE_P(
    TestSplits, ComputeMinibatchingSplitTest,
    Values(
        // Full Merge
        std::make_tuple(std::vector<int32_t>(8, 10), 100, std::bitset<7>(0)),
        // No Merges
        std::make_tuple(std::vector<int32_t>(8, 100), 150,
                        std::bitset<7>(0b1111111)),
        // Partial Merge
        std::make_tuple(std::vector<int32_t>{10, 20, 30, 40, 50, 60, 70, 80},
                        100, std::bitset<7>(0b1101100)),
        // Partial Merge 2
        std::make_tuple(std::vector<int32_t>{50, 50, 20, 80, 10, 10, 70, 10},
                        100, std::bitset<7>(0b1010000)),
        // Partial Merge 3
        std::make_tuple(std::vector<int32_t>{90, 10, 90, 10, 90, 10, 90, 10},
                        100, std::bitset<7>(0b1110000))));

class GetSplitPosTest : public ::testing::TestWithParam<
                            std::tuple<std::bitset<7>, std::bitset<7>>> {};

TEST_P(GetSplitPosTest, TestSplitPos) {
  std::bitset<7> split;
  std::bitset<7> expected_splitpos;
  std::tie(split, expected_splitpos) = GetParam();

  std::bitset<7> splitpos = internal::GetSplitPos<8>(split);
  EXPECT_EQ(splitpos, expected_splitpos);
}

INSTANTIATE_TEST_SUITE_P(
    TestSplitPos, GetSplitPosTest,
    Values(
        // No Merge
        std::make_tuple(std::bitset<7>(0b1111111), std::bitset<7>(0b1111111)),
        // Partial Merge
        std::make_tuple(std::bitset<7>(0b1100011), std::bitset<7>(0b0101101)),
        // Full Merge
        std::make_tuple(std::bitset<7>(0b0), std::bitset<7>(0b0)),
        // Partial Merge 2
        std::make_tuple(std::bitset<7>(0b1010000), std::bitset<7>(0b0001010)),
        // Partial Merge 3
        std::make_tuple(std::bitset<7>(0b1110000), std::bitset<7>(0b0101010))));

}  // namespace
}  // namespace jax_sc_embedding
