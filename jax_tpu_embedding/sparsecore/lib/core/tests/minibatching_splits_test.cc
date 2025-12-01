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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/minibatching_splits_impl.h"

namespace jax_sc_embedding {
namespace {

using ::testing::UnorderedElementsAreArray;
using ::testing::Values;

class ComputeMinibatchingSplitTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<int32_t>, int32_t, std::bitset<7>>> {};

TEST_P(ComputeMinibatchingSplitTest, TestSplits) {
  auto [unique_ids_per_bucket, max_ids_per_partition, expected_split] =
      GetParam();

  EXPECT_EQ(internal::ComputeMinibatchingSplit<8>(
                absl::MakeSpan(unique_ids_per_bucket), max_ids_per_partition),
            expected_split);
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

class MergeBucketsTest : public ::testing::TestWithParam<
                             std::tuple<std::vector<int32_t>, std::bitset<7>,
                                        std::vector<std::pair<int, int>>>> {};

TEST_P(MergeBucketsTest, TestMergeBuckets) {
  auto [unique_ids_per_bucket, split_pos, expected_merged_buckets] = GetParam();
  std::vector<std::pair<int, int>> merged_buckets;

  auto merge_fn = [&](int left, int right) {
    merged_buckets.push_back(std::make_pair(left, right));
  };
  internal::MergeBuckets<8>(split_pos, merge_fn);

  EXPECT_THAT(merged_buckets,
              UnorderedElementsAreArray(expected_merged_buckets));
}

INSTANTIATE_TEST_SUITE_P(
    TestMergeBuckets, MergeBucketsTest,
    Values(
        // No Merge
        std::make_tuple(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8},
                        std::bitset<7>(0b1111111),
                        std::vector<std::pair<int, int>>{}),
        // Partial Merge
        std::make_tuple(std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8},
                        std::bitset<7>(0b0101101),
                        std::vector<std::pair<int, int>>{
                            {2, 3}, {0, 2}, {0, 4}}),
        // Full Merge
        std::make_tuple(
            std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}, std::bitset<7>(0b0),
            std::vector<std::pair<int, int>>{
                {0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {4, 6}, {0, 4}}),
        // Partial Merge 2
        std::make_tuple(std::vector<int32_t>{50, 50, 20, 80, 10, 10, 70, 10},
                        std::bitset<7>(0b0001010),
                        std::vector<std::pair<int, int>>{
                            {0, 1}, {4, 5}, {0, 2}, {4, 6}, {0, 4}}),
        // Partial Merge 3
        std::make_tuple(std::vector<int32_t>{90, 10, 90, 10, 90, 10, 90, 10},
                        std::bitset<7>(0b0101010),
                        std::vector<std::pair<int, int>>{
                            {0, 1}, {4, 5}, {0, 2}, {0, 4}})));

}  // namespace
}  // namespace jax_sc_embedding
