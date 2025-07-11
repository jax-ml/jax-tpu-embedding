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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAreArray;

TEST(InputPreprocessingUtilTest, MergeStats) {
  SparseDenseMatmulInputStats stats1;
  SparseDenseMatmulInputStats stats2;

  // Populate stats1
  stats1.max_ids_per_partition["table_0"].resize(4);
  stats1.max_ids_per_partition["table_0"] << 10, 20, 30, 40;
  stats1.max_ids_per_partition["table_1"].resize(4);
  stats1.max_ids_per_partition["table_1"] << 1, 2, 3, 4;
  stats1.max_unique_ids_per_partition["table_0"].resize(4);
  stats1.max_unique_ids_per_partition["table_0"] << 5, 6, 7, 8;
  stats1.max_unique_ids_per_partition["table_1"].resize(4);
  stats1.max_unique_ids_per_partition["table_1"] << 10, 11, 12, 13;
  stats1.required_buffer_sizes["table_0"].resize(4);
  stats1.required_buffer_sizes["table_0"] << 100, 101, 102, 103;
  stats1.required_buffer_sizes["table_1"].resize(4);
  stats1.required_buffer_sizes["table_1"] << 200, 201, 202, 203;

  // Populate stats2
  stats2.max_ids_per_partition["table_0"].resize(4);
  stats2.max_ids_per_partition["table_0"] << 12, 13, 14, 15;
  stats2.max_ids_per_partition["table_2"].resize(4);
  stats2.max_ids_per_partition["table_2"] << 30, 31, 32, 33;
  stats2.max_unique_ids_per_partition["table_0"].resize(4);
  stats2.max_unique_ids_per_partition["table_0"] << 6, 7, 8, 9;
  stats2.max_unique_ids_per_partition["table_2"].resize(4);
  stats2.max_unique_ids_per_partition["table_2"] << 12, 13, 14, 15;
  stats2.required_buffer_sizes["table_0"].resize(4);
  stats2.required_buffer_sizes["table_0"] << 120, 121, 122, 123;
  stats2.required_buffer_sizes["table_2"].resize(4);
  stats2.required_buffer_sizes["table_2"] << 300, 301, 302, 303;

  stats1.merge(stats2);

  // The merged stats should contain the element-wise maximum of the
  // corresponding stats.
  EXPECT_THAT(stats1.max_ids_per_partition["table_0"],
              ElementsAreArray({12, 20, 30, 40}));
  EXPECT_THAT(stats1.max_ids_per_partition["table_1"],
              ElementsAreArray({1, 2, 3, 4}));
  EXPECT_THAT(stats1.max_ids_per_partition["table_2"],
              ElementsAreArray({30, 31, 32, 33}));

  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_0"],
              ElementsAreArray({6, 7, 8, 9}));
  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_1"],
              ElementsAreArray({10, 11, 12, 13}));
  EXPECT_THAT(stats1.max_unique_ids_per_partition["table_2"],
              ElementsAreArray({12, 13, 14, 15}));

  EXPECT_THAT(stats1.required_buffer_sizes["table_0"],
              ElementsAreArray({120, 121, 122, 123}));
  EXPECT_THAT(stats1.required_buffer_sizes["table_1"],
              ElementsAreArray({200, 201, 202, 203}));
  EXPECT_THAT(stats1.required_buffer_sizes["table_2"],
              ElementsAreArray({300, 301, 302, 303}));
}

}  // namespace
}  // namespace jax_sc_embedding
