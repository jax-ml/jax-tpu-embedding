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
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"

#include <bitset>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

TEST(PartitionedCooTensorsTest, Empty) {
  PartitionedCooTensors tensors(/*reserve_count=*/0, /*sc_count=*/1,
                                /*global_sc_count=*/1,
                                /*bucket_count=*/1);
  tensors.FillRemainingScBuckets();
  EXPECT_THAT(tensors(0, 0), SizeIs(0));
  EXPECT_EQ(tensors.Size(0), 0);
}

TEST(PartitionedCooTensorsTest, SingleScSingleBucket) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/1,
                                /*global_sc_count=*/1,
                                /*bucket_count=*/1);
  CooFormat coo1(1, 2, 3.0);
  CooFormat coo2(4, 5, 6.0);
  tensors.Add(0, 0, coo1);
  tensors.Add(0, 0, coo2);
  tensors.FillRemainingScBuckets();

  EXPECT_THAT(tensors(0, 0), ElementsAre(coo1, coo2));
  EXPECT_EQ(tensors.Size(0), 2);
}

TEST(PartitionedCooTensorsTest, MultiScSingleBucket) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/2,
                                /*bucket_count=*/1);
  CooFormat coo1(1, 2, 3.0);
  CooFormat coo2(4, 5, 6.0);
  CooFormat coo3(7, 8, 9.0);

  tensors.Add(0, 0, coo1);
  tensors.Add(1, 0, coo2);
  tensors.Add(1, 0, coo3);
  tensors.FillRemainingScBuckets();

  EXPECT_THAT(tensors(0, 0), ElementsAre(coo1));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo2, coo3));
  EXPECT_EQ(tensors.Size(0), 1);
  EXPECT_EQ(tensors.Size(1), 2);
}

TEST(PartitionedCooTensorsTest, MultiScMultiBucket) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/2,
                                /*global_sc_count=*/2,
                                /*bucket_count=*/2);
  CooFormat coo1(1, 1, 1.0);
  CooFormat coo2(2, 2, 2.0);
  CooFormat coo3(3, 3, 3.0);
  CooFormat coo4(4, 4, 4.0);
  CooFormat coo5(5, 5, 5.0);

  tensors.Add(0, 0, coo1);
  tensors.Add(0, 1, coo2);
  tensors.Add(1, 0, coo3);
  tensors.Add(1, 0, coo4);
  tensors.Add(1, 1, coo5);
  tensors.FillRemainingScBuckets();

  EXPECT_THAT(tensors(0, 0), ElementsAre(coo1));
  EXPECT_THAT(tensors(0, 1), ElementsAre(coo2));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo3, coo4));
  EXPECT_THAT(tensors(1, 1), ElementsAre(coo5));
  EXPECT_EQ(tensors.Size(0), 2);
  EXPECT_EQ(tensors.Size(1), 3);
}

TEST(PartitionedCooTensorsTest, MergeBuckets) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/2,
                                /*global_sc_count=*/2,
                                /*bucket_count=*/2);
  CooFormat coo1(1, 1, 1.0);
  CooFormat coo2(2, 2, 2.0);
  CooFormat coo3(3, 3, 3.0);
  CooFormat coo4(4, 4, 4.0);
  CooFormat coo5(5, 5, 5.0);

  tensors.Add(0, 0, coo1);
  tensors.Add(0, 1, coo2);
  tensors.Add(1, 0, coo3);
  tensors.Add(1, 0, coo4);
  tensors.Add(1, 1, coo5);

  tensors.MergeAll<2>();

  EXPECT_EQ(tensors.GetNumMinibatches(), 1);
  EXPECT_THAT(tensors(0, 0), ElementsAre(coo2, coo1));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo3, coo4, coo5));
  EXPECT_EQ(tensors.Size(0), 2);
  EXPECT_EQ(tensors.Size(1), 3);
}

TEST(PartitionedCooTensorsTest, PartialMerge) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/2,
                                /*global_sc_count=*/2,
                                /*bucket_count=*/4);
  CooFormat coo1(1, 1, 1.0);
  CooFormat coo2(2, 2, 2.0);
  CooFormat coo3(3, 3, 3.0);
  CooFormat coo4(4, 4, 4.0);
  CooFormat coo5(5, 5, 5.0);
  CooFormat coo6(6, 6, 6.0);
  CooFormat coo7(7, 7, 7.0);
  CooFormat coo8(8, 8, 8.0);

  tensors.Add(0, 0, coo1);
  tensors.Add(0, 1, coo2);
  tensors.Add(0, 2, coo3);
  tensors.Add(0, 3, coo4);
  tensors.Add(1, 0, coo5);
  tensors.Add(1, 1, coo6);
  tensors.Add(1, 2, coo7);
  tensors.Add(1, 3, coo8);

  // This split ({2,3}) results in minibatches {0,1,2} and {3}.
  tensors.Merge<4>(std::bitset<3>(0b010));

  EXPECT_EQ(tensors.GetNumMinibatches(), 2);
  EXPECT_THAT(tensors(0, 0), ElementsAre(coo2, coo1, coo3));
  EXPECT_THAT(tensors(0, 1), ElementsAre(coo4));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo6, coo5, coo7));
  EXPECT_THAT(tensors(1, 1), ElementsAre(coo8));
  EXPECT_EQ(tensors.Size(0), 4);
  EXPECT_EQ(tensors.Size(1), 4);
}

TEST(PartitionedCooTensorsTest, NoMerge) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/2,
                                /*global_sc_count=*/2,
                                /*bucket_count=*/4);
  CooFormat coo1(1, 1, 1.0);
  CooFormat coo2(2, 2, 2.0);
  CooFormat coo3(3, 3, 3.0);
  CooFormat coo4(4, 4, 4.0);
  CooFormat coo5(5, 5, 5.0);
  CooFormat coo6(6, 6, 6.0);
  CooFormat coo7(7, 7, 7.0);
  CooFormat coo8(8, 8, 8.0);

  tensors.Add(0, 0, coo1);
  tensors.Add(0, 1, coo2);
  tensors.Add(0, 2, coo3);
  tensors.Add(0, 3, coo4);
  tensors.Add(1, 0, coo5);
  tensors.Add(1, 1, coo6);
  tensors.Add(1, 2, coo7);
  tensors.Add(1, 3, coo8);

  tensors.Merge<4>(std::bitset<3>(0b111));

  EXPECT_EQ(tensors.GetNumMinibatches(), 4);
  EXPECT_THAT(tensors(0, 0), ElementsAre(coo1));
  EXPECT_THAT(tensors(0, 1), ElementsAre(coo2));
  EXPECT_THAT(tensors(0, 2), ElementsAre(coo3));
  EXPECT_THAT(tensors(0, 3), ElementsAre(coo4));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo5));
  EXPECT_THAT(tensors(1, 1), ElementsAre(coo6));
  EXPECT_THAT(tensors(1, 2), ElementsAre(coo7));
  EXPECT_THAT(tensors(1, 3), ElementsAre(coo8));
  EXPECT_EQ(tensors.Size(0), 4);
  EXPECT_EQ(tensors.Size(1), 4);
}

}  // namespace
}  // namespace jax_sc_embedding
