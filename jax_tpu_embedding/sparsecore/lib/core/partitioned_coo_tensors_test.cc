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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"

namespace jax_sc_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

TEST(PartitionedCooTensorsTest, Empty) {
  PartitionedCooTensors tensors(/*reserve_count=*/0, /*sc_count=*/1,
                                /*bucket_count=*/1);
  tensors.FillRemainingScBuckets();
  EXPECT_THAT(tensors(0, 0), SizeIs(0));
  EXPECT_EQ(tensors.Size(0), 0);
}

TEST(PartitionedCooTensorsTest, SingleScSingleBucket) {
  PartitionedCooTensors tensors(/*reserve_count=*/10, /*sc_count=*/1,
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

  tensors.MergeAll();

  EXPECT_EQ(tensors.GetBucketCount(), 1);
  EXPECT_THAT(tensors(0, 0), ElementsAre(coo1, coo2));
  EXPECT_THAT(tensors(1, 0), ElementsAre(coo3, coo4, coo5));
  EXPECT_EQ(tensors.Size(0), 2);
  EXPECT_EQ(tensors.Size(1), 3);
}

}  // namespace
}  // namespace jax_sc_embedding
