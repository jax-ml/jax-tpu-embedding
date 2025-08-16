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
#include "jax_tpu_embedding/sparsecore/lib/core/fake_all_reduce.h"

#include <cstdint>
#include <memory>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"

namespace jax_sc_embedding {

namespace testing_utils {
namespace internal {
enum class BarrierStage { kArriving, kDeparting };

struct ReductionGroupState {
  absl::Mutex mu;
  absl::CondVar cv;
  const int expected_count;

  BarrierStage stage ABSL_GUARDED_BY(mu) = BarrierStage::kArriving;
  // Counts threads currently in the barrier
  int arrived_count ABSL_GUARDED_BY(mu) = 0;
  int departed_count ABSL_GUARDED_BY(mu) = 0;
  uint64_t current_result ABSL_GUARDED_BY(mu);

  explicit ReductionGroupState(int count) : expected_count(count) {
    CHECK_GT(expected_count, 0) << "host_count must be positive.";
  }

  uint64_t Participate(uint64_t local_val) {
    absl::MutexLock lock(&mu);  // NOLINT (b/438618768)

    // Wait for the round to be in the Arriving stage.
    while (stage != BarrierStage::kArriving) {
      cv.Wait(&mu);
    }

    // Phase 1: Arriving and Accumulating
    if (arrived_count == 0) {
      // First thread of a new reduction round.
      current_result = 0ULL;  // Identity for bitwise OR
      departed_count = 0;     // Reset for this round's departure
    }
    current_result |= local_val;
    ++arrived_count;

    if (arrived_count == expected_count) {
      // Last thread to arrive. Transition to Departing stage.
      stage = BarrierStage::kDeparting;
      cv.SignalAll();
    } else {
      // Wait for all expected_count threads to arrive.
      while (stage == BarrierStage::kArriving) {
        cv.Wait(&mu);
      }
    }
    // Sync Point 1: All threads have arrived
    // The accumulated current_result is ready.

    uint64_t result_to_return = current_result;

    // Phase 2: Departing
    ++departed_count;
    if (departed_count == expected_count) {
      // Last thread to depart. Reset for the next round.
      stage = BarrierStage::kArriving;
      arrived_count = 0;  // Reset for next round's arrival
      cv.SignalAll();     // Wake up any threads waiting to start the *next*
                          // round.
    } else {
      // Wait for all threads to depart this round.
      while (stage == BarrierStage::kDeparting) {
        cv.Wait(&mu);
      }
    }
    // Sync Point 2: All threads have departed
    return result_to_return;
  }
};
}  // namespace internal

// Returns a callback function that performs an AllReduce operation using
// bitwise OR. Boolean values should be passed as 0ULL (false) or 1ULL (true).
// The callback is thread-safe and reusable across multiple reduction rounds.
AllReduceCallback CreateFakeAllReduceCallback(int host_count) {
  auto map_mutex = std::make_shared<absl::Mutex>();
  auto groups = std::make_shared<absl::flat_hash_map<
      int, std::shared_ptr<internal::ReductionGroupState>>>();

  return [map_mutex, groups, host_count](int sync_key,
                                         uint64_t local_val) -> uint64_t {
    std::shared_ptr<internal::ReductionGroupState> group;
    {
      absl::MutexLock map_lock(map_mutex.get());  // NOLINT (b/438618768)
      auto& group_ptr = (*groups)[sync_key];
      if (!group_ptr) {
        group_ptr = std::make_shared<internal::ReductionGroupState>(host_count);
      }
      group = group_ptr;
    }
    return group->Participate(local_val);
  };
}
}  // namespace testing_utils
}  // namespace jax_sc_embedding
