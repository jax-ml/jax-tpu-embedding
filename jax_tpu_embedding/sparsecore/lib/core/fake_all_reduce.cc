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

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace jax_sc_embedding {

namespace testing_utils {

uint64_t ReductionGroupState::Participate(uint64_t local_val) {
  {
    absl::MutexLock lock(&mu);
    current_result |= local_val;
  }

  // Wait for all threads to accumulate the results.
  if (barrier->Block()) {
    on_done();  // Removes this group from the map.
  }

  return current_result;
}

FakeAllReduce::FakeAllReduce(int host_count) : host_count_(host_count) {}

absl::StatusOr<bool> FakeAllReduce::BlockingAllReduce(
    int sync_key, bool minibatching_required) {
  return BlockingAllReduce(sync_key,
                           static_cast<uint64_t>(minibatching_required));
}

absl::StatusOr<uint64_t> FakeAllReduce::BlockingAllReduce(
    int sync_key, uint64_t minibatching_split) {
  std::shared_ptr<ReductionGroupState> group;
  {
    absl::MutexLock map_lock(&map_mutex_);
    auto& group_ptr = groups_[sync_key];
    if (group_ptr == nullptr) {
      group_ptr = std::make_shared<ReductionGroupState>(
          host_count_, [this, sync_key]() { this->RemoveGroup(sync_key); });
    }
    group = group_ptr;
  }
  return group->Participate(minibatching_split);
}

void FakeAllReduce::RemoveGroup(int sync_key) {
  absl::MutexLock map_lock(&map_mutex_);
  groups_.erase(sync_key);
}

}  // namespace testing_utils
}  // namespace jax_sc_embedding
