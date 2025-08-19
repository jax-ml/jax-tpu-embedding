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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/barrier.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"

namespace jax_sc_embedding {

namespace testing_utils {

struct ReductionGroupState {
  const int expected_count;

  absl::Mutex mu;
  uint64_t current_result;
  absl::AnyInvocable<void()> on_done;

  std::unique_ptr<absl::Barrier> barrier;

  explicit ReductionGroupState(int count,
                               absl::AnyInvocable<void()> done_callback)
      : expected_count(count),
        current_result(0),
        on_done(std::move(done_callback)) {
    CHECK_GT(expected_count, 0) << "host_count must be positive.";
    barrier = std::make_unique<absl::Barrier>(expected_count);
  }

  uint64_t Participate(uint64_t local_val);
};

class FakeAllReduce : public AllReduceInterface {
 public:
  explicit FakeAllReduce(int host_count);
  ~FakeAllReduce() override = default;
  absl::StatusOr<bool> BlockingAllReduce(int sync_key,
                                         bool minibatching_required) override;
  absl::StatusOr<uint64_t> BlockingAllReduce(
      int sync_key, uint64_t minibatching_split) override;

 private:
  const int host_count_;
  absl::Mutex map_mutex_;
  absl::flat_hash_map<int, std::shared_ptr<ReductionGroupState>> groups_
      ABSL_GUARDED_BY(map_mutex_);

  void RemoveGroup(int sync_key);
};

}  // namespace testing_utils
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_FAKE_ALL_REDUCE_H_
