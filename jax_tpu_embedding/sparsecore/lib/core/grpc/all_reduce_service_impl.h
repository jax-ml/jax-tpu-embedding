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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/concurrency/chain.h"  // from @xla
#include "tsl/platform/env.h"  // from @tsl

namespace jax_sc_embedding {
namespace rpc {

// Implementation of the gRPC AllReduce service. This class manages the state
// for multiple concurrent all-reduce operations, identified by a `sync_key`.
class AllReduceServiceImpl
    : public AllReduceGrpcService::CallbackService,
      public std::enable_shared_from_this<AllReduceServiceImpl> {
  struct AllReduceState {
    AllReduceState() = default;
    AllReduceState(AllReduceState&&) = default;
    AllReduceState& operator=(AllReduceState&&) = default;

    AllReduceData local_data;
    // Counter for all local threads to retrieve the results and delete
    // this state from the map when done.
    int results_counter;
    tsl::CountDownAsyncValueRef<tsl::Chain> local_reduction_countdown;
    tsl::CountDownAsyncValueRef<tsl::Chain> global_values_countdown;

    // Track which peers have contributed data for diagnostics.
    std::vector<bool> received_from_task;
    int local_received_count = 0;
  };

 public:
  explicit AllReduceServiceImpl(int task_id, int num_tasks,
                                int threads_per_task = 1,
                                tsl::Env* env = tsl::Env::Default())
      : task_id_(task_id),
        num_tasks_(num_tasks),
        threads_per_task_(threads_per_task),
        env_(env) {}

  // Called by remote peers. Returns this server's local value for the sync_key.
  ::grpc::ServerUnaryReactor* ContributeData(
      ::grpc::CallbackServerContext* context, const AllReduceData* request,
      AllReduceResponse* response) override;

  // Method to register the local data for a given sync_key. Called by the local
  // client. Returns true if this is the last local thread to contribute.
  bool InitializeOrUpdateState(int sync_key, const AllReduceData& data);

  // Gets locally reduced value.
  tsl::AsyncValueRef<AllReduceData> GetLocalReducedValue(int sync_key);

  // Gets locally and globally reduced result.
  tsl::AsyncValueRef<AllReduceData> GetResult(int sync_key);

  // Only for testing.
  int GetActiveStatesCount() const;

  enum class WaitType { kLocalReduction, kGlobalValues };

 private:
  // Helper to initialize AllReduceState.
  void InitializeState(AllReduceState& state, const AllReduceData& data);

  // Records a completed sync key to the history. If history capacity is
  // exceeded, it wraps around and overwrites the oldest entries.
  void RecordCompletedSyncKey(int sync_key)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    completed_sync_keys_[completed_sync_keys_head_] = sync_key;
    completed_sync_keys_head_ =
        (completed_sync_keys_head_ + 1) % kCompletedSyncKeysHistorySize;
    completed_sync_keys_size_ =
        std::min(completed_sync_keys_size_ + 1, kCompletedSyncKeysHistorySize);
  }

  // Returns true if the sync key is present in the completed history.
  // Performs a linear search limited to the populated portion of the buffer.
  // Linear search on 4KB contiguous array is faster than hash set lookup
  // due to L1 cache locality and compiler SIMD vectorization.
  bool IsSyncKeyCompleted(int sync_key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    absl::Span<const int> populated_keys(completed_sync_keys_.data(),
                                         completed_sync_keys_size_);
    return absl::c_find(populated_keys, sync_key) != populated_keys.end();
  }

  // Decrements the results counter for the state. If it reaches 0, records
  // the sync key as completed and erases the state from the map.
  void CleanupStateIfDone(int sync_key, AllReduceState& state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (--state.results_counter == 0) {
      RecordCompletedSyncKey(sync_key);
      all_reduce_state_map_.erase(sync_key);
    }
  }

  int task_id_;
  int num_tasks_;
  // Number of threads (within the same process) that will participate in the
  // all-reduce operation.
  int threads_per_task_;
  tsl::Env* env_;

  // Schedules a recurring watchdog warning every 5 minutes until the
  // AsyncValue becomes available.
  void ScheduleWatchdog(int sync_key, WaitType wait_type,
                        tsl::AsyncValueRef<tsl::Chain> av, int elapsed_minutes);

  // Callback executed by the scheduled watchdog closure.
  void WatchdogCallback(int sync_key, WaitType wait_type,
                        tsl::AsyncValueRef<tsl::Chain> av, int elapsed_minutes);

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<int, AllReduceState> all_reduce_state_map_
      ABSL_GUARDED_BY(mutex_);

  // History size of 1024 is chosen to be 2x larger than the maximum possible
  // pipeline depth (prefetch queue size of 256 steps / 512 sync keys), ensuring
  // that lagging threads retrying old steps will always hit the history.
  static constexpr int kCompletedSyncKeysHistorySize = 1024;
  // Circular buffer for tracking completed sync keys. A flat array is used
  // instead of a hash set to avoid heap allocations in the critical path and
  // to leverage cache locality and SIMD vectorization for the linear search.
  std::array<int, kCompletedSyncKeysHistorySize> completed_sync_keys_
      ABSL_GUARDED_BY(mutex_) = {};
  int completed_sync_keys_head_ ABSL_GUARDED_BY(mutex_) = 0;
  int completed_sync_keys_size_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_ALL_REDUCE_SERVICE_IMPL_H_
