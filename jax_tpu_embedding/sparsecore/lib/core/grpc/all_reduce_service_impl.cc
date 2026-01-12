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
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"

#include <memory>
#include <optional>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/barrier.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace rpc {
namespace {
void ReduceData(const AllReduceData& value, AllReduceData& accumulator) {
  if (accumulator.has_bool_val()) {
    accumulator.set_bool_val(value.bool_val() || accumulator.bool_val());
  } else if (accumulator.has_uint64_val()) {
    accumulator.set_uint64_val(value.uint64_val() | accumulator.uint64_val());
  } else {
    LOG(FATAL) << "Unsupported data type for reduction: "
               << accumulator.value_case();
  }
}
}  // namespace

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  VLOG(2) << "Received data for sync_key: " << request->sync_key()
            << " from peer: " << context->peer();
  auto* reactor = context->DefaultReactor();
  absl::MutexLock lock(mutex_);

  // Wait for local state to be finalized.
  while (all_reduce_state_map_.find(request->sync_key()) ==
         all_reduce_state_map_.end()) {
    bool timeout =
        local_reduced_cv_.WaitWithTimeout(&mutex_, absl::Seconds(7200));
    if (timeout) {
      grpc::Status status = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                   "Timed out waiting for local value.");
      LOG(ERROR) << "Timeout while waiting for local value for sync_key: "
                 << request->sync_key() << " from peer: " << context->peer()
                 << " with status: "
                 << absl::Status(
                        static_cast<absl::StatusCode>(status.error_code()),
                        status.error_message());
      reactor->Finish(status);
      return reactor;
    }
  }

  // Get the state.
  auto it = all_reduce_state_map_.find(request->sync_key());
  CHECK(it != all_reduce_state_map_.end());

  // Combine remote value with local value.
  AllReduceData& local_data = it->second.local_data;

  if (local_data.value_case() != request->value_case()) {
    it->second.incoming_rpc_counter->DecrementCount();
    reactor->Finish(
        grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                     "Request data type does not match local data type."));
    return reactor;
  }

  ReduceData(*request, local_data);

  all_reduce_state_map_[request->sync_key()]
      .incoming_rpc_counter->DecrementCount();

  VLOG(2) << "Finished processing data for sync_key: " << request->sync_key()
            << " from peer: " << context->peer();
  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

absl::StatusOr<std::optional<AllReduceData>>
AllReduceServiceImpl::InitializeOrUpdateState(int sync_key,
                                              const AllReduceData& data) {
  tsl::profiler::TraceMe traceme(
      "AllReduceServiceImpl::InitializeOrUpdateState");
  absl::MutexLock lock(mutex_);
  auto it = all_reduce_state_map_.find(sync_key);

  if (it == all_reduce_state_map_.end()) {
    // Initialize the state and wait for all other tasks.
    all_reduce_state_map_[sync_key] = {
        .local_data = data,
        .local_contributions_counter =
            std::make_unique<absl::BlockingCounter>(threads_per_task_ - 1),
        .results_counter =
            std::make_unique<absl::BlockingCounter>(threads_per_task_),
        .global_results_barrier =
            std::make_unique<absl::Barrier>(threads_per_task_),
        .incoming_rpc_counter =
            std::make_unique<absl::BlockingCounter>(num_tasks_ - 1),
    };
    auto* local_contributions_counter =
        all_reduce_state_map_[sync_key].local_contributions_counter.get();

    // Wait without mutex to avoid deadlock.
    mutex_.unlock();
    local_contributions_counter->Wait();

    // Lock to update CV.
    mutex_.lock();
    local_reduced_cv_.SignalAll();

    return all_reduce_state_map_[sync_key].local_data;
  } else {
    // Update the state.
    ReduceData(data, it->second.local_data);
    it->second.local_contributions_counter->DecrementCount();
  }
  return std::nullopt;
}

void AllReduceServiceImpl::WaitIncomingRPCs(int sync_key) {
  tsl::profiler::TraceMe traceme("AllReduceServiceImpl::WaitIncomingRPCs");
  absl::BlockingCounter* incoming_rpc_counter = nullptr;
  {
    absl::MutexLock lock(mutex_);
    incoming_rpc_counter =
        all_reduce_state_map_.at(sync_key).incoming_rpc_counter.get();
  }

  VLOG(2) << "Waiting for incoming RPCs for sync_key: " << sync_key;

  // Wait for the counter outside of the mutex lock to prevent deadlock.
  CHECK(incoming_rpc_counter != nullptr);
  incoming_rpc_counter->Wait();
}

void AllReduceServiceImpl::WaitResults(int sync_key) {
  tsl::profiler::TraceMe traceme("AllReduceServiceImpl::WaitResults");
  absl::Barrier* global_results_barrier = nullptr;
  {
    absl::MutexLock lock(mutex_);
    global_results_barrier =
        all_reduce_state_map_.at(sync_key).global_results_barrier.get();
  }
  VLOG(2) << "Waiting for global results for sync_key: " << sync_key;
  CHECK(global_results_barrier != nullptr);
  global_results_barrier->Block();
}

absl::StatusOr<AllReduceData> AllReduceServiceImpl::GetResult(int sync_key) {
  absl::MutexLock lock(mutex_);
  auto& state = all_reduce_state_map_[sync_key];
  AllReduceData result = state.local_data;
  if (state.results_counter->DecrementCount()) {
    all_reduce_state_map_.erase(sync_key);
  }
  VLOG(2) << "GetResult for sync_key: " << sync_key
            << " result: " << result.DebugString();
  return result;
}

}  // namespace rpc
}  // namespace jax_sc_embedding
