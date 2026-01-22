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

void AllReduceServiceImpl::InitializeState(AllReduceState& state,
                                           const AllReduceData& data) {
  state.local_data = data;
  state.local_contributions_counter =
      std::make_unique<absl::BlockingCounter>(threads_per_task_);
  state.results_counter =
      std::make_unique<absl::BlockingCounter>(threads_per_task_);
  state.global_results_barrier =
      std::make_unique<absl::Barrier>(threads_per_task_);
  state.incoming_rpc_counter =
      std::make_unique<absl::BlockingCounter>(num_tasks_ - 1);
}

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  VLOG(2) << "Received data for sync_key: " << request->sync_key()
          << " from peer: " << context->peer();
  auto* reactor = context->DefaultReactor();
  absl::MutexLock lock(mutex_);

  auto [it, inserted] = all_reduce_state_map_.try_emplace(request->sync_key());
  AllReduceState& state = it->second;
  if (inserted) {
    // If a remote RPC arrives before any local InitializeOrUpdateState call,
    // the state is initialized with the remote request data.
    InitializeState(state, *request);
  } else {
    // State already exists.
    if (state.local_data.value_case() != request->value_case()) {
      state.incoming_rpc_counter->DecrementCount();
      reactor->Finish(
          grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                       "Request data type does not match local data type."));
      return reactor;
    }

    ReduceData(*request, state.local_data);
  }

  VLOG(2) << "Finished processing data for sync_key: " << request->sync_key()
            << " from peer: " << context->peer();
  state.incoming_rpc_counter->DecrementCount();
  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

absl::StatusOr<std::optional<AllReduceData>>
AllReduceServiceImpl::InitializeOrUpdateState(int sync_key,
                                              const AllReduceData& data) {
  tsl::profiler::TraceMe traceme(
      "AllReduceServiceImpl::InitializeOrUpdateState");
  absl::MutexLock lock(mutex_);
  auto result = all_reduce_state_map_.try_emplace(sync_key);
  AllReduceState& state = result.first->second;
  if (result.second) {
    // Initialize the state and wait for all other tasks.
    InitializeState(state, data);
  } else {
    if (state.local_data.value_case() != data.value_case()) {
      state.local_contributions_counter->DecrementCount();
      return absl::InvalidArgumentError(
          "Local data type does not match existing state data type.");
    }
    // Update the state.
    ReduceData(data, state.local_data);
  }

  if (state.local_contributions_counter->DecrementCount()) {
    return state.local_data;
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
