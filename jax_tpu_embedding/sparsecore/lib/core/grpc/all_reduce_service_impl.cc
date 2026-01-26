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

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/concurrency/chain.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"
namespace jax_sc_embedding {
namespace rpc {
namespace {
void ReduceData(const AllReduceData& value, AllReduceData& accumulator) {
  DCHECK_EQ(accumulator.value_case(), value.value_case())
      << "Data type mismatch during reduction.";
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
  // The result counter is initialized to the number of consumers:
  // `threads_per_task_` for `GetResult()` calls, plus one additional
  // consumer for `GetLocalReducedValue()` if `num_tasks_ > 1`.
  state.results_counter = threads_per_task_ + (num_tasks_ > 1 ? 1 : 0);
  state.local_reduction_countdown =
      tsl::CountDownAsyncValueRef<tsl::Chain>(threads_per_task_);
  state.global_values_countdown =
      tsl::CountDownAsyncValueRef<tsl::Chain>(num_tasks_ - 1);
}

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  VLOG(2) << "Received data for sync_key: " << request->sync_key()
          << " from peer: " << context->peer();
  auto* reactor = context->DefaultReactor();
  tsl::CountDownAsyncValueRef<tsl::Chain> countdown;
  {
    absl::MutexLock lock(mutex_);
    auto [it, inserted] =
        all_reduce_state_map_.try_emplace(request->sync_key());
    AllReduceState& state = it->second;
    if (inserted) {
      // If a remote RPC arrives before any local InitializeOrUpdateState call,
      // the state is initialized with the remote request data.
      InitializeState(state, *request);
    } else {
      // State already exists.
      ReduceData(*request, state.local_data);
    }

    VLOG(2) << "Finished processing data for sync_key: " << request->sync_key()
            << " from peer: " << context->peer();
    countdown = state.global_values_countdown;
  }
  countdown.CountDown();
  reactor->Finish(grpc::Status::OK);
  return reactor;
}

bool AllReduceServiceImpl::InitializeOrUpdateState(int sync_key,
                                                   const AllReduceData& data) {
  tsl::profiler::TraceMe traceme(
      "AllReduceServiceImpl::InitializeOrUpdateState");
  tsl::CountDownAsyncValueRef<tsl::Chain> countdown;

  {
    absl::MutexLock lock(mutex_);
    auto result = all_reduce_state_map_.try_emplace(sync_key);
    AllReduceState& state = result.first->second;
    if (result.second) {
      // Initialize the state and wait for all other tasks.
      InitializeState(state, data);
    } else {
      // Update the state.
      ReduceData(data, state.local_data);
    }
    countdown = state.local_reduction_countdown;
  }

  return countdown.CountDown();
}

tsl::AsyncValueRef<AllReduceData> AllReduceServiceImpl::GetLocalReducedValue(
    int sync_key) {
  tsl::AsyncValueRef<AllReduceData> result =
      tsl::MakeUnconstructedAsyncValueRef<AllReduceData>();

  tsl::AsyncValueRef<tsl::Chain> local_reduction_done_av;

  {
    absl::MutexLock lock(mutex_);
    AllReduceState& state = all_reduce_state_map_.at(sync_key);
    local_reduction_done_av = state.local_reduction_countdown.AsRef();
  }

  local_reduction_done_av.AndThen([this, sync_key, result]() mutable {
    absl::MutexLock lock(mutex_);
    AllReduceState& state = all_reduce_state_map_.at(sync_key);
    result.emplace(state.local_data);
    if (--state.results_counter == 0) {
      all_reduce_state_map_.erase(sync_key);
    }
  });
  return result;
}

tsl::AsyncValueRef<AllReduceData> AllReduceServiceImpl::GetResult(
    int sync_key) {
  tsl::AsyncValueRef<AllReduceData> result =
      tsl::MakeUnconstructedAsyncValueRef<AllReduceData>();

  tsl::AsyncValueRef<tsl::Chain> local_reduction_done_av;
  tsl::AsyncValueRef<tsl::Chain> global_values_done_av;

  {
    absl::MutexLock lock(mutex_);
    AllReduceState& state = all_reduce_state_map_.at(sync_key);
    local_reduction_done_av = state.local_reduction_countdown.AsRef();
    global_values_done_av = state.global_values_countdown.AsRef();
  }

  tsl::RunWhenReady(
      absl::MakeConstSpan({local_reduction_done_av, global_values_done_av}),
      [this, sync_key, result]() mutable {
        absl::MutexLock lock(mutex_);
        AllReduceState& state = all_reduce_state_map_.at(sync_key);

        // Only set for non-concrete and non-error state.
        if (result.IsUnavailable()) {
          result.emplace(state.local_data);
        }

        if (--state.results_counter == 0) {
          all_reduce_state_map_.erase(sync_key);
        }
      });
  return result;
}

}  // namespace rpc
}  // namespace jax_sc_embedding
