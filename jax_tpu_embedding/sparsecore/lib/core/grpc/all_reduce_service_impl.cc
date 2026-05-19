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

#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.pb.h" // from internal
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/concurrency/chain.h"  // from @xla
#include "tsl/platform/env.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace rpc {
namespace {

constexpr absl::Duration kWatchdogTimeout = absl::Minutes(5);

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

// Schedules a watchdog for a synchronization operation. The watchdog will log a
// WARNING if the synchronization is not complete after the specified interval.
void AllReduceServiceImpl::ScheduleWatchdog(int sync_key, WaitType wait_type,
                                            tsl::AsyncValueRef<tsl::Chain> av,
                                            int elapsed_minutes) {
  // NOTE: It is safe to capture `this` here, since `AllReduceServiceImpl` will
  // outlive the watchdog.
  env_->SchedClosureAfter(
      absl::ToInt64Microseconds(kWatchdogTimeout),
      [this, sync_key, wait_type, av = std::move(av),
       elapsed_minutes]() mutable {
        // If the synchronization is finished or failed, we stop the watchdog.
        if (av.IsAvailable() || av.IsError()) return;

        // Collect information about which hosts or threads are still missing.
        auto get_missing_info = [&]() {
          absl::MutexLock lock(mutex_);
          auto it = all_reduce_state_map_.find(sync_key);
          if (it == all_reduce_state_map_.end()) return std::string();

          if (wait_type == WaitType::kLocalReduction) {
            return absl::StrCat(
                " (missing ",
                threads_per_task_ - it->second.local_received_count,
                " threads)");
          }

          std::vector<int> missing;
          for (int i = 0; i < num_tasks_; ++i) {
            if (!it->second.received_from_task[i]) missing.push_back(i);
          }
          return absl::StrCat(" (missing ranks: ", absl::StrJoin(missing, ", "),
                              ")");
        };

        std::string missing_info = get_missing_info();
        std::string wait_type_str = (wait_type == WaitType::kLocalReduction)
                                        ? "local reduction"
                                        : "global values";

        LOG(WARNING) << "Host is waiting for more than " << elapsed_minutes
                     << " minutes for " << wait_type_str
                     << " for sync_key: " << sync_key
                     << " at rank: " << task_id_ << missing_info;

        // Schedule the next watchdog check.
        ScheduleWatchdog(sync_key, wait_type, std::move(av),
                         /*elapsed_minutes=*/elapsed_minutes + 5);
      });
}

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
  state.received_from_task.assign(num_tasks_, false);
  state.received_from_task[task_id_] = true;
  state.local_received_count = 0;

  // NOTE: It is cheap to create new watchdogs since minibatching operation
  // would be infrequent.

  // Start watchdogs to monitor synchronization progress.
  ScheduleWatchdog(data.sync_key(), WaitType::kLocalReduction,
                   state.local_reduction_countdown.AsRef(),
                   /*elapsed_minutes=*/5);
  if (num_tasks_ > 1) {
    ScheduleWatchdog(data.sync_key(), WaitType::kGlobalValues,
                     state.global_values_countdown.AsRef(),
                     /*elapsed_minutes=*/5);
  }
}

absl::Status AllReduceServiceImpl::ContributeDataInternal(
    const AllReduceData& request) {
  if (request.src_rank() < 0 || request.src_rank() >= num_tasks_ ||
      request.src_rank() == task_id_) {
    return absl::InvalidArgumentError("Invalid src_rank");
  }
  tsl::CountDownAsyncValueRef<tsl::Chain> countdown;
  {
    absl::MutexLock lock(mutex_);
    if (max_completed_sync_key_.has_value() &&
        request.sync_key() <= *max_completed_sync_key_) {
      VLOG(1) << "Task " << task_id_
              << " ignoring late RPC for sync_key: " << request.sync_key()
              << " because max completed is " << *max_completed_sync_key_;
      return absl::OkStatus();
    }
    auto [it, inserted] = all_reduce_state_map_.try_emplace(request.sync_key());
    AllReduceState& state = it->second;
    if (inserted) {
      // If a remote RPC arrives before any local InitializeOrUpdateState call,
      // the state is initialized with the remote request data.
      InitializeState(state, request);
    } else {
      if (state.received_from_task[request.src_rank()]) {
        VLOG(1) << "Task " << task_id_
                << " ignoring duplicate RPC for sync_key: "
                << request.sync_key() << " from task: " << request.src_rank();
        return absl::OkStatus();
      }
      // State already exists.
      ReduceData(request, state.local_data);
    }

    countdown = state.global_values_countdown;
    state.received_from_task[request.src_rank()] = true;
  }
  countdown.CountDown();
  return absl::OkStatus();
}

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  VLOG(2) << "Task " << task_id_
          << " received data for sync_key: " << request->sync_key()
          << " from peer: " << context->peer()
          << " claiming src_rank: " << request->src_rank();
  auto* reactor = context->DefaultReactor();
  absl::Status s = ContributeDataInternal(*request);
  if (!s.ok()) {
    LOG(WARNING) << "Task " << task_id_
                 << " ignoring RPC with invalid/self src_rank: "
                 << request->src_rank()
                 << " for sync_key: " << request->sync_key()
                 << " error: " << s.message();
    grpc::StatusCode code = grpc::StatusCode::UNKNOWN;
    if (s.code() == absl::StatusCode::kInvalidArgument) {
      code = grpc::StatusCode::INVALID_ARGUMENT;
    }
    reactor->Finish(grpc::Status(code, std::string(s.message())));
  } else {
    reactor->Finish(grpc::Status::OK);
  }
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
    state.local_received_count++;
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
    AllReduceData local_data = state.local_data;
    local_data.set_src_rank(task_id_);
    result.emplace(local_data);
    EraseStateIfCompleted(sync_key, state);
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

        EraseStateIfCompleted(sync_key, state);
      });
  return result;
}

void AllReduceServiceImpl::EraseStateIfCompleted(int sync_key,
                                                 AllReduceState& state) {
  if (--state.results_counter == 0) {
    all_reduce_state_map_.erase(sync_key);
    max_completed_sync_key_ =
        std::max(max_completed_sync_key_.value_or(sync_key), sync_key);
  }
}

int AllReduceServiceImpl::GetActiveStatesCount() const {
  absl::MutexLock lock(mutex_);
  return all_reduce_state_map_.size();
}

}  // namespace rpc
}  // namespace jax_sc_embedding
