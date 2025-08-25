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

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/server_callback.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.proto.h"

namespace jax_sc_embedding {

::grpc::ServerUnaryReactor* AllReduceServiceImpl::ContributeData(
    ::grpc::CallbackServerContext* context, const AllReduceData* request,
    AllReduceResponse* response) {
  auto* reactor = context->DefaultReactor();
  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)

  // Wait for local value.
  while (all_reduce_state_map_.find(request->sync_key()) ==
         all_reduce_state_map_.end()) {
    local_data_cv_.Wait(&mutex_);
  }
  AllReduceData& local_data =
      all_reduce_state_map_[request->sync_key()].local_data;

  if (local_data.value_case() != request->value_case()) {
    reactor->Finish(absl::InvalidArgumentError(
        "Request data type does not match local data type."));
    return reactor;
  }

  LOG(INFO) << "ContributeData: Reducing for sync_key: " << request->sync_key()
            << " on " << task_id_ << " from " << request->src_rank();

  // Reduce local value.
  if (local_data.has_bool_val()) {
    local_data.set_bool_val(request->bool_val() || local_data.bool_val());
  } else if (local_data.has_uint64_val()) {
    local_data.set_uint64_val(request->uint64_val() | local_data.uint64_val());
  }

  all_reduce_state_map_[request->sync_key()].rpc_counter->DecrementCount();

  reactor->Finish(::grpc::Status::OK);
  return reactor;
}

void AllReduceServiceImpl::InitializeState(int sync_key,
                                           const AllReduceData& data) {
  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
  all_reduce_state_map_[sync_key] = {
      .local_data = data,
      .rpc_counter = std::make_unique<absl::BlockingCounter>(num_tasks_ - 1),
      .overall_status = absl::OkStatus(),
  };
  local_data_cv_.SignalAll();
}

absl::StatusOr<AllReduceData> AllReduceServiceImpl::Wait(int sync_key) {
  absl::BlockingCounter* rpc_counter = nullptr;
  {
    absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
    rpc_counter = all_reduce_state_map_.at(sync_key).rpc_counter.get();
  }

  // Wait for the counter outside of the mutex lock to prevent deadlock.
  CHECK_NE(rpc_counter, nullptr);
  rpc_counter->Wait();

  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
  AllReduceData result = all_reduce_state_map_[sync_key].local_data;
  all_reduce_state_map_.erase(sync_key);
  return result;
}

}  // namespace jax_sc_embedding
