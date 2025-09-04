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
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_interface.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "include/grpcpp/client_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/create_channel.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/grpc_credentials.h"
#include "tsl/platform/statusor.h"  // from @tsl

namespace jax_sc_embedding {
namespace rpc {

void GrpcAllReduceInterface::SetUp() {
  for (const auto& peer_address : peer_addresses_) {
    auto creds = GetDefaultChannelCredentials();
    auto channel = ::grpc::CreateChannel(peer_address, creds);
    auto stub = AllReduceGrpcService::NewStub(channel);
    stubs_.push_back(std::move(stub));
  }
}

absl::StatusOr<AllReduceData> GrpcAllReduceInterface::BlockingAllReduce(
    const AllReduceData& request) {
  if (stubs_.empty()) {
    return absl::FailedPreconditionError("No gRPC stubs available.");
  }

  // Initialize State on the local service.
  local_service_->InitializeState(request.sync_key(), request);
  absl::Mutex mutex;
  grpc::Status overall_status ABSL_GUARDED_BY(mutex) = grpc::Status::OK;
  absl::BlockingCounter outgoing_rpcs(stubs_.size());

  struct ContributeDataArgs {
    ::grpc::ClientContext context;
    AllReduceData request;
    AllReduceResponse response;
  };

  // Send our data to all other peers asynchronously.
  DCHECK_EQ(stubs_.size(), num_tasks_ - 1);
  for (const auto& stub : stubs_) {
    auto args = std::make_shared<ContributeDataArgs>();
    args->context.set_deadline(
        absl::ToChronoTime(absl::Now() + absl::Seconds(60)));
    args->request = request;

    stub->async()->ContributeData(
        &args->context, &args->request, &args->response,
        [&, args](::grpc::Status s) {
          {
            absl::MutexLock lock(&mutex);  // NOLINT (b/438618768)
            if (!s.ok() && overall_status.ok()) {
              overall_status = s;
            }
          }
          outgoing_rpcs.DecrementCount();
        });
  }

  // Wait to receive data from all other hosts (Local service performs the
  // reduction).
  TF_ASSIGN_OR_RETURN(AllReduceData result,
                      local_service_->Wait(request.sync_key()));

  // Wait for all outgoing RPCs to complete.
  outgoing_rpcs.Wait();

  // Propagate any RPC errors.
  if (!overall_status.ok()) {
    return absl::Status(
        static_cast<absl::StatusCode>(overall_status.error_code()),
        overall_status.error_message());
  }

  return result;
}

absl::StatusOr<bool> GrpcAllReduceInterface::BlockingAllReduce(
    int sync_key, bool minibatching_required) {
  AllReduceData request;
  request.set_sync_key(sync_key);
  request.set_src_rank(task_id_);
  request.set_bool_val(minibatching_required);
  TF_ASSIGN_OR_RETURN(auto response, BlockingAllReduce(request));
  return response.bool_val();
}

absl ::StatusOr<uint64_t> GrpcAllReduceInterface::BlockingAllReduce(
    int sync_key, uint64_t minibatching_split) {
  AllReduceData request;
  request.set_sync_key(sync_key);
  request.set_src_rank(task_id_);
  request.set_uint64_val(minibatching_split);
  TF_ASSIGN_OR_RETURN(auto response, BlockingAllReduce(request));
  return response.uint64_val();
}

}  // namespace rpc
}  // namespace jax_sc_embedding
