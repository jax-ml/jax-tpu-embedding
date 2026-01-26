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

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "include/grpcpp/client_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/create_channel.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce.grpc.pb.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/grpc_credentials.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace rpc {
namespace {

// Helper function to send ContributeData RPCs to all peers.
void SendLocalData(
    const absl::flat_hash_map<
        std::string, std::unique_ptr<AllReduceGrpcService::Stub>>& stubs,
    const AllReduceData& request, int num_hosts,
    tsl::AsyncValueRef<AllReduceData> result_av,
    std::shared_ptr<std::atomic<bool>> error_set) {
  tsl::profiler::TraceMe traceme("GrpcAllReduceInterface::SendLocalData");

  struct ContributeDataArgs {
    ::grpc::ClientContext context;
    AllReduceData request;
    AllReduceResponse response;
  };

  // Send our data to all other peers asynchronously.
  DCHECK_EQ(stubs.size(), num_hosts - 1);
  {
    tsl::profiler::TraceMe traceme_send(
        "GrpcAllReduceInterface::SendLocalData::SendToPeers");
    auto deadline = absl::ToChronoTime(absl::Now() + absl::Seconds(7200));
    for (const auto& [peer_address, stub] : stubs) {
      auto args = std::make_shared<ContributeDataArgs>();
      args->context.set_deadline(deadline);
      args->request = request;
      VLOG(2) << "Sending RPC to peer: " << peer_address
              << " for sync_key: " << request.sync_key();

      stub->async()->ContributeData(
          &args->context, &args->request, &args->response,
          [args, peer_address = peer_address, result_av,
           error_set](::grpc::Status s) mutable {
            if (!s.ok() && !result_av.IsAvailable() &&
                !error_set->exchange(true)) {
              absl::Status status =
                  absl::Status(static_cast<absl::StatusCode>(s.error_code()),
                               s.error_message());
              LOG(ERROR) << "ContributeData async RPC to peer " << peer_address
                         << " for sync_key: " << args->request.sync_key()
                         << " failed with status: " << status;
              result_av.SetError(status);
            } else {
              VLOG(2) << "ContributeData async RPC to peer " << peer_address
                      << " for sync_key: " << args->request.sync_key()
                      << " completed successfully.";
            }
          });
    }
  }
}

}  // namespace

void GrpcAllReduceInterface::SetUp() {
  for (const auto& peer_address : peer_addresses_) {
    VLOG(2) << "Attempting to create channel for peer: " << peer_address;
    auto creds = GetDefaultChannelCredentials();
    auto channel = ::grpc::CreateChannel(peer_address, creds);
    auto stub = AllReduceGrpcService::NewStub(channel);
    stubs_.insert({peer_address, std::move(stub)});
  }
  LOG(INFO) << "GrpcAllReduceInterface channel creation complete for task_id: "
            << task_id_ << ", num_tasks: " << num_tasks_
            << ", peer_addresses: " << absl::StrJoin(peer_addresses_, ",");
}

tsl::AsyncValueRef<AllReduceData>
GrpcAllReduceInterface::AsyncAllReduceInternal(const AllReduceData& request) {
  CHECK_EQ(num_tasks_, stubs_.size() + 1);

  // Initialize or update state on the local service.
  bool is_last_local =
      local_service_->InitializeOrUpdateState(request.sync_key(), request);
  tsl::AsyncValueRef<AllReduceData> result_av =
      local_service_->GetResult(request.sync_key());

  // If I'm the last local thread to check in, and we are doing distributed
  // computation, send local data to peers.
  if (is_last_local && num_tasks_ > 1) {
    tsl::AsyncValueRef<AllReduceData> local_reduced_value =
        local_service_->GetLocalReducedValue(request.sync_key());
    std::shared_ptr<std::atomic<bool>> error_set =
        std::make_shared<std::atomic<bool>>(false);

    // Send our data to all other peers asynchronously and wait for completion.
    local_reduced_value.AndThen(
        [this, local_reduced_value, result_av, error_set]() mutable {
          AllReduceData local_data = local_reduced_value.get();
          SendLocalData(stubs_, local_data, num_tasks_, result_av, error_set);
        });
  }

  return result_av;
}

tsl::AsyncValueRef<bool> GrpcAllReduceInterface::AsyncAllReduce(
    int sync_key, bool minibatching_required) {
  AllReduceData request;
  request.set_sync_key(sync_key);
  request.set_src_rank(task_id_);
  request.set_bool_val(minibatching_required);
  auto response_av = AsyncAllReduceInternal(request);
  return response_av.Map<bool>(
      [](const AllReduceData& resp) { return resp.bool_val(); });
}

tsl::AsyncValueRef<uint64_t> GrpcAllReduceInterface::AsyncAllReduce(
    int sync_key, uint64_t minibatching_split) {
  AllReduceData request;
  request.set_sync_key(sync_key);
  request.set_src_rank(task_id_);
  request.set_uint64_val(minibatching_split);
  auto response_av = AsyncAllReduceInternal(request);
  return response_av.Map<uint64_t>(
      [](const AllReduceData& resp) { return resp.uint64_val(); });
}

}  // namespace rpc
}  // namespace jax_sc_embedding
