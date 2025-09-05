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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_MINIBATCHING_NODE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_MINIBATCHING_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "include/grpcpp/server_builder.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/all_reduce_service_impl.h"
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/grpc_credentials.h"

namespace jax_sc_embedding {
namespace rpc {

class MinibatchingNode {
 public:
  MinibatchingNode(int host_id, int num_hosts,
                   std::vector<std::string> peer_addresses,
                   int minibatching_port)
      : all_reduce_service_(
            std::make_unique<AllReduceServiceImpl>(host_id, num_hosts)),
        all_reduce_interface_(std::make_unique<GrpcAllReduceInterface>(
            peer_addresses, host_id, num_hosts, minibatching_port,
            all_reduce_service_.get())),
        all_reduce_server_(
            ::grpc::ServerBuilder()
                .AddListeningPort(absl::StrCat("[::]:", minibatching_port),
                                  GetDefaultServerCredentials())
                .RegisterService(all_reduce_service_.get())
                .BuildAndStart()) {
    all_reduce_interface_->SetUp();
  }

  AllReduceInterface* GetAllReduceInterface() ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return all_reduce_interface_.get();
  }

 private:
  std::unique_ptr<AllReduceServiceImpl> all_reduce_service_;
  std::unique_ptr<rpc::GrpcAllReduceInterface> all_reduce_interface_;
  std::unique_ptr<::grpc::Server> all_reduce_server_;
};
}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_MINIBATCHING_NODE_H_
