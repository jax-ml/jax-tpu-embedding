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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_GRPC_CREDENTIALS_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_GRPC_CREDENTIALS_H_

#include <memory>

#include "include/grpcpp/security/credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc

namespace jax_sc_embedding {
namespace rpc {

// Returns default credentials for use when creating a gRPC server.
std::shared_ptr<::grpc::ServerCredentials> GetDefaultServerCredentials();

// Returns default credentials for use when creating a gRPC channel.
std::shared_ptr<::grpc::ChannelCredentials> GetDefaultChannelCredentials();

}  // namespace rpc
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_GRPC_GRPC_CREDENTIALS_H_
