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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_TEST_UTILS_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_TEST_UTILS_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/grpc/minibatching_node.h"
#include "tsl/platform/test.h"  // from @tsl

namespace jax_sc_embedding {
namespace testing_utils {

// Helper function to set up MinibatchingNode instances for each host.
inline std::vector<std::unique_ptr<rpc::MinibatchingNode>>
SetUpMinibatchingNodes(int num_hosts) {
  std::vector<int> ports;
  ports.reserve(num_hosts);
  for (int i = 0; i < num_hosts; ++i) {
    ports.push_back(tsl::testing::PickUnusedPortOrDie());
  }

  std::vector<std::string> peer_addresses;
  peer_addresses.reserve(num_hosts);
  for (int i = 0; i < num_hosts; ++i) {
    peer_addresses.push_back(absl::StrCat("localhost:", ports[i]));
  }

  std::vector<std::unique_ptr<rpc::MinibatchingNode>> nodes;
  nodes.reserve(num_hosts);
  for (int i = 0; i < num_hosts; ++i) {
    std::vector<std::string> other_peer_addresses;
    for (int j = 0; j < num_hosts; ++j) {
      if (i == j) continue;
      other_peer_addresses.push_back(peer_addresses[j]);
    }
    nodes.push_back(std::make_unique<rpc::MinibatchingNode>(
        /*task_id=*/i, /*num_tasks=*/num_hosts, other_peer_addresses,
        ports[i]));
  }
  return nodes;
}

}  // namespace testing_utils
}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_TEST_UTILS_H_
