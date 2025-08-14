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
'''#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PJRT_ALL_REDUCE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PJRT_ALL_REDUCE_H_

#include <memory>

#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "third_party/tensorflow/compiler/xla/pjrt/distributed/client.h"

    namespace jax_sc_embedding {

  // Creates an AllReduceCallback using a PjRt DistributedRuntimeClient.
  // The all-reduce operation performed is a bitwise OR.
  //
  // Args:
  //   client: The PjRt distributed runtime client.
  //   node_id: The ID of the current node.
  //   num_nodes: The total number of nodes participating in the all-reduce.
  //   timeout: Timeout duration for distributed operations.
  //
  // Returns:
  //   An AllReduceCallback function.
  AllReduceCallback CreatePjRtAllReduceCallback(
      std::shared_ptr<xla::DistributedRuntimeClient> client, int node_id,
      int num_nodes, absl::Duration timeout = absl::Seconds(60));

}  // namespace jax_sc_embedding

#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_PJRT_ALL_REDUCE_H_
'''