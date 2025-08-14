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
#include "jax_tpu_embedding/sparsecore/lib/core/pjrt_all_reduce.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "third_party/tensorflow/compiler/xla/pjrt/distributed/client.h"

namespace jax_sc_embedding {

AllReduceCallback CreatePjRtAllReduceCallback(
    std::shared_ptr<xla::DistributedRuntimeClient> client, int node_id,
    int num_nodes, absl::Duration timeout) {
  return [client, node_id, num_nodes, timeout](int sync_key,
                                               uint64_t local_val) -> uint64_t {
    const std::string key_prefix = absl::StrCat("allreduce/", sync_key, "/");
    const std::string node_val_key = absl::StrCat(key_prefix, "val/", node_id);

    // 1. Post local value.
    CHECK_OK(client->KeyValueSet(node_val_key, std::to_string(local_val)));

    // 2. Barrier to ensure all nodes have posted their values.
    CHECK_OK(client->WaitAtBarrier(absl::StrCat(key_prefix, "arrive"), timeout,
                                   std::nullopt));

    // 3. Node 0 performs the reduction.
    uint64_t reduced_val = 0ULL;  // Identity for bitwise OR
    if (node_id == 0) {
      for (int i = 0; i < num_nodes; ++i) {
        const std::string current_val_key = absl::StrCat(key_prefix, "val/", i);
        auto val_str_status =
            client->BlockingKeyValueGet(current_val_key, timeout);
        CHECK_OK(val_str_status.status());
        reduced_val |= std::stoull(val_str_status.value());
      }
      // Post the result.
      CHECK_OK(client->KeyValueSet(absl::StrCat(key_prefix, "result"),
                                   std::to_string(reduced_val)));
    }

    // 4. Barrier to ensure the result is available.
    CHECK_OK(client->WaitAtBarrier(absl::StrCat(key_prefix, "depart"), timeout,
                                   std::nullopt));

    // 5. All nodes retrieve the result.
    auto result_str_status = client->BlockingKeyValueGet(
        absl::StrCat(key_prefix, "result"), timeout);
    CHECK_OK(result_str_status.status());
    return std::stoull(result_str_status.value());
  };
}

}  // namespace jax_sc_embedding
