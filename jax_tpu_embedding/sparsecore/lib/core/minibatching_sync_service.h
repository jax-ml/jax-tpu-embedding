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
#ifndef JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SYNC_SERVICE_H_
#define JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SYNC_SERVICE_H_

#include <cstdint>
#include <memory>
#include <type_traits>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/barrier.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/all_reduce_interface.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "tsl/platform/statusor.h"  // from @tsl

namespace jax_sc_embedding {

namespace internal {

inline uint64_t Serialize(MinibatchingSplit value) { return value.to_ullong(); }
inline MinibatchingSplit Deserialize(uint64_t value) {
  return MinibatchingSplit(value);
}

inline bool Serialize(bool value) { return value; }
inline bool Deserialize(bool value) { return value; }

}  // namespace internal

// A service for synchronizing values across multiple stacked tables and hosts
// during minibatching. The template parameter `T` is the type of the value
// being synchronized. Currently, `T` must be a type for which
// `internal::Serialize` and `internal::Deserialize` are specialized.
// Specifically:
// -   `bool` is used for synchronizing `minibatching_required`.
// -   `MinibatchingSplit` (an alias for `std::bitset`) is used for
//     synchronizing `minibatching_split`.
// This class is single-use. The internal barriers are destroyed after being
// used once, so SyncValue can only be called successfully once per service
// instance.
template <typename T>
class MinibatchingSyncService {
 public:
  explicit MinibatchingSyncService(int num_stacked_tables);

  // Synchronizes the value across all stacked tables and hosts.
  absl::StatusOr<T> SyncValue(T table_value, int sync_key,
                              AllReduceInterface* all_reduce_interface);

  // Returns the number of minibatches. This function is only enabled when `T`
  // is `MinibatchingSplit` using SFINAE (`std::enable_if_t`), because it
  // relies on the `count()` member function, which is specific to
  // `std::bitset`.
  template <typename U = T,
            std::enable_if_t<std::is_same_v<U, MinibatchingSplit>, bool> = true>
  int GetNumMinibatches() {
    absl::MutexLock lock(&mutex_);
    return shared_value_.count() + 1;
  }

 private:
  const int num_stacked_tables_;
  absl::Mutex mutex_;
  T shared_value_;
  std::unique_ptr<absl::Barrier> local_reduced_;
  std::unique_ptr<absl::Barrier> global_reduced_;
};

template <typename T>
MinibatchingSyncService<T>::MinibatchingSyncService(int num_stacked_tables)
    : num_stacked_tables_(num_stacked_tables), shared_value_() {
  local_reduced_ = std::make_unique<absl::Barrier>(num_stacked_tables_);
  global_reduced_ = std::make_unique<absl::Barrier>(num_stacked_tables_);
}

template <typename T>
absl::StatusOr<T> MinibatchingSyncService<T>::SyncValue(
    T table_value, int sync_key,
    AllReduceInterface* absl_nullable all_reduce_interface) {
  DCHECK_NE(local_reduced_, nullptr);
  DCHECK_NE(global_reduced_, nullptr);
  {
    absl::MutexLock lock(&mutex_);
    shared_value_ |= table_value;
  }

  // Wait for local threads to combine their values.
  if (local_reduced_->Block()) {
    // This is the last thread for this host.
    if (all_reduce_interface != nullptr) {
      TF_ASSIGN_OR_RETURN(auto reduced_value,
                          all_reduce_interface->BlockingAllReduce(
                              sync_key, internal::Serialize(shared_value_)));
      shared_value_ = internal::Deserialize(reduced_value);
    }
    local_reduced_.reset();
  }

  // Wait for last thread to sync across hosts (if we need cross host sync).
  if (all_reduce_interface != nullptr && global_reduced_->Block()) {
    global_reduced_.reset();
  }

  // The `absl::Barrier::Block()` calls act as memory barriers. When the last
  // thread passes `local_reduced_->Block()`, the writes to `shared_value_`
  // performed by that thread (including the result of `BlockingAllReduce`)
  // become visible to all other threads that have also passed the barrier.
  // Thus, all threads are guaranteed to see the globally reduced
  // `shared_value_`.
  return shared_value_;
}

}  // namespace jax_sc_embedding
#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SYNC_SERVICE_H_
