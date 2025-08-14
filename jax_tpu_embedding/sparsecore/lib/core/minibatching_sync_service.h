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
#include <type_traits>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "tsl/platform/statusor.h"  // from @tsl

namespace jax_sc_embedding {
namespace internal {
template <typename T>
struct Converter {
  static uint64_t to_uint64_t(T value) { return static_cast<uint64_t>(value); }
  static T from_uint64_t(uint64_t value) { return static_cast<T>(value); }
};

template <>
struct Converter<MinibatchingSplit> {
  static uint64_t to_uint64_t(MinibatchingSplit value) {
    return value.to_ullong();
  }
  static MinibatchingSplit from_uint64_t(uint64_t value) {
    return MinibatchingSplit(value);
  }
};
}  // namespace internal

template <typename T>
class MinibatchingSyncService {
 public:
  explicit MinibatchingSyncService(int num_stacked_tables);

  // Synchronizes the value across all stacked tables and hosts.
  absl::StatusOr<T> SyncValue(T table_value, int sync_key,
                              const AllReduceCallback& all_reduce_callback);

  template <typename U = T,
            std::enable_if_t<std::is_same_v<U, MinibatchingSplit>, bool> = true>
  int GetNumMinibatches() {
    absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
    return shared_value_.count() + 1;
  }

 private:
  const int num_stacked_tables_;
  absl::Mutex mutex_;
  T shared_value_ ABSL_GUARDED_BY(mutex_);
  int num_tables_arrived_ ABSL_GUARDED_BY(mutex_);
  absl::CondVar cv_ ABSL_GUARDED_BY(mutex_);
};

template <typename T>
MinibatchingSyncService<T>::MinibatchingSyncService(int num_stacked_tables)
    : num_stacked_tables_(num_stacked_tables),
      shared_value_(),
      num_tables_arrived_(0) {}

template <typename T>
absl::StatusOr<T> MinibatchingSyncService<T>::SyncValue(
    T table_value, int sync_key, const AllReduceCallback& all_reduce_callback) {
  absl::MutexLock lock(&mutex_);  // NOLINT (b/438618768)
  shared_value_ = shared_value_ | table_value;
  num_tables_arrived_++;

  if (num_tables_arrived_ == num_stacked_tables_) {
    // This is the last thread for this host.
    if (all_reduce_callback) {
      uint64_t value_to_reduce =
          internal::Converter<T>::to_uint64_t(shared_value_);
      TF_ASSIGN_OR_RETURN(uint64_t reduced_value,
                          all_reduce_callback(sync_key, value_to_reduce));
      shared_value_ = internal::Converter<T>::from_uint64_t(reduced_value);
    }
    // Reset counter and notify others.
    num_tables_arrived_ = 0;
    cv_.SignalAll();
  } else {
    // Wait for the last thread to finish all-reduce.
    while (num_tables_arrived_ != 0) {
      cv_.Wait(&mutex_);
    }
  }
  return shared_value_;
}

}  // namespace jax_sc_embedding
#endif  // JAX_TPU_EMBEDDING_SPARSECORE_LIB_CORE_MINIBATCHING_SYNC_SERVICE_H_
