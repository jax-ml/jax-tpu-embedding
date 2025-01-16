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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_with_mini_batching.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/blocking_counter.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "hwy/base.h"  // from @highway
#include "hwy/contrib/sort/order.h"  // from @highway
#include "hwy/contrib/sort/vqsort.h"  // from @highway
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_py_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_threads.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "pybind11/gil.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tsl/profiler/lib/connected_traceme.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace jax_sc_embedding {

namespace {

namespace py = ::pybind11;

using MiniBatchingSplit = int64_t;

using BufferSizes = absl::flat_hash_map<std::string, int32_t>;

// This device specific list is a 1D list.
//
// The 1st level of vector is for the local devices.
// The value is the batch size for each sparsecore on this device.
using DeviceBatchSizeList = std::vector<int32_t>;
using DeviceBatchSizeLists =
    absl::flat_hash_map<std::string, DeviceBatchSizeList>;

// This device specific list is a 2D list.
//
// The 1st level of vector is for the local devices.
// The 2nd level of vector is for the device-local list of data.
template <typename T>
using DeviceDataList = std::vector<std::vector<T>>;
template <typename T>
// The map from (stacked) table name to associated DeviceDataList.
using DeviceDataLists = absl::flat_hash_map<std::string, DeviceDataList<T>>;

// ID counter per device, aggregated across all local sparsecores.
// The index is for the global sparsecores.
using AggregatedIdCounterPerDevice = std::vector<int32_t>;

// ID counter for all devices. The index is for the local devices.
using AggregatedIdCounter = std::vector<AggregatedIdCounterPerDevice>;

// The map from (stacked) table name to associated AggregatedIdCounter.
using AggregatedIdCounters =
    absl::flat_hash_map<std::string, AggregatedIdCounter>;

// This sparsecore specific list is a 2D list.
//
// The 1st level of vector is for the device-local sparsecores.
// The 2nd level of vector is for the sparsecore-local list of data.
template <typename T>
using SparsecoreDataList = std::vector<std::vector<T>>;

// This sparsecore specific list is a 3D list.
//
// The 1st level of vector is for the process-local devices.
// The 2nd level of vector is for the device-local sparsecores.
// The 3rd level of vector is for the sparsecore-local list of data.
template <typename T>
using DeviceSparsecoreDataList = std::vector<SparsecoreDataList<T>>;

// The map from (stacked) table name to associated DeviceSparsecoreDataList.
template <typename T>
using DeviceSparsecoreDataLists =
    absl::flat_hash_map<std::string, DeviceSparsecoreDataList<T>>;

// This sparsecore specific list is a 4D list.
//
// The 1st level of vector is for the process-local devices.
// The 2nd level of vector is for the device-local sparsecores.
// The 3rd level of vector is for the mini-batching split.
// The 4th level of vector is for the sparsecore-local list of data.
template <typename T>
using DeviceSparsecoreMiniBatchingDataList =
    std::vector<DeviceSparsecoreDataList<T>>;

// The map from (stacked) table name to associated
// DeviceSparsecoreMiniBatchingDataList.
template <typename T>
using DeviceSparsecoreMiniBatchingDataLists =
    absl::flat_hash_map<std::string, DeviceSparsecoreMiniBatchingDataList<T>>;

template <typename T>
void Convert2dToPyDictLocked(py::dict& py_dict, const DeviceDataLists<T>& map) {
  for (const auto& [key, value] : map) {
    if (value[0].size() > 0) {
      py::array_t<T> py_value({value.size(), value[0].size()});
      for (int local_device = 0; local_device < value.size(); ++local_device) {
        T* const py_data_ptr = py_value.mutable_data(local_device);
        std::copy(value[local_device].begin(), value[local_device].end(),
                  py_data_ptr);
      }
      py_dict[key.c_str()] = py_value;
    } else {
      py_dict[key.c_str()] = py::none();
    }
  }
}

template <typename T>
void Reshape2dToPyDictLocked(py::dict& py_dict, const DeviceDataLists<T>& map) {
  for (const auto& [key, value] : map) {
    if (value[0].size() > 0) {
      py::array_t<T> py_value(
          {static_cast<ssize_t>(value.size() * value[0].size())});
      T* py_data_ptr = py_value.mutable_data();
      for (int local_device = 0; local_device < value.size(); ++local_device) {
        std::copy(value[local_device].begin(), value[local_device].end(),
                  py_data_ptr);
        py_data_ptr += value[local_device].size();
      }
      py_dict[key.c_str()] = py_value;
    } else {
      py_dict[key.c_str()] = py::none();
    }
  }
}

template <typename T>
void Extend2dToPyDictLocked(py::dict& py_dict, const DeviceDataLists<T>& map,
                            const BufferSizes& intended_buffer_sizes) {
  for (const auto& [key, value] : map) {
    auto buffer_size_it = intended_buffer_sizes.find(key);
    CHECK(buffer_size_it != intended_buffer_sizes.end());
    const int buffer_size = buffer_size_it->second;
    if (buffer_size < value[0].size()) {
      throw std::runtime_error("The intended buffer size is too small.");
    }

    if (value[0].size() > 0) {
      // Note that the elements in the python array are only valid to the point
      // of the actual size of the C++ vector. Beyond that, the elements in the
      // python array are uninitialized, and accessing them could lead to
      // undefined behavior. This is to save time, but we still need the shape
      // to be consistent across all steps to avoid costly re-compilations.
      py::array_t<T> py_value({static_cast<int>(value.size()), buffer_size});
      for (int local_device = 0; local_device < value.size(); ++local_device) {
        T* const py_data_ptr = py_value.mutable_data(local_device);
        std::copy(value[local_device].begin(), value[local_device].end(),
                  py_data_ptr);
      }
      py_dict[key.c_str()] = py_value;
    } else {
      py_dict[key.c_str()] = py::none();
    }
  }
}

std::tuple<bool, SparsecoreDataList<CooFormat>, AggregatedIdCounterPerDevice,
           AggregatedIdCounterPerDevice, AggregatedIdCounterPerDevice>
SortAndGroupCooTensorsWithIdDrop(
    const std::vector<CooFormat>& coo_tensors_for_device, bool drop_ids,
    int num_scs, int num_sc_per_device, int batch_size_per_sc,
    int max_ids_per_partition, int max_unique_ids_per_partition,
    int initial_num_coo_tensors_per_sc, std::vector<int32_t>& max_ids_per_sc,
    std::vector<int32_t>& max_unique_ids_per_sc,
    std::vector<int32_t>& id_drop_counter_per_sc, std::vector<uint64_t>& keys) {
  tsl::profiler::TraceMe t("SortAndGroupCooTensorsWithIdDrop");

  bool mini_batching_needed = false;

  SparsecoreDataList<CooFormat> coo_tensors_by_sc;

  // Initialize the counters to be 0 for all SCs.
  // Index is for global SCs as sources of the embedding data.
  AggregatedIdCounterPerDevice max_id_counter_by_sc(num_scs, 0);
  AggregatedIdCounterPerDevice max_unique_id_counter_by_sc(num_scs, 0);
  AggregatedIdCounterPerDevice id_drop_counter_by_sc(num_scs, 0);

  coo_tensors_by_sc.resize(num_sc_per_device);
  for (auto& coo_tensors_by_client_sc : coo_tensors_by_sc) {
    coo_tensors_by_client_sc.reserve(initial_num_coo_tensors_per_sc);
  }

  uint32_t index = 0;
  const int32_t num_scs_bit = std::log2(num_scs);
  const int total_coo_tensors = coo_tensors_for_device.size();
  for (int32_t i = 0; i < num_sc_per_device; ++i) {
    // Reset the counters as the vectors are reused.
    max_ids_per_sc.clear();
    max_ids_per_sc.resize(num_scs, 0);
    max_unique_ids_per_sc.clear();
    max_unique_ids_per_sc.resize(num_scs, 0);
    id_drop_counter_per_sc.clear();
    id_drop_counter_per_sc.resize(num_scs, 0);
    keys.clear();
    // We take the advantage of the fact that the row_ids are already sorted
    // within each batch.
    while (index < total_coo_tensors &&
           (unsigned)(coo_tensors_for_device[index].row_id -
                      i * batch_size_per_sc) < batch_size_per_sc) {
      // The key here is [col_ids % num_scs, col_ids / num_scs, index].
      // Note that this assumes `num_scs` is a power of 2.
      keys.push_back(
          (static_cast<uint64_t>(absl::rotr(
               static_cast<uint32_t>(coo_tensors_for_device[index].col_id),
               num_scs_bit))
           << 32) +
          index);
      ++index;
    }
    hwy::VQSort(keys.data(), keys.size(), hwy::SortAscending());

    uint32_t prev_col_id = std::numeric_limits<uint32_t>::max();
    for (const auto key : keys) {
      uint32_t sc_id = static_cast<uint32_t>(key >> (64 - num_scs_bit));
      if (static_cast<uint32_t>(key >> 32) != prev_col_id) {
        max_unique_ids_per_sc[sc_id] += 1;
      }
      max_ids_per_sc[sc_id] += 1;

      // If either max_unique_ids_per_partition or max_ids_per_partition is
      // exceeded, we drop the id.
      if (max_unique_ids_per_sc[sc_id] > max_unique_ids_per_partition ||
          max_ids_per_sc[sc_id] > max_ids_per_partition) {
        if (drop_ids) {
          prev_col_id = static_cast<uint32_t>(key >> 32);
          // Record that the id is dropped.
          id_drop_counter_per_sc[sc_id] += 1;
          continue;
        } else {
          // Don't drop the id. Just record that mini-batching is needed.
          mini_batching_needed = true;
        }
      }

      coo_tensors_by_sc[i].push_back(
          coo_tensors_for_device[static_cast<uint32_t>(key)]);
      prev_col_id = static_cast<uint32_t>(key >> 32);
    }

    for (int s = 0; s < num_scs; ++s) {
      // Taking the max of the id counters for each SC.
      // Note here the max is taken across all local sparsecores.
      max_id_counter_by_sc[s] =
          std::max(max_id_counter_by_sc[s], max_ids_per_sc[s]);
      max_unique_id_counter_by_sc[s] =
          std::max(max_unique_id_counter_by_sc[s], max_unique_ids_per_sc[s]);

      // Accumulate ID drop counters across all local sparsecores.
      id_drop_counter_by_sc[s] += id_drop_counter_per_sc[s];
    }
  }
  return std::make_tuple(mini_batching_needed, std::move(coo_tensors_by_sc),
                         std::move(max_id_counter_by_sc),
                         std::move(max_unique_id_counter_by_sc),
                         std::move(id_drop_counter_by_sc));
}

DeviceSparsecoreMiniBatchingDataLists<CooFormat> SplitCooTensorsByVocabularyDiv(
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    int split_count, const DeviceSparsecoreDataLists<CooFormat>& coo_tensors) {
  tsl::profiler::TraceMe t("SplitCooTensorsByVocabularyDiv");
  DeviceSparsecoreMiniBatchingDataLists<CooFormat> split_coo_tensors;
  for (const auto& [stacked_table_name, stacked_table_metadata] :
       stacked_tables) {
    // Vocabulary size is the same for all stacked tables.
    const int vocabulary_size =
        stacked_table_metadata[0].stacked_table_vocab_size;
    const int split_size = vocabulary_size / split_count;

    DeviceSparsecoreMiniBatchingDataList<CooFormat>
        split_coo_tensors_current_table;

    const auto& coo_tensors_current_table = coo_tensors.at(stacked_table_name);

    // local device count
    split_coo_tensors_current_table.resize(coo_tensors_current_table.size());
    for (int local_device = 0; local_device < coo_tensors_current_table.size();
         ++local_device) {
      // number of sparsecores per device
      split_coo_tensors_current_table[local_device].resize(
          coo_tensors_current_table[local_device].size());
      for (int sc_index = 0;
           sc_index < coo_tensors_current_table[local_device].size();
           ++sc_index) {
        // number of mini batches per sparsecore
        split_coo_tensors_current_table[local_device][sc_index].resize(
            split_count);

        // Reserve space for each mini batch.
        for (int mini_batch_index = 0; mini_batch_index < split_count;
             ++mini_batch_index) {
          split_coo_tensors_current_table
              [local_device][sc_index][mini_batch_index]
                  .reserve(
                      coo_tensors_current_table[local_device][sc_index].size() /
                      split_count);
        }

        for (const auto& coo_tensor :
             coo_tensors_current_table[local_device][sc_index]) {
          const int mini_batch_index = coo_tensor.col_id / split_size;
          split_coo_tensors_current_table[local_device][sc_index]
                                         [mini_batch_index]
                                             .push_back(coo_tensor);
        }
      }
    }

    split_coo_tensors[stacked_table_name] =
        std::move(split_coo_tensors_current_table);
  }
  return split_coo_tensors;
}

DeviceSparsecoreMiniBatchingDataLists<CooFormat> SplitCooTensorsByVocabularyMod(
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    int modulus, const DeviceSparsecoreDataLists<CooFormat>& coo_tensors) {
  tsl::profiler::TraceMe t("SplitCooTensorsByVocabularyMod");
  DeviceSparsecoreMiniBatchingDataLists<CooFormat> split_coo_tensors;
  for (const auto& [stacked_table_name, stacked_table_metadata] :
       stacked_tables) {
    DeviceSparsecoreMiniBatchingDataList<CooFormat>
        split_coo_tensors_current_table;

    const auto& coo_tensors_current_table = coo_tensors.at(stacked_table_name);

    // local device count
    split_coo_tensors_current_table.resize(coo_tensors_current_table.size());
    for (int local_device = 0; local_device < coo_tensors_current_table.size();
         ++local_device) {
      // number of sparsecores per device
      split_coo_tensors_current_table[local_device].resize(
          coo_tensors_current_table[local_device].size());
      for (int sc_index = 0;
           sc_index < coo_tensors_current_table[local_device].size();
           ++sc_index) {
        // number of mini batches per sparsecore
        split_coo_tensors_current_table[local_device][sc_index].resize(modulus);

        // Reserve space for each mini batch.
        for (int mini_batch_index = 0; mini_batch_index < modulus;
             ++mini_batch_index) {
          split_coo_tensors_current_table
              [local_device][sc_index][mini_batch_index]
                  .reserve(
                      coo_tensors_current_table[local_device][sc_index].size() /
                      modulus);
        }

        for (const auto& coo_tensor :
             coo_tensors_current_table[local_device][sc_index]) {
          const int mini_batch_index = coo_tensor.col_id % modulus;
          split_coo_tensors_current_table[local_device][sc_index]
                                         [mini_batch_index]
                                             .push_back(coo_tensor);
        }
      }
    }

    split_coo_tensors[stacked_table_name] =
        std::move(split_coo_tensors_current_table);
  }
  return split_coo_tensors;
}

void PadDataTensorsToEndOfRegisterWidth(DeviceDataList<int32_t>* embedding_ids,
                                        DeviceDataList<int32_t>* sample_ids,
                                        DeviceDataList<float>* gains,
                                        const int local_device_id,
                                        const int sparsecore_register_width) {
  CHECK_NE(embedding_ids, nullptr);
  CHECK_NE(sample_ids, nullptr);
  CHECK_NE(gains, nullptr);

  // All data tensors should have the same size.
  CHECK_EQ((*sample_ids)[local_device_id].size(),
           (*embedding_ids)[local_device_id].size());
  CHECK_EQ((*gains)[local_device_id].size(),
           (*sample_ids)[local_device_id].size());

  const int remainder =
      (*embedding_ids)[local_device_id].size() % sparsecore_register_width;
  const int padding_size =
      (sparsecore_register_width - remainder) % sparsecore_register_width;
  for (int i = 0; i < padding_size; ++i) {
    (*embedding_ids)[local_device_id].push_back(INT_MAX);
    (*sample_ids)[local_device_id].push_back(INT_MAX);
    (*gains)[local_device_id].push_back(NAN);
  }
}

std::tuple<DeviceDataLists<int32_t>, DeviceDataLists<int32_t>,
           DeviceDataLists<int32_t>, DeviceDataLists<float>, BufferSizes>
EncodeMiniBatchingDataUnlocked(
    const DeviceSparsecoreMiniBatchingDataLists<CooFormat>& split_coo_tensors,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const DeviceBatchSizeLists& batch_sizes, const int local_device_count,
    const int global_device_count, const int num_sc_per_device,
    const int sparsecore_register_width, const bool has_leading_dimension,
    const int static_buffer_size_multiplier) {
  // Global number of sparsecores for embedding table
  const int num_scs = num_sc_per_device * global_device_count;

  // All tables have to have the same mini-batch size.
  int mini_batch_size = -1;

  if (!split_coo_tensors.empty()) {
    // Note that if the COO tensors are empty, we will not have any data.
    // So we can use the first table to get the mini batch size.
    mini_batch_size = split_coo_tensors.begin()->second[0][0].size();
  }

  struct {
    absl::Mutex mutex;
    DeviceDataLists<int32_t> row_pointers ABSL_GUARDED_BY(mutex);
    DeviceDataLists<int32_t> embedding_ids ABSL_GUARDED_BY(mutex);
    DeviceDataLists<int32_t> sample_ids ABSL_GUARDED_BY(mutex);
    DeviceDataLists<float> gains ABSL_GUARDED_BY(mutex);
    BufferSizes buffer_sizes ABSL_GUARDED_BY(mutex);
  } results;

  absl::BlockingCounter counter(stacked_tables.size());
  tsl::profiler::TraceMeProducer producer("EncodingMainThread");
  {
    for (const auto& [stacked_table_name, split_coo_tensors_current_table] :
         split_coo_tensors) {
      PreprocessingThreadPool()->Schedule([&, context_id =
                                                  producer.GetContextId()] {
        tsl::profiler::TraceMeConsumer consumer(
            [&] { return absl::StrCat("Encoding-", stacked_table_name); },
            context_id);

        // Initialize the resulting data lists.
        // Note that through padding, the resulting data lists are always
        // the same size, no matter how much is required after encoding.
        DeviceDataList<int32_t> row_pointers_per_table(local_device_count);
        DeviceDataList<int32_t> embedding_ids_per_table(local_device_count);
        DeviceDataList<int32_t> sample_ids_per_table(local_device_count);
        DeviceDataList<float> gains_per_table(local_device_count);

        // Reserve space for lists. Note that we're not initializing the lists
        // yet, so the size() member for each list is still 0.
        const int expected_row_pointers_size_per_device =
            num_sc_per_device * mini_batch_size *
            std::max(num_scs, sparsecore_register_width);

        auto& batch_size_per_sc_for_current_table =
            batch_sizes.at(stacked_table_name);

        // Allocate the static buffers.
        auto stacked_table_metadata = stacked_tables.find(stacked_table_name);
        CHECK(stacked_table_metadata != stacked_tables.end());
        const int pad_per_device_array_to_size = ComputeCooBufferSize(
            num_scs, num_sc_per_device, stacked_table_metadata->second,
            static_buffer_size_multiplier);

        for (int local_device = 0; local_device < local_device_count;
             ++local_device) {
          row_pointers_per_table[local_device].reserve(
              expected_row_pointers_size_per_device);
          embedding_ids_per_table[local_device].reserve(
              pad_per_device_array_to_size);
          sample_ids_per_table[local_device].reserve(
              pad_per_device_array_to_size);
          gains_per_table[local_device].reserve(pad_per_device_array_to_size);
        }

        for (int local_device = 0; local_device < local_device_count;
             ++local_device) {
          const int batch_size_per_sc =
              batch_size_per_sc_for_current_table[local_device];

          // A client sparsecore handles input samples, grouped by mini-batch.
          for (int client_sc_id = 0; client_sc_id < num_sc_per_device;
               ++client_sc_id) {
            for (int mini_batch_id = 0; mini_batch_id < mini_batch_size;
                 ++mini_batch_id) {
              auto& coo_tensors_within_mini_batch =
                  split_coo_tensors_current_table[local_device][client_sc_id]
                                                 [mini_batch_id];

              auto next_coo_tensor = coo_tensors_within_mini_batch.begin();
              auto coo_tensors_within_mini_batch_end =
                  coo_tensors_within_mini_batch.end();
              for (int expected_server_sc_id = 0;
                   expected_server_sc_id < num_scs;) {
                // This is a "partition", as defined by the combination of
                // client sc, server sc, and mini batch.
                int id_counter = 0;
                int previous_row_id_within_server_sc = -1;
                int unique_id_counter = 0;
                int server_sc_id = -1;
                while (next_coo_tensor != coo_tensors_within_mini_batch_end) {
                  // Consume the next COO tensor.
                  const auto& coo_tensor = *next_coo_tensor;

                  // Which sc should provide this embedding data.
                  server_sc_id = coo_tensor.col_id % num_scs;

                  if (server_sc_id != expected_server_sc_id) {
                    // Break from the COO tensor loop, so id counters are reset.
                    break;
                  }

                  // Within this server sc, which (embedding table) row has the
                  // desired data.
                  const int row_id_within_server_sc =
                      coo_tensor.col_id / num_scs;

                  // Within this client sc, which (input sample) row should
                  // receive (accumulate) the embedding data.
                  const int row_id_within_client_sc =
                      coo_tensor.row_id % batch_size_per_sc;

                  id_counter++;
                  if (unique_id_counter == 0) {
                    // Set unique id counter to 1 if it's the first id.
                    unique_id_counter = 1;
                  } else if (row_id_within_server_sc !=
                             previous_row_id_within_server_sc) {
                    // If this is not the first id, and it's not the same row id
                    // as the previous id, then it's a new unique id.
                    unique_id_counter++;
                    previous_row_id_within_server_sc = row_id_within_server_sc;
                  }

                  // Record the data for this COO tensor.
                  embedding_ids_per_table[local_device].push_back(
                      row_id_within_server_sc);
                  sample_ids_per_table[local_device].push_back(
                      row_id_within_client_sc);
                  gains_per_table[local_device].push_back(coo_tensor.gain);

                  ++next_coo_tensor;
                }  // End of COO tensor loop.

                if (next_coo_tensor == coo_tensors_within_mini_batch_end) {
                  // we've consumed all COO tensors for this mini batch.
                  // Set the server_sc_id to end of this mini batch, so that we
                  // can pad the row pointers properly.
                  server_sc_id = std::max(num_scs, sparsecore_register_width);
                }

                int padding_size = 1;
                if (server_sc_id >= 0) {
                  // server_sc_id == -1 if there is no data for this partition.
                  // Since the COO tensors are sorted, as long as there is data,
                  // server_sc_id must be larger than expected_server_sc_id.
                  CHECK_GT(server_sc_id, expected_server_sc_id);
                  padding_size = server_sc_id - expected_server_sc_id;
                }
                // padding_size is at least 1.
                CHECK_GE(padding_size, 1);
                // Push one new row pointer for this particular server sc.
                row_pointers_per_table[local_device].push_back(
                    embedding_ids_per_table[local_device].size());

                // Pad all three tensors up to sparsecore register width as
                // we're ending this partition. Note that if there is no new
                // data for this partition, we do not pad more entries.
                PadDataTensorsToEndOfRegisterWidth(
                    &embedding_ids_per_table, &sample_ids_per_table,
                    &gains_per_table, local_device, sparsecore_register_width);

                for (int i = 1; i < padding_size; ++i) {
                  // Push a new row pointer for each server sc.
                  row_pointers_per_table[local_device].push_back(
                      embedding_ids_per_table[local_device].size());
                }
                expected_server_sc_id += padding_size;
              }  // End of "partition" loop.
            }  // Mini batch loop.
          }  // Client SC loop.
        }  // Local device loop.

        {
          // Move the data lists to the main thread.
          absl::MutexLock lock(&results.mutex);
          results.row_pointers[stacked_table_name] =
              std::move(row_pointers_per_table);
          results.embedding_ids[stacked_table_name] =
              std::move(embedding_ids_per_table);
          results.sample_ids[stacked_table_name] =
              std::move(sample_ids_per_table);
          results.gains[stacked_table_name] = std::move(gains_per_table);
          results.buffer_sizes[stacked_table_name] =
              pad_per_device_array_to_size;
        }

        // Signal the main thread that this task is done.
        counter.DecrementCount();
      }  // End of lambda for threaded task.
      );  // End of Schedule.
    }
    counter.Wait();
  }  // End of EncodingMainThread context.

  {
    absl::MutexLock lock(&results.mutex);
    return std::make_tuple(
        std::move(results.row_pointers), std::move(results.embedding_ids),
        std::move(results.sample_ids), std::move(results.gains),
        std::move(results.buffer_sizes));
  }
}

inline int GetColIdInline(const int col_id, const int col_shift,
                          const int col_offset, const int num_scs_mod,
                          const int num_scs_mod_inv) {
  // This is equivalent to:
  // (col_ids + col_shift) % num_sc_shards +
  //    (col_ids // num_sc_shards * num_sc_shards) + col_offset
  return ((col_id + col_shift) & num_scs_mod) + (col_id & num_scs_mod_inv) +
         col_offset;
}

class FeatureWeightRepresentation {
 public:
  using index_ref_type = py::detail::unchecked_reference<int64_t, 2>;
  using value_ref_type = py::detail::unchecked_reference<int32_t, 1>;
  using weights_ref_type = py::detail::unchecked_reference<float, 1>;

  FeatureWeightRepresentation(const index_ref_type& indices,
                              const value_ref_type& values,
                              const weights_ref_type& weights)
      : indices_(indices), values_(values), weights_(weights) {
    index_stride = &indices(1, 0) - &indices(0, 0);
    value_stride = &values(1) - &values(0);
    weight_stride = &weights(1) - &weights(0);
  }

  void ExtractCooTensors(const int start_index, const int end_index,
                         const int row_offset, const int col_offset,
                         const int col_shift, const int num_scs,
                         const int global_device_count,
                         std::vector<CooFormat>& coo_tensors) const {
    tsl::profiler::TraceMe t([] { return "ExtractCooTensors"; });

    const int num_scs_bit = std::log2(num_scs);
    const int num_scs_mod = (1 << num_scs_bit) - 1;
    const int num_scs_mod_inv = ~num_scs_mod;

    const int row_offset_per_device = row_offset / global_device_count;

    // Get the range of elements in the indices array for samples between
    // start_index and end_index.
    auto [begin_cursor, end_cursor] = GetElemRange(start_index, end_index);

    // Expand the size of the vector to accommodate the new COO tensors.
    coo_tensors.reserve(coo_tensors.size() + end_cursor - begin_cursor);

    const bool has_weights = weights_.size() > 0;

    // Iterate through all elements in the current slice of theindices array.
    // These pointers are created to avoid repeated calculations around shape
    // and strides.
    const int64_t* indices_ptr = &indices_(begin_cursor, 0);
    const int32_t* values_ptr = &values_(begin_cursor);
    const float* weights_ptr = has_weights ? &weights_(begin_cursor) : nullptr;
    for (int cursor = begin_cursor; cursor < end_cursor; ++cursor) {
      const int sample_id = *indices_ptr;
      const int adjusted_sample_id =
          sample_id - start_index + row_offset_per_device;

      coo_tensors.emplace_back(
          adjusted_sample_id,
          GetColIdInline(*values_ptr, col_shift, col_offset, num_scs_mod,
                         num_scs_mod_inv),
          has_weights ? *weights_ptr : 1.0f);

      indices_ptr += index_stride;
      values_ptr += value_stride;
      weights_ptr += weight_stride;
    }
  }

 private:
  // Returns a tuple of the range of elements in the indices array for samples
  // between start_index and end_index.
  std::tuple<int, int> GetElemRange(int start_index, int end_index) const {
    int begin_cursor = -1;
    int end_cursor = -1;
    for (int i = 0; i < indices_.shape(0); ++i) {
      const auto row = indices_(i, 0);
      if (row >= start_index && row < end_index) {
        if (begin_cursor == -1) {
          begin_cursor = i;
        }
        end_cursor = i;
      }
    }
    CHECK_GE(begin_cursor, 0);
    CHECK_GT(end_cursor, 0);
    return std::make_tuple(begin_cursor, end_cursor + 1);
  }

  bool HasWeights() const { return weights_.size() > 0; }

  int index_stride;
  int value_stride;
  int weight_stride;
  py::detail::unchecked_reference<int64_t, 2> indices_;
  py::detail::unchecked_reference<int32_t, 1> values_;
  py::detail::unchecked_reference<float, 1> weights_;
};

// This function handles one local device and one stacked table, which feeds to
// multiple features (due to feature stacking) and potentially multiple tables
// (due to table stacking). Returns a tuple of data for the stacked tables on
// the current device:
// 1. All COO tensors.
// 2. Batch size.
std::tuple<std::vector<CooFormat>, int>
GetCooTensorsForStackedTablesOnDeviceUnlocked(
    const int local_batch_size,
    const std::vector<FeatureWeightRepresentation>& features,
    const std::vector<StackedTableMetadata>& stacked_table_metadata,
    const int local_device_id, const int local_device_count,
    const int global_device_count, const int num_sc_per_device) {
  tsl::profiler::TraceMe t("GetCooTensorsForAllTablesOnDeviceUnlocked");
  const int num_scs = num_sc_per_device * global_device_count;
  std::vector<CooFormat> coo_tensors;
  int batch_size_for_device = 0;

  // Iterate through all features that have been stacked into the same table.
  for (const auto& metadata : stacked_table_metadata) {
    const int feature_index = metadata.feature_index;
    const int row_offset = metadata.row_offset;
    const int col_offset = metadata.col_offset;
    const int col_shift = metadata.col_shift;

    const auto& curr_feature = features[feature_index];

    // Split the feature and feature weights into per-device spans.
    const int num_samples = local_batch_size;
    const int num_samples_per_split = num_samples / local_device_count;
    const int start_index = local_device_id * num_samples_per_split;
    int end_index = (local_device_id + 1) * num_samples_per_split;
    if (local_device_id == local_device_count - 1) {
      // Just in case the last split is not a full batch.
      end_index = num_samples;
    }

    batch_size_for_device += (end_index - start_index);

    // In the case of feature stacking, we need to group all the COO
    // tensors at this stage (i.e., before the sorting later on).
    curr_feature.ExtractCooTensors(start_index, end_index, row_offset,
                                   col_offset, col_shift, num_scs,
                                   global_device_count, coo_tensors);
  }

  return std::make_tuple(std::move(coo_tensors), batch_size_for_device);
}

/*
Returns a tuple of data for all stacked tables on all local devices, and all
local sparsecores:
1. Whether mini-batching is needed.
2. All COO tensors.
3. Id counter.
4. Unique id counter.
5. Id drop counter.
*/
std::tuple<bool, DeviceBatchSizeLists, DeviceSparsecoreDataLists<CooFormat>,
           AggregatedIdCounters, AggregatedIdCounters, AggregatedIdCounters>
SortDeviceListOfCooTensorsWithIdDropUnlocked(
    const int local_batch_size,
    const std::vector<FeatureWeightRepresentation>& features,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const bool drop_ids, const int local_device_count,
    const int global_device_count, const int num_sc_per_device) {
  tsl::profiler::TraceMe t("SortDeviceListOfCooTensorsWithIdDropUnlocked");

  const int num_scs = num_sc_per_device * global_device_count;

  absl::BlockingCounter counter(stacked_tables.size());

  struct {
    absl::Mutex mutex;
    bool mini_batching_needed = false;
    DeviceBatchSizeLists batch_sizes ABSL_GUARDED_BY(mutex);

    // Stacked table names to a list of COO tensors
    DeviceSparsecoreDataLists<CooFormat> coo_tensors ABSL_GUARDED_BY(mutex);

    // Stacked table names to a list of id counters
    AggregatedIdCounters max_id_counters ABSL_GUARDED_BY(mutex);
    AggregatedIdCounters max_unique_id_counters ABSL_GUARDED_BY(mutex);
    AggregatedIdCounters id_drop_counters ABSL_GUARDED_BY(mutex);
  } results;

  tsl::profiler::TraceMeProducer producer("SortListOfCooTensorsMainThread");
  {
    for (const auto& [stacked_table_name, stacked_table_metadata] :
         stacked_tables) {
      PreprocessingThreadPool()->Schedule([&, context_id =
                                                  producer.GetContextId()] {
        // Each thread handles one (stacked) table for all local devices.
        tsl::profiler::TraceMeConsumer consumer(
            [&] {
              return absl::StrCat("InputPreprocessingTable-",
                                  stacked_table_name);
            },
            context_id);

        // The following lists contains data for this
        // stacked table to be processed by all local devices.
        // 1st dimension is for local devices.
        // 2nd dimension is for SCs per device.
        // 3rd dimension is for the sparsecore-local list of data.
        DeviceSparsecoreDataList<CooFormat> coo_tensors_for_current_table(
            local_device_count);
        AggregatedIdCounter max_id_counter_for_current_table(
            local_device_count);
        AggregatedIdCounter max_unique_id_counter_for_current_table(
            local_device_count);
        AggregatedIdCounter id_drop_counter_for_current_table(
            local_device_count);
        bool mini_batching_needed_for_current_table = false;
        DeviceBatchSizeList batch_size_per_sc_for_current_table(
            local_device_count);

        // Temporary storage for the per-sparsecore data.
        // Avoid reallocations by pre-allocating the vectors. The keys could
        // grow pretty large.
        std::vector<int32_t> max_ids_per_sc_temp_storage(num_scs, 0);
        std::vector<int32_t> max_unique_ids_per_sc_temp_storage(num_scs, 0);
        std::vector<int32_t> id_drop_counter_per_sc_temp_storage(num_scs, 0);
        std::vector<uint64_t> keys_temp_storage;

        for (int local_device = 0; local_device < local_device_count;
             ++local_device) {
          //
          // Per-device Step 1: Extract the COO tensors for each table.
          //
          auto [coo_tensors_for_device, batch_size_for_device] =
              GetCooTensorsForStackedTablesOnDeviceUnlocked(
                  local_batch_size, features, stacked_table_metadata,
                  local_device, local_device_count, global_device_count,
                  num_sc_per_device);

          //
          // Per-device Step 2: Sort the COO tensors and group them by SC.
          //
          const int batch_size_per_sc =
              CeilOfRatio(batch_size_for_device, num_sc_per_device);
          const int approximate_num_coo_tensors_per_sc =
              coo_tensors_for_device.size() / num_sc_per_device + 1;

          // Make sure the keys are large enough to hold at least these many
          // elements.
          keys_temp_storage.reserve(batch_size_per_sc);

          auto [mini_batching_needed_for_current_device, coo_tensors_by_sc,
                max_id_counter_by_sc, max_unique_id_counter_by_sc,
                id_drop_counter_by_sc] =
              SortAndGroupCooTensorsWithIdDrop(
                  coo_tensors_for_device, drop_ids, num_scs, num_sc_per_device,
                  batch_size_per_sc,
                  stacked_table_metadata[0].max_ids_per_partition,
                  stacked_table_metadata[0].max_unique_ids_per_partition,
                  approximate_num_coo_tensors_per_sc,
                  max_ids_per_sc_temp_storage,
                  max_unique_ids_per_sc_temp_storage,
                  id_drop_counter_per_sc_temp_storage, keys_temp_storage);

          mini_batching_needed_for_current_table |=
              mini_batching_needed_for_current_device;

          batch_size_per_sc_for_current_table[local_device] =
              batch_size_for_device / num_sc_per_device;
          coo_tensors_for_current_table[local_device] =
              std::move(coo_tensors_by_sc);
          max_id_counter_for_current_table[local_device] =
              std::move(max_id_counter_by_sc);
          max_unique_id_counter_for_current_table[local_device] =
              std::move(max_unique_id_counter_by_sc);
          id_drop_counter_for_current_table[local_device] =
              std::move(id_drop_counter_by_sc);
        }

        // Save the COO tensors for this table for all local devices.
        {
          absl::MutexLock lock(&results.mutex);
          results.mini_batching_needed |=
              mini_batching_needed_for_current_table;
          results.batch_sizes[stacked_table_name] =
              std::move(batch_size_per_sc_for_current_table);
          results.coo_tensors[stacked_table_name.c_str()] =
              std::move(coo_tensors_for_current_table);
          results.max_id_counters[stacked_table_name.c_str()] =
              std::move(max_id_counter_for_current_table);
          results.max_unique_id_counters[stacked_table_name.c_str()] =
              std::move(max_unique_id_counter_for_current_table);
          results.id_drop_counters[stacked_table_name.c_str()] =
              std::move(id_drop_counter_for_current_table);
        }
        counter.DecrementCount();
      }  // End of lambda for threaded task.
      );  // End of Schedule.
    }
    counter.Wait();
  }

  absl::MutexLock lock(&results.mutex);

  return std::make_tuple(
      std::move(results.mini_batching_needed), std::move(results.batch_sizes),
      std::move(results.coo_tensors), std::move(results.max_id_counters),
      std::move(results.max_unique_id_counters),
      std::move(results.id_drop_counters));
}

int GetMiniBatchSize(const py::dict& mini_batching_config) {
  return mini_batching_config["MINI_BATCH_SIZE"].cast<int>();
}

MiniBatchingMode GetMiniBatchingMode(const py::dict& mini_batching_config) {
  int mode = mini_batching_config["MINI_BATCH_MODE"].cast<int>();
  switch (mode) {
    case static_cast<int>(MiniBatchingMode::kNone):
      return MiniBatchingMode::kNone;
    case static_cast<int>(MiniBatchingMode::kVocabularyDimension):
      return MiniBatchingMode::kVocabularyDimension;
    case static_cast<int>(MiniBatchingMode::kSampleDimension):
      return MiniBatchingMode::kSampleDimension;
    case static_cast<int>(MiniBatchingMode::kExperimentalForceVocabularyDiv):
      return MiniBatchingMode::kExperimentalForceVocabularyDiv;
    case static_cast<int>(MiniBatchingMode::kExperimentalForceVocabularyMod):
      return MiniBatchingMode::kExperimentalForceVocabularyMod;
    default:
      throw std::invalid_argument("Not supported mini-batching mode.");
  }
}

py::tuple _PreprocessSparseDenseMatmulInput(
    const int local_batch_size,
    const std::vector<FeatureWeightRepresentation>& features,
    const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>&
        stacked_tables,
    const py::dict& mini_batching_config, const int local_device_count,
    const int global_device_count, const int static_buffer_size_multiplier,
    const int num_sc_per_device, const int sparsecore_register_width,
    const int sharding_strategy, const bool has_leading_dimension) {
  tsl::profiler::TraceMe t("_PreprocessSparseDenseMatmulInput");

  if (has_leading_dimension != true) {
    throw std::invalid_argument(
        "Currently, only leading dimension is supported for mini-batching.");
  }

  // GIL is held when we enter this function.
  py::dict lhs_row_pointers;
  py::dict lhs_embedding_ids;
  py::dict lhs_sample_ids;
  py::dict lhs_gains;
  py::dict id_counter_per_table;
  py::dict unique_id_counter_per_table;
  py::dict id_drop_counter_per_table;
  int mini_batch_size = GetMiniBatchSize(mini_batching_config);
  MiniBatchingMode mini_batching_mode =
      GetMiniBatchingMode(mini_batching_config);
  {
    // Release GIL here as we don't need python objects after this point.
    py::gil_scoped_release main_release;

    // Sort COO tensors and group them by SC.
    // Note this function would release and reacquire the GIL.
    // If mini-batching mode is set to NONE, embedding ids beyond limitations
    // are directly dropped.
    auto [mini_batching_needed, batch_sizes, coo_tensors, id_counters,
          unique_id_counters, id_drop_counters] =
        SortDeviceListOfCooTensorsWithIdDropUnlocked(
            local_batch_size, features, stacked_tables,
            mini_batching_mode == MiniBatchingMode::kNone, local_device_count,
            global_device_count, num_sc_per_device);

    {
      py::gil_scoped_acquire acq;
      Reshape2dToPyDictLocked(id_counter_per_table, id_counters);
      Reshape2dToPyDictLocked(unique_id_counter_per_table, unique_id_counters);
      Reshape2dToPyDictLocked(id_drop_counter_per_table, id_drop_counters);
    }

    // Communicate with other tasks to see if mini-batching is needed.
    // Here we assume the mini-batching size is the same across all tables.

    // If mini-batching is needed in this task, determine the split points.
    // Here we assume a simple mod-N split.
    mini_batching_needed = true;

    // Communicate with other tasks to reach consensus on mini-batching split
    // points.

    // In this prototype, we always use mini-batching if allowed.
    if (mini_batching_mode == MiniBatchingMode::kNone) {
      // No mini-batching.
      mini_batching_mode = MiniBatchingMode::kExperimentalForceVocabularyDiv;
      // force the mini-batch size to be 1.
      mini_batch_size = 1;
    }

    if (mini_batching_mode ==
        MiniBatchingMode::kExperimentalForceVocabularyDiv) {
      DeviceSparsecoreMiniBatchingDataLists<CooFormat> split_coo_tensors =
          SplitCooTensorsByVocabularyDiv(stacked_tables, mini_batch_size,
                                         coo_tensors);

      auto [row_pointers, embedding_ids, sample_ids, gains, buffer_sizes] =
          EncodeMiniBatchingDataUnlocked(
              split_coo_tensors, stacked_tables, batch_sizes,
              local_device_count, global_device_count, num_sc_per_device,
              sparsecore_register_width, has_leading_dimension,
              static_buffer_size_multiplier);
      {
        py::gil_scoped_acquire acq;
        Convert2dToPyDictLocked(lhs_row_pointers, row_pointers);
        Extend2dToPyDictLocked(lhs_embedding_ids, embedding_ids, buffer_sizes);
        Extend2dToPyDictLocked(lhs_sample_ids, sample_ids, buffer_sizes);
        Extend2dToPyDictLocked(lhs_gains, gains, buffer_sizes);
      }
    } else if (mini_batching_mode ==
               MiniBatchingMode::kExperimentalForceVocabularyMod) {
      // Note modulus is the mini-batch size here.
      DeviceSparsecoreMiniBatchingDataLists<CooFormat> split_coo_tensors =
          SplitCooTensorsByVocabularyMod(stacked_tables, mini_batch_size,
                                         coo_tensors);

      auto [row_pointers, embedding_ids, sample_ids, gains, buffer_sizes] =
          EncodeMiniBatchingDataUnlocked(
              split_coo_tensors, stacked_tables, batch_sizes,
              local_device_count, global_device_count, num_sc_per_device,
              sparsecore_register_width, has_leading_dimension,
              static_buffer_size_multiplier);
      {
        py::gil_scoped_acquire acq;
        Convert2dToPyDictLocked(lhs_row_pointers, row_pointers);
        Extend2dToPyDictLocked(lhs_embedding_ids, embedding_ids, buffer_sizes);
        Extend2dToPyDictLocked(lhs_sample_ids, sample_ids, buffer_sizes);
        Extend2dToPyDictLocked(lhs_gains, gains, buffer_sizes);
      }
    } else {
      throw std::invalid_argument("Not supported mini-batching mode.");
    }
  }

  py::dict stats;
  stats["max_ids"] = std::move(id_counter_per_table);
  stats["max_unique_ids"] = std::move(unique_id_counter_per_table);
  stats["id_drop_counters"] = std::move(id_drop_counter_per_table);
  return py::make_tuple(lhs_row_pointers, lhs_embedding_ids, lhs_sample_ids,
                        lhs_gains, mini_batch_size, stats);
}

py::tuple PreprocessSparseDenseMatmulInputWithBCOO(
    const int local_batch_size, const py::list& indices, const py::list& values,
    const py::list& weights, const py::list& feature_specs,
    const py::dict& mini_batching_config, const int local_device_count,
    const int global_device_count, const int static_buffer_size_multiplier,
    const int num_sc_per_device, const int sparsecore_register_width,
    const int sharding_strategy, const bool has_leading_dimension) {
  tsl::profiler::TraceMe t("PreprocessSparseDenseMatmulInputWithBCOO");

  const auto num_features = indices.size();
  CHECK_EQ(num_features, indices.size());
  CHECK_EQ(num_features, values.size());
  CHECK_EQ(num_features, weights.size());
  CHECK_EQ(num_features, feature_specs.size());

  py::array_t<float> dummy_weights(0);
  auto dummy_weights_ref = dummy_weights.unchecked<1>();

  std::vector<FeatureWeightRepresentation> sparse_features;
  sparse_features.reserve(num_features);

  // Fill the sparse features and weights from the BCOO tensors.
  for (int feature_index = 0; feature_index < num_features; ++feature_index) {
    const auto& current_values =
        values[feature_index].cast<py::array_t<int32_t>>();

    const auto& current_index =
        indices[feature_index].cast<py::array_t<int64_t>>();

    if (!weights[feature_index].is_none()) {
      const auto& current_weights =
          weights[feature_index].cast<py::array_t<float>>();
      sparse_features.emplace_back(current_index.unchecked<2>(),
                                   current_values.unchecked<1>(),
                                   current_weights.unchecked<1>());
    } else {
      sparse_features.emplace_back(current_index.unchecked<2>(),
                                   current_values.unchecked<1>(),
                                   dummy_weights_ref);
    }
  }

  // Get the stacked table metadata for each top level table.
  // The keys are stacked table names (or the table itself if not stacked) and
  // the values are a vector of StackedTableMetadata for each feature that is
  // mapped to the table.
  const absl::flat_hash_map<std::string, std::vector<StackedTableMetadata>>
      stacked_tables = GetStackedTableMetadata(feature_specs, local_batch_size);

  return _PreprocessSparseDenseMatmulInput(
      local_batch_size, sparse_features, stacked_tables, mini_batching_config,
      local_device_count, global_device_count, static_buffer_size_multiplier,
      num_sc_per_device, sparsecore_register_width, sharding_strategy,
      has_leading_dimension);
}

}  // namespace

PYBIND11_MODULE(input_preprocessing_with_mini_batching_cc, m) {
  m.def("PreprocessSparseDenseMatmulInputWithBCOO",
        &PreprocessSparseDenseMatmulInputWithBCOO,
        pybind11::arg("local_batch_size"), pybind11::arg("indices"),
        pybind11::arg("values"), pybind11::arg("weights"),
        pybind11::arg("feature_specs"), pybind11::arg("mini_batching_config"),
        pybind11::arg("local_device_count"),
        pybind11::arg("global_device_count"),
        pybind11::arg("static_buffer_size_multiplier"),
        pybind11::arg("num_sc_per_device"),
        pybind11::arg("sparsecore_register_width"),
        pybind11::arg("sharding_strategy"),
        pybind11::arg("has_leading_dimension"));
}

}  // namespace jax_sc_embedding
