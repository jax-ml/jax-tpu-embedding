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
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_megacore_util.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/numeric/bits.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "Eigen/Core"  // from @eigen_archive
#include "jax_tpu_embedding/sparsecore/lib/core/coo_format.h"
#include "jax_tpu_embedding/sparsecore/lib/core/input_preprocessing_util.h"
#include "jax_tpu_embedding/sparsecore/lib/core/partitioned_coo_tensors.h"
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"

namespace jax_sc_embedding {
namespace {

// Number of per-SC inputs stored inline (without heap allocation) when
// filling Megacore CSR buffers.
constexpr int kInlinedScPerDevice = 16;

// Pads the Megacore COO buffer with (INT_MAX, INT_MAX, 0.0f) up to the next HBM
// word boundary, without exceeding `coo_end`. `INT_MAX` padding IDs sort after
// all valid IDs, so consumers that sort the whole buffer keep padding last.
void AlignMegacoreCooBuffer(int& coo_index, int coo_end,
                            int hbm_word_size_in_4b,
                            internal::CsrArraysRefPerDevice& csr) {
  while (coo_index % hbm_word_size_in_4b != 0 && coo_index < coo_end) {
    csr.embedding_ids[coo_index] = std::numeric_limits<int32_t>::max();
    csr.sample_ids[coo_index] = std::numeric_limits<int32_t>::max();
    csr.gains[coo_index] = 0.0f;
    ++coo_index;
  }
}

template <typename CooT>
int GlobalScId(const CooT& coo, uint32_t sc_mask) {
  using UColIdT = std::make_unsigned_t<decltype(CooT::col_id)>;
  return static_cast<int>(static_cast<UColIdT>(coo.col_id) &
                          static_cast<UColIdT>(sc_mask));
}

// Sets `offsets[g]` to the index of the first element of `coo_tensors` whose
// global SparseCore ID is >= `g`. `coo_tensors` must be sorted by
// `(global_sc_id, emb_id)`.
template <typename CooT>
void FindGlobalScOffsets(absl::Span<const CooT> coo_tensors, uint32_t sc_mask,
                         absl::Span<int> offsets) {
  auto it = coo_tensors.begin();
  for (int g = 0; g < offsets.size(); ++g) {
    it = std::partition_point(
        it, coo_tensors.end(),
        [&](const CooT& coo) { return GlobalScId(coo, sc_mask) < g; });
    offsets[g] = static_cast<int>(it - coo_tensors.begin());
  }
}

// Precomputed per-partition offsets for parallel Megacore CSR buffer filling.
struct MegacoreFillPlan {
  // Flat array of size `num_sc_per_device * (num_scs + 1)`.
  // `sc_offsets[sc_id * (num_scs + 1) + g]` is the starting index of partition
  // `g` in `grouped_coo_tensors(sc_id, 0)`.
  std::vector<int> sc_offsets;
  // Output COO buffer range `[partition_coo_begin[g], partition_coo_end[g])`
  // for each `global_sc_id` `g` in `[0, num_scs)`.
  std::vector<int> partition_coo_begin;
  std::vector<int> partition_coo_end;
};

// Copies sorted `coo_tensors` for a single SparseCore into the output CSR
// slice, which must be at least as large as `coo_tensors`.
template <typename CooT>
void CopyMegacoreCooTensors(absl::Span<const CooT> coo_tensors,
                            int dest_sc_offset, int num_sc_bits,
                            int max_row_id_exclusive,
                            Eigen::Ref<RowVectorXi> embedding_ids,
                            Eigen::Ref<RowVectorXi> sample_ids,
                            Eigen::Ref<RowVectorXf> gains) {
  using UColIdT = std::make_unsigned_t<decltype(CooT::col_id)>;
  DCHECK_LE(coo_tensors.size(), embedding_ids.size());
  const int count = coo_tensors.size();
  const auto coo_it = coo_tensors.begin();
  for (int i = 0; i < count; ++i) {
    const CooT& coo = coo_it[i];
    DCHECK_GE(coo.col_id, 0);
    DCHECK_LT(coo.row_id, max_row_id_exclusive);
    const UColIdT emb_id = static_cast<UColIdT>(coo.col_id) >> num_sc_bits;
    embedding_ids[i] = dest_sc_offset + static_cast<int>(emb_id);
    sample_ids[i] = coo.row_id;
    gains[i] = coo.gain;
  }
}

// The unmerged remainder `[head, end)` of one local SparseCore's COO tensors in
// `FillMegacorePartition`. `head_col_id` caches `head->col_id` for the
// min-scan, and `last_col_id` caches `(end - 1)->col_id` to detect the last
// run.
template <typename CooT>
struct ScPartitionInput {
  using ColIdT = decltype(CooT::col_id);
  typename absl::Span<const CooT>::const_iterator head;
  typename absl::Span<const CooT>::const_iterator end;
  ColIdT head_col_id;
  ColIdT last_col_id;
};

// Counts how many elements of partition `g` fit within `max_elements` and
// `max_unique_ids`, adding the number of distinct `col_id`s consumed to
// `device_unique_ids`.
template <typename CooT>
int CountMegacorePartitionElements(
    const DevicePartitionedCooTensorsT<CooT>& grouped_coo_tensors,
    const MegacoreFillPlan& plan, int g, int num_scs, int num_sc_per_device,
    int max_elements, int max_unique_ids, int& device_unique_ids) {
  using ColIdT = decltype(CooT::col_id);
  absl::InlinedVector<absl::Span<const CooT>, kInlinedScPerDevice> sc_spans;
  for (int sc_id = 0; sc_id < num_sc_per_device; ++sc_id) {
    const int offset_base = sc_id * (num_scs + 1) + g;
    const int begin = plan.sc_offsets[offset_base];
    const int count = plan.sc_offsets[offset_base + 1] - begin;
    if (count > 0) {
      sc_spans.push_back(
          grouped_coo_tensors(sc_id, /*bucket_id=*/0).subspan(begin, count));
    }
  }

  int written = 0;
  while (written < max_elements && device_unique_ids < max_unique_ids) {
    ColIdT min_col_id = 0;
    bool found = false;
    for (absl::Span<const CooT> span : sc_spans) {
      if (!span.empty() && (!found || span.front().col_id < min_col_id)) {
        min_col_id = span.front().col_id;
        found = true;
      }
    }
    if (!found) {
      break;
    }
    ++device_unique_ids;
    for (absl::Span<const CooT>& span : sc_spans) {
      while (!span.empty() && span.front().col_id == min_col_id) {
        span.remove_prefix(1);
        ++written;
      }
    }
  }
  return std::min(written, max_elements);
}

// Performs a k-way merge of the non-empty per-local-SC `sc_coo_tensors` for a
// single `global_sc_id` partition into the output CSR slice (`embedding_ids`,
// `sample_ids`, `gains`).
//
// Each element of `sc_coo_tensors` is already sorted by `(col_id, row_id)`, and
// `sc_coo_tensors` is ordered by `local_sc_id` (which partitions `row_id` into
// disjoint contiguous ranges `[sc_id * batch_size_per_sc, (sc_id + 1) *
// batch_size_per_sc)`). All elements in this partition share the same
// `global_sc_id` in the low `num_sc_bits` bits of `col_id`, so comparing raw
// `col_id` values is equivalent to comparing `emb_id = col_id >> num_sc_bits`,
// and ties in `col_id` naturally resolve to the lowest `local_sc_id` (smallest
// `row_id`).
//
// The output slice must hold exactly the total number of elements across
// `sc_coo_tensors`, so the merge needs no bounds checks.
template <typename CooT>
void FillMegacorePartition(
    absl::Span<const absl::Span<const CooT>> sc_coo_tensors, int dest_sc_offset,
    int num_sc_bits, int max_row_id_exclusive,
    Eigen::Ref<RowVectorXi> embedding_ids, Eigen::Ref<RowVectorXi> sample_ids,
    Eigen::Ref<RowVectorXf> gains) {
  using ColIdT = decltype(CooT::col_id);
  using UColIdT = std::make_unsigned_t<ColIdT>;
  Eigen::Index out_index = 0;
  DCHECK_EQ(sample_ids.size(), embedding_ids.size());
  DCHECK_EQ(gains.size(), embedding_ids.size());

  absl::InlinedVector<ScPartitionInput<CooT>, kInlinedScPerDevice> sc_inputs;
  Eigen::Index total_count = 0;
  for (absl::Span<const CooT> coo_tensors : sc_coo_tensors) {
    sc_inputs.push_back({.head = coo_tensors.begin(),
                         .end = coo_tensors.end(),
                         .head_col_id = coo_tensors.front().col_id,
                         .last_col_id = coo_tensors.back().col_id});
    total_count += coo_tensors.size();
  }
  DCHECK_EQ(total_count, embedding_ids.size());
  // The active per-SC inputs are `[sc_inputs_begin, sc_inputs_end)`, kept
  // ordered by `local_sc_id`.
  const auto sc_inputs_begin = sc_inputs.begin();
  auto sc_inputs_end = sc_inputs.end();

  // Copies the remainder of `sc_input` to the output.
  auto copy_remainder = [&](const ScPartitionInput<CooT>& sc_input) {
    const Eigen::Index count = sc_input.end - sc_input.head;
    CopyMegacoreCooTensors<CooT>(
        absl::MakeConstSpan(sc_input.head, sc_input.end), dest_sc_offset,
        num_sc_bits, max_row_id_exclusive,
        embedding_ids.segment(out_index, count),
        sample_ids.segment(out_index, count), gains.segment(out_index, count));
    out_index += count;
  };

  while (sc_inputs_end - sc_inputs_begin > 1) {
    auto min_sc_input = sc_inputs_begin;
    ColIdT min_col_id = min_sc_input->head_col_id;
    for (auto sc_input = min_sc_input + 1; sc_input != sc_inputs_end;
         ++sc_input) {
      if (sc_input->head_col_id < min_col_id) {
        min_sc_input = sc_input;
        min_col_id = sc_input->head_col_id;
      }
    }
    DCHECK_GE(min_col_id, 0);

    // Copy the contiguous run of `min_col_id` from `*min_sc_input`. Because
    // lower `local_sc_id`s own smaller `row_id`s, all elements with
    // `col_id == min_col_id` in `*min_sc_input` precede any elements with the
    // same `col_id` in later `sc_inputs`.
    if (min_col_id == min_sc_input->last_col_id) {
      // The run is the whole remainder of `*min_sc_input` (e.g. low-vocabulary
      // inputs), so copy it in one tight loop, then remove the exhausted
      // `sc_input` by shifting the remaining ones left so `sc_inputs` stays
      // ordered by `local_sc_id` (preserving `row_id` tie-breaking) and the
      // last remaining `sc_input` lands at `*sc_inputs_begin`.
      copy_remainder(*min_sc_input);
      std::copy(min_sc_input + 1, sc_inputs_end, min_sc_input);
      --sc_inputs_end;
    } else {
      const UColIdT emb_id = static_cast<UColIdT>(min_col_id) >> num_sc_bits;
      const int linearized_emb_id = dest_sc_offset + static_cast<int>(emb_id);
      // `min_sc_input->last_col_id != min_col_id`, so the run ends before
      // `*min_sc_input` is exhausted and no end check is needed.
      auto coo_it = min_sc_input->head;
      do {
        DCHECK_LT(coo_it->row_id, max_row_id_exclusive);
        embedding_ids[out_index] = linearized_emb_id;
        sample_ids[out_index] = coo_it->row_id;
        gains[out_index] = coo_it->gain;
        ++out_index;
        ++coo_it;
      } while (coo_it->col_id == min_col_id);
      min_sc_input->head = coo_it;
      min_sc_input->head_col_id = coo_it->col_id;
    }
  }

  // Exactly one `sc_input` remains, so copy its remainder in one tight loop.
  copy_remainder(*sc_inputs_begin);
}

// Fills the Megacore COO buffer for partitions `g` in
// `[partition_begin, partition_end)` using the precomputed `plan`.
template <typename CooT>
void FillMegacorePartitionRange(
    const DevicePartitionedCooTensorsT<CooT>& grouped_coo_tensors,
    const MegacoreFillPlan& plan, int partition_begin, int partition_end,
    int batch_size_per_sc, int rows_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    internal::CsrArraysRefPerDevice& csr_arrays) {
  const int num_scs = options.GetNumScs();
  const int num_sc_per_device = options.num_sc_per_device;
  const int num_sc_bits = absl::countr_zero(static_cast<uint32_t>(num_scs));
  const int max_row_id_exclusive = batch_size_per_sc * num_sc_per_device;

  absl::InlinedVector<absl::Span<const CooT>, kInlinedScPerDevice>
      sc_coo_tensors;
  for (int g = partition_begin; g < partition_end; ++g) {
    const int coo_index = plan.partition_coo_begin[g];
    const int partition_size = plan.partition_coo_end[g] - coo_index;
    if (partition_size == 0) {
      continue;
    }
    sc_coo_tensors.clear();
    int total_count = 0;
    for (int sc_id = 0; sc_id < num_sc_per_device; ++sc_id) {
      const int offset_base = sc_id * (num_scs + 1) + g;
      const int begin = plan.sc_offsets[offset_base];
      const int count = plan.sc_offsets[offset_base + 1] - begin;
      if (count > 0) {
        sc_coo_tensors.push_back(
            grouped_coo_tensors(sc_id, /*bucket_id=*/0).subspan(begin, count));
        total_count += count;
      }
    }
    const int dest_sc = g & (num_sc_per_device - 1);
    const int dest_sc_offset = static_cast<int>(int64_t{dest_sc} * rows_per_sc);
    if (total_count == partition_size) {
      FillMegacorePartition<CooT>(
          sc_coo_tensors, dest_sc_offset, num_sc_bits, max_row_id_exclusive,
          csr_arrays.embedding_ids.segment(coo_index, partition_size),
          csr_arrays.sample_ids.segment(coo_index, partition_size),
          csr_arrays.gains.segment(coo_index, partition_size));
      continue;
    }
    // Either a per-device row limit (`max_ids_per_device` or
    // `max_unique_ids_per_device`) or the static COO buffer limit was reached
    // and IDs are dropped. `FillMegacorePartition` needs room for the whole
    // partition, so merge it into scratch buffers and keep its first
    // `partition_size` elements.
    RowVectorXi embedding_ids(total_count);
    RowVectorXi sample_ids(total_count);
    RowVectorXf gains(total_count);
    FillMegacorePartition<CooT>(sc_coo_tensors, dest_sc_offset, num_sc_bits,
                                max_row_id_exclusive, embedding_ids, sample_ids,
                                gains);
    csr_arrays.embedding_ids.segment(coo_index, partition_size) =
        embedding_ids.head(partition_size);
    csr_arrays.sample_ids.segment(coo_index, partition_size) =
        sample_ids.head(partition_size);
    csr_arrays.gains.segment(coo_index, partition_size) =
        gains.head(partition_size);
  }
}

}  // namespace

template <typename CooT>
tsl::AsyncValueRef<int> FillMegacoreLocalDeviceBufferAsync(
    const DevicePartitionedCooTensorsT<CooT>& grouped_coo_tensors,
    const int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::string_view stacked_table_name, MegacoreCsrOptions megacore_csr,
    internal::CsrArraysRefPerDevice csr_arrays) {
  DCHECK_GT(batch_size_per_sc, 0);
  DCHECK(options.num_sc_per_device == 1 || megacore_csr.rows_per_sc > 0);
  DCHECK_GT(megacore_csr.max_ids_per_device, 0);
  DCHECK_GT(megacore_csr.max_unique_ids_per_device, 0);
  DCHECK_EQ(grouped_coo_tensors.GetNumMinibatches(), 1);

  const int num_scs = options.GetNumScs();
  const int num_sc_per_device = options.num_sc_per_device;
  const uint32_t sc_mask = static_cast<uint32_t>(num_scs - 1);
  const int coo_end = csr_arrays.embedding_ids.size();

  auto plan = std::make_shared<MegacoreFillPlan>();
  plan->sc_offsets.resize(num_sc_per_device * (num_scs + 1));
  plan->partition_coo_begin.resize(num_scs);
  plan->partition_coo_end.resize(num_scs);

  int total_coo_size = 0;
  for (int sc_id = 0; sc_id < num_sc_per_device; ++sc_id) {
    absl::Span<const CooT> coo_tensors =
        grouped_coo_tensors(sc_id, /*bucket_id=*/0);
    total_coo_size += coo_tensors.size();
    absl::Span<int> offsets = absl::MakeSpan(plan->sc_offsets)
                                  .subspan(sc_id * (num_scs + 1), num_scs + 1);
    FindGlobalScOffsets<CooT>(coo_tensors, sc_mask, offsets);
  }

  const int max_ids_per_device = options.allow_id_dropping
                                     ? megacore_csr.max_ids_per_device
                                     : std::numeric_limits<int>::max();
  const int max_unique_ids_per_device =
      options.allow_id_dropping ? megacore_csr.max_unique_ids_per_device
                                : std::numeric_limits<int>::max();
  // In device minibatching without ID dropping, unique IDs are neither capped
  // nor fatal, so skip the extra k-way merge count.
  const bool check_unique_ids =
      options.allow_id_dropping ||
      options.GetMinibatchingMode() != MinibatchingMode::kDevice;
  int observed_max_ids_per_device = 0;
  int observed_max_unique_ids_per_device = 0;
  int coo_index = 0;
  int processed = 0;
  int g = 0;
  absl::InlinedVector<int, kInlinedScPerDevice> partition_counts(
      num_sc_per_device);
  for (int device_id = 0; device_id < options.global_device_count;
       ++device_id) {
    int device_ids = 0;
    for (int dest_sc = 0; dest_sc < num_sc_per_device; ++dest_sc) {
      const int partition_g = device_id * num_sc_per_device + dest_sc;
      int partition_count = 0;
      for (int sc_id = 0; sc_id < num_sc_per_device; ++sc_id) {
        const int offset_base = sc_id * (num_scs + 1) + partition_g;
        partition_count +=
            plan->sc_offsets[offset_base + 1] - plan->sc_offsets[offset_base];
      }
      partition_counts[dest_sc] = partition_count;
      device_ids += partition_count;
    }

    // TODO: Skip per-device capping in MinibatchingMode::kDevice once device
    // minibatching is supported for Megacore CSR.
    const int row_end_limit = static_cast<int>(
        std::min<int64_t>(coo_end, int64_t{coo_index} + max_ids_per_device));
    int device_unique_ids = 0;
    for (int dest_sc = 0; dest_sc < num_sc_per_device; ++dest_sc, ++g) {
      plan->partition_coo_begin[g] = coo_index;
      const int max_elements = row_end_limit - coo_index;
      int written = 0;
      if (partition_counts[dest_sc] > 0) {
        if (!check_unique_ids ||
            device_ids <= megacore_csr.max_unique_ids_per_device) {
          written = std::min(partition_counts[dest_sc], max_elements);
        } else if (max_elements > 0 || !options.allow_id_dropping) {
          const int count_limit = options.allow_id_dropping
                                      ? max_elements
                                      : partition_counts[dest_sc];
          written = std::min(
              max_elements,
              CountMegacorePartitionElements<CooT>(
                  grouped_coo_tensors, *plan, g, num_scs, num_sc_per_device,
                  count_limit, max_unique_ids_per_device, device_unique_ids));
        }
      }
      coo_index += written;
      processed += written;
      plan->partition_coo_end[g] = coo_index;
    }
    if (!options.allow_id_dropping) {
      observed_max_ids_per_device =
          std::max(observed_max_ids_per_device, device_ids);
      observed_max_unique_ids_per_device =
          std::max(observed_max_unique_ids_per_device, device_unique_ids);
    }
    csr_arrays.row_pointers[device_id] = coo_index;
    AlignMegacoreCooBuffer(coo_index, coo_end, options.hbm_word_size_in_4b,
                           csr_arrays);
  }
  if (!options.allow_id_dropping) {
    if (options.GetMinibatchingMode() == MinibatchingMode::kDevice) {
      internal::LogIfDeviceMinibatchingRequired</*kPerDevice=*/true>(
          observed_max_ids_per_device, /*observed_max_unique_ids=*/0,
          megacore_csr.max_ids_per_device,
          megacore_csr.max_unique_ids_per_device, stacked_table_name);
    } else {
      internal::ValidateMaxIdsOrDie</*kPerDevice=*/true>(
          observed_max_ids_per_device, observed_max_unique_ids_per_device,
          megacore_csr.max_ids_per_device,
          megacore_csr.max_unique_ids_per_device, stacked_table_name,
          /*allow_id_dropping=*/false);
    }
  }
  for (int row_index = options.global_device_count;
       row_index < csr_arrays.row_pointers.size(); ++row_index) {
    csr_arrays.row_pointers[row_index] = coo_index;
  }
  // Pad the tail past the last aligned row as well, because device consumers
  // may sort and scan the whole buffer. The buffers may be reused across
  // batches, so this must be written every batch.
  // TODO(b/578149210): Remove the last-row tail padding once a dynamic slice
  // is used to bound the valid range.
  const int tail_size = coo_end - coo_index;
  csr_arrays.embedding_ids.segment(coo_index, tail_size)
      .setConstant(std::numeric_limits<int32_t>::max());
  csr_arrays.sample_ids.segment(coo_index, tail_size)
      .setConstant(std::numeric_limits<int32_t>::max());
  csr_arrays.gains.segment(coo_index, tail_size).setZero();
  int dropped_id_count = 0;
  if (processed < total_coo_size) {
    LOG_EVERY_N(WARNING, 100)
        << "The static buffer is too small or device ID limits were exceeded "
           "for the current batch. IDs will be dropped! Megacore COO buffer "
           "size: "
        << coo_end << ", total number of input COOs: " << total_coo_size
        << ", while currently processed only: " << processed;
    dropped_id_count = total_coo_size - processed;
  }

  if (processed == 0) {
    return tsl::MakeAvailableAsyncValueRef<int>(dropped_id_count);
  }

  const int max_tasks = options.num_sc_per_device;
  absl::InlinedVector<std::pair<int, int>, kInlinedScPerDevice> task_ranges;
  int current_partition = 0;
  for (int task_id = 1; task_id <= max_tasks && current_partition < num_scs;
       ++task_id) {
    int next_partition = num_scs;
    if (task_id < max_tasks) {
      const int target_coo =
          (static_cast<int64_t>(coo_index) * task_id) / max_tasks;
      const auto it =
          absl::c_lower_bound(plan->partition_coo_end, target_coo);
      next_partition = std::max<int>(
          current_partition,
          static_cast<int>(it - plan->partition_coo_end.begin()));
      if (next_partition < num_scs &&
          (next_partition == current_partition ||
           plan->partition_coo_end[next_partition] - target_coo <=
               target_coo - plan->partition_coo_begin[next_partition])) {
        ++next_partition;
      }
    }
    if (plan->partition_coo_begin[current_partition] <
        plan->partition_coo_end[next_partition - 1]) {
      task_ranges.push_back({current_partition, next_partition});
    }
    current_partition = next_partition;
  }

  tsl::CountDownAsyncValueRef<int> count_down(task_ranges.size(),
                                              dropped_id_count);
  for (int task_idx = 0; task_idx < task_ranges.size(); ++task_idx) {
    const auto [partition_begin, partition_end] = task_ranges[task_idx];
    options.async_task_scheduler([=, &grouped_coo_tensors]() mutable {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode(
            absl::StrCat("FillMegacoreBuffer/", stacked_table_name, "/T",
                         task_idx),
            {{"batch_number", options.batch_number}});
      });
      FillMegacorePartitionRange<CooT>(
          grouped_coo_tensors, *plan, partition_begin, partition_end,
          batch_size_per_sc, megacore_csr.rows_per_sc, options, csr_arrays);
      count_down.CountDown();
    });
  }
  return count_down.AsRef();
}

template tsl::AsyncValueRef<int> FillMegacoreLocalDeviceBufferAsync<CooFormat>(
    const DevicePartitionedCooTensors& grouped_coo_tensors,
    int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::string_view stacked_table_name, MegacoreCsrOptions megacore_csr,
    internal::CsrArraysRefPerDevice csr_arrays);

template tsl::AsyncValueRef<int>
FillMegacoreLocalDeviceBufferAsync<CooFormatT<int64_t>>(
    const DevicePartitionedCooTensorsT<CooFormatT<int64_t>>&
        grouped_coo_tensors,
    int batch_size_per_sc,
    const PreprocessSparseDenseMatmulInputOptions& options,
    absl::string_view stacked_table_name, MegacoreCsrOptions megacore_csr,
    internal::CsrArraysRefPerDevice csr_arrays);

}  // namespace jax_sc_embedding
