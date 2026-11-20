# JAX SparseCore AI Agent Directives

> **Google-internal contributors:** This file is written for the open-source
> layout of the repository (`jax_tpu_embedding/sparsecore/...` paths, `bazel`
> commands) and is exported to GitHub. The translation to the internal layout
> and all internal-only guidance live in
> [`_agents/AGENTS.md`](_agents/AGENTS.md), which is not exported. Never add
> internal paths, targets, or links to this file.

These guidelines enforce structural invariants for AI assistants operating
within the JAX SparseCore (`jax_tpu_embedding/sparsecore/`) codebase.

## Array vs. Sharding PyTree Annotations

**Rule**: Do NOT use `Array | Sharding` (or `Union[Array, Sharding]`) in PyTree
generic or leaf types. Keep all public PyTree leaf types strictly typed as
`Array` or `Tensor`.

**Why**: JAX pytrees are runtime constructs that cannot be statically typed
([JAX #26572][26572]). `jax.jit`'s `in_shardings` requires a pytree of
`Sharding` leaves structurally matching the input pytree of `Array` leaves.
`jax.jit` itself does not preserve callable signatures via `ParamSpec`
([JAX PR #14688][14688], still open). Bloating public types with `Sharding`
degrades API ergonomics for the 99% data-centric use case, and still cannot
satisfy the type checker for structural pytree matching.

**Resolution**: Suppress type errors at the *injection site* using
`# pyrefly: ignore[bad-argument-type]` with a brief comment. Do not alter the
class definition.

[26572]: https://github.com/jax-ml/jax/issues/26572#issuecomment-2665778697
[14688]: https://github.com/jax-ml/jax/pull/14688

## Input Preprocessing Performance Evidence

**Rule**: Treat C++ host input preprocessing in `sparsecore/lib/core/` as
performance-critical. Every change to its hot path needs a before/after run of
`sparsecore/lib/core/benchmarks:input_preprocessing_benchmark` (or
`sparsecore/lib/nn/tests:preprocess_input_benchmarks` for Python-bound input
batches), with the results (or the reason no run is needed) summarized in the
change description. For internal contributors, the internal review flow asks for
this as a `PERF=` tag and gives the exact command to run.

**Why**: This code runs on the host for every batch and every stacked table, so
small per-ID costs add up quickly. Past measurements show how sensitive it is:

*   Replacing integer division and modulo with masks and shifts in the buffer
    filling loop cut its wall time by 29% on dense tables and 8% on sparse ones,
    so even per-ID arithmetic in that loop is measurable.
*   Moving statistics updates out of the sort-and-group loop cut its latency by
    35-41%.
*   An in-loop deduplication tweak saved 13% of instructions on duplicate-heavy
    inputs but cost 0.6% on large-vocabulary inputs, so one change can help one
    workload and hurt another.

Unit and end-to-end tests check correctness, not speed, so none of these effects
would show up without a benchmark.

**Scope**: The hot path is the C++ code that runs per ID or per batch: COO
extraction, sort and group, and CSR buffer filling (`coo_format.h`,
`input_preprocessing*.{h,cc}`,
`{numpy,ragged_tensor,sparse_coo}_input_batch.{h,cc}`,
`process_coo_tensors_impl.h`, `partitioned_coo_tensors.h`, and
`sort_and_group_coo_tensors_impl.h`). Skipping the benchmark is appropriate only
when a change to those files cannot alter generated code on that path, for
example comments, renames, or code that runs once per table such as validation.
Changes restricted to unit tests (`tests/`), benchmarks (`benchmarks/`), gRPC
transport (`grpc/`), TPU primitives (`primitives/`), or Python reference modules
are outside this scope and need no performance note.

**Writing hot-path code**:

*   Keep the default path unchanged when adding a new mode. Select the mode at
    compile time (`if constexpr` or a template parameter) instead of adding
    runtime fields that every ID pays for. Comparing the optimized benchmark
    binary or the function's disassembly against the base revision confirms that
    the default path compiles to the same code (state the takeaway in a phrase
    rather than pasting `.text` or symbol byte counts).
*   Avoid integer division and modulo by runtime values in per-ID loops. The
    total SC count is validated to be a power of two, so use masks and shifts.
*   Avoid per-batch heap allocations and copies of COO data; reuse buffers.
*   Compute values that are fixed for a table (flags, strides, sizes) once, not
    per batch or per device.

**Benchmark coverage**: If a change adds a new hot path (a new mode, flag, or
layout), add a matching case to the benchmark in the same change, so the new
path has a baseline and future changes to it are measured.

**Resolution**: Run the benchmark on the final version of the change and report
one authoritative run per suite in a single compact Markdown table of at most 8
rows (or one sentence when no workload changes at p<0.05), following the
template in [Concise Change Descriptions](#concise-change-descriptions):

*   Rank rows by wall time (`time/op`), then CPU time (`cpu/op`), then
    allocations (`allocs/op`). Group combiners or workloads that move together.
*   Format `p < 0.05` deltas as `base -> new` (`-X.X%`) and neutral cells as
    `~`. Never cite the geomean, and ignore peak memory per op (`allocs/op`
    tracks memory).
*   Omit inline `(new - base) / base` arithmetic, duplicate `Nx` speedup
    multipliers, multiple p-value tests per cell, raw benchmark parameter
    suffixes (such as `/20/1/0/threads:8`) when the workload label identifies
    the case, preliminary reruns, and prose explaining noise on unchanged
    workloads.
*   Summarize remaining neutral cases in one line below the table, and explain
    any cross-metric tradeoff in one sentence.

## Verify Each Optimization Independently

**Rule**: When a change combines several optimizations (as a rule of thumb,
three or more), measure each one on its own with an ablation study instead of
only reporting the combined result. Benchmark the baseline, then either each
optimization alone or the full change with one optimization removed at a time,
using the same benchmark, workloads and settings. Prefer splitting independent
optimizations into separate changes, each with its own results.

**Why**: A combined result can hide opposite effects: one optimization can
regress a workload while another improves it by more, so the net gain looks fine
while a regression ships. Small effects also disappear into run-to-run noise.
Micro-optimizations such as `__restrict__` qualifiers, branch hints
(`ABSL_PREDICT_TRUE` / `ABSL_PREDICT_FALSE`), forced inlining or manual
unrolling often leave the generated code unchanged or make no measurable
difference, yet they add complexity and can become wrong as the code evolves.

**Resolution**: Keep an optimization only if it shows a statistically
significant improvement (p<0.05) on its own, ranked by wall time, then CPU time,
then allocations, or if it is needed for another kept optimization to work. For
hints that only steer the compiler, first check whether they change the
generated code by comparing the optimized benchmark binary or the function's
disassembly with and without them; drop those that do not. When optimizations
are split into separate changes, report only that change's benchmark table and
omit ablation logs and rejected experiments (note a non-obvious pitfall in a
one-line code comment if needed). Include a per-optimization breakdown in the
description only when one change bundles multiple optimizations, as one compact
table row per kept optimization.

## Concise Change Descriptions

**Rule**: Keep all prose in a change description (title, motivation paragraph,
and summary bullets, excluding benchmark tables and collapsed `<details>`
blocks) to **at most 150 words**.

**Why**: Reviewers and `git log` readers can already see modified files, symbol
names, and unit tests in the diff. Long code-heavy paragraphs, diff inventories,
and raw benchmark dumps hide why the change is needed and what tradeoffs it
makes.

**Resolution**: Follow this structure and template:

*   **Title (<= 65 characters)**: Imperative summary of the change.
*   **Motivation (1-3 sentences)**: State the problem or bottleneck in plain
    English (`Previously, ...`), without pasting multi-line code expressions or
    line numbers.
*   **Summary (2-4 bullets)**: State the key design changes only. Do not list
    modified file paths, recite the diff function by function, narrate added
    unit tests, paste `.o`/`.so` byte sizes, or list rejected experiments tried
    during exploration. Put incidental drive-by fixes in a collapsed `<details>`
    block (with a blank line after `<details>`) if worth noting at all.

```markdown
Store PartitionedCooTensorsT bucket offsets inline

Previously, `PartitionedCooTensorsT` stored `bucket_offsets_` in a
`std::vector<int>` and grew it inside `Add`, causing a heap allocation per
SparseCore partition and keeping capacity checks in the inner deduplication
loop.

* Store `bucket_offsets_` in `absl::InlinedVector<int,
  CooFormat::kMaxMinibatchingBuckets + 1>` (65 ints max) to avoid per-partition
  heap allocations.
* Replace the `push_back` loop in `AdvanceBucketOffsets` with an early return on
  bucket 0 plus a single `resize`, folding the bucket update away when
  minibatching is disabled.

| Benchmark | `time/op` | `cpu/op` | `allocs/op` |
| :--- | :--- | :--- | :--- |
| `SortAndGroup` `kMean/LargeVocab` | 935 -> 851 ms (-9.0%) | 895 -> 823 ms (-8.0%) | 110 -> 87 (-20.9%) |
| `SortAndGroup` `kSum/LargeVocab` | 913 -> 831 ms (-8.9%) | 849 -> 803 ms (-5.5%) | ~ |
| `SortAndGroup` `kMean/ModerateMultivalency` | ~ | 68.2 -> 64.2 ms (-5.8%) | 71 -> 67 (-5.6%) |
| `preprocess_input_benchmarks` (all 5) | -2.9% to -4.9% | ~ | ~ |

12 runs, p < 0.05 shown (`~` = no significant change); 0 regressions across 22
cases.
```
