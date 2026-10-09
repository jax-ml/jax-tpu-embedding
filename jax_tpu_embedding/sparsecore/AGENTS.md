# JAX SparseCore AI Agent Directives

These guidelines enforce structural invariants for AI assistants operating
within the JAX SparseCore (`third_party/py/jax_tpu_embedding/sparsecore/...`)
codebase.

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

**Resolution**: Suppress type errors at the *injection site* using `# pyrefly:
ignore[bad-argument-type]` with a brief comment. Do not alter the class
definition.

[26572]: https://github.com/jax-ml/jax/issues/26572#issuecomment-2665778697
[14688]: https://github.com/jax-ml/jax/pull/14688

## Input Preprocessing Performance Evidence

**Rule**: Treat input preprocessing in `lib/core/` as performance-critical.
Every change to its hot path needs a before/after run of
`lib/core/benchmarks:input_preprocessing_benchmark`, reported as `PERF=<results
link>` or `PERF=not required (<reason>)` in the change description. Follow the
presubmit failure message for the exact command.

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

**Scope**: The hot path is everything that runs per ID or per batch: COO
extraction, sort and group, minibatching splits and merges, and CSR buffer
filling (`coo_format.h`, `input_preprocessing*.{h,cc}`,
`partitioned_coo_tensors.h`, `sort_and_group_coo_tensors_impl.h` and the
`*_impl.h` input streams). `PERF=not required (<reason>)` is appropriate only
when the change cannot alter generated code on that path, for example comments,
renames, or code that runs once per table such as validation.

**Writing hot-path code**:

*   Keep the default path unchanged when adding a new mode. Select the mode at
    compile time (`if constexpr` or a template parameter) instead of adding
    runtime fields that every ID pays for. Comparing the optimized benchmark
    binary or the function's disassembly against the base revision confirms that
    the default path compiles to the same code.
*   Avoid integer division and modulo by runtime values in per-ID loops. The
    total SC count is validated to be a power of two, so use masks and shifts.
*   Avoid per-batch heap allocations and copies of COO data; reuse buffers.
*   Compute values that are fixed for a table (flags, strides, sizes) once, not
    per batch or per device.

**Benchmark coverage**: If a change adds a new hot path (a new mode, flag, or
layout), add a matching case to the benchmark in the same change, so the new
path has a baseline and future changes to it are measured.

**Resolution**: Run the benchmark on the final version of the change. Rank
results by wall time, then CPU time, then allocations. Cite only deltas with
p<0.05, never the geomean, and explain any tradeoff between metrics. Ignore peak
memory per op, whose run-to-run noise is too large to be useful; use allocations
per op for memory instead.
