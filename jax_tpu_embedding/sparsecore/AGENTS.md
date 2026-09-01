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
