# Using JAX SparseCore Securely

This document discusses the security model for JAX SparseCore (JAX SC). It
describes the security risks to consider when using models, checkpoints, input
data, or distributed training. We also provide guidelines on what constitutes a
vulnerability in JAX SC and how to report them.

## JAX models are programs

JAX models are python programs that are compiled (typically via `jax.jit`) into
XLA computation graphs. Since models are programs, executing untrusted models
is equivalent to running untrusted code.

If you need to run untrusted models, execute them inside a sandbox.

### Checkpoints

JAX SC models typically use Orbax (or other JAX checkpointing libraries) to save
and restore model state, including `EmbeddingVariables`.

When loading untrusted checkpoints, the values of the embedding tables are
untrusted. If your model or application code uses these values in interactions
with the filesystem, network, or other system resources, a maliciously crafted
checkpoint could alter the target of these operations.

### Model Configuration and Specs

Configuration parameters such as `max_ids_per_partition`,
`max_unique_ids_per_partition`, embedding table dimensions, device counts
(`local_device_count`, `global_device_count`), and hardware/SparseCore counts
(`num_sc_per_device`) are considered part of the model definition (code).
Since these parameters are defined in code, they are trusted. Resource
exhaustion (such as Out-of-Memory crashes) or arithmetic errors (such as
division by zero) caused by setting these configuration parameters to invalid
or excessively large values are not considered vulnerabilities. An attacker
with the ability to modify these parameters already has the ability to execute
arbitrary code in the training process.

### Minibatching Server (Multi-host training)

JAX SC supports multi-host training with minibatching. This feature starts a
process-local gRPC server (`MinibatchingNode`) on each host to synchronize
decisions across hosts (using `AllReduceGrpcService`).

In the open-source version of JAX SC, the default credentials used for the gRPC
server and client channels are insecure (unencrypted and unauthenticated).
This server only handles basic synchronization metadata and does not support
arbitrary code execution. Consequently, the security risk is limited to
eavesdropping on synchronization traffic and potential denial of service
(such as training deadlocks) if a malicious actor sends fake synchronization
requests.

## Untrusted inputs during training and prediction

JAX SC provides APIs to preprocess input data into the format required by the
SparseCore hardware (e.g., `preprocess_sparse_dense_matmul_input`).

This preprocessing is partially implemented in C++ for performance reasons.
Parsing maliciously crafted inputs could exploit potential vulnerabilities (such
as buffer overflows) in the C++ preprocessing code.

Ensure that input data from untrusted sources is validated and processed in a
sandboxed environment.

### Feedback Directed Optimization (FDO)

FDO in JAX SC can record and load stats from files (using `.npz` format).

If FDO is configured to load stats from an untrusted directory, a malicious
actor could place crafted `.npz` files to influence buffer size calculations.
While numpy's `allow_pickle` is disabled by default, loading malformed files
could still cause crashes or unexpected behavior. Ensure the FDO base directory
is secure and writable only by trusted processes.

### Mathematical Stability and Correctness

During training, mathematical operations (such as gradient calculations in
optimizers like Adagrad, F2A, or FTRL, or tensor processing in
`ProcessCooTensors`) may encounter numerical instability, such as division by
zero or square root of zero, if inputs or accumulators become zero.

This can result in `NaN` or `Infinity` values propagating through the model,
leading to data corruption (ruined training weights) or process termination.
While these are correctness bugs and numerical stability issues that should be
addressed with proper mathematical guardrails (such as epsilons), they do not
constitute security vulnerabilities as they do not allow privilege escalation,
arbitrary code execution, or unauthorized data access.

## Multi-Tenant environments

Running multiple JAX SC models in parallel on shared hardware (TPUs) introduces
multitenant risks.

### Tenant isolation

Isolation between different users or models sharing a system is the
responsibility of the infrastructure provider. JAX SC does not provide tenant
isolation mechanisms.

### Resource allocation

One model exhausting TPU memory or host resources can cause denial of service
for other models on the same system. JAX SC does not enforce resource limits
between models.

### Hardware attacks

Physical TPUs, like GPUs, can be the target of side-channel attacks. Report
physical hardware vulnerabilities to the hardware vendor (Google for TPUs).

## Reporting vulnerabilities

Please use [Google Bug Hunters reporting form](https://g.co/vulnz) to report
security vulnerabilities. Please include the following information:

*   A descriptive title.
*   A description of the technical details.
*   A minimal reproducing example.
*   An explanation of the impact and attack scenario.
