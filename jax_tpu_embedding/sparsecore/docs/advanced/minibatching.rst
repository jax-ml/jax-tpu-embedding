Minibatching for SparseCore Embedding Lookups
============================================

Overview
--------

Minibatching is a feature in the SparseCore embedding library for handling input
batches that are too large to be processed by SparseCore in a single pass due to
on-chip memory constraints.

The SparseCore hardware has limits on the number of embedding IDs it can process in
one operation for a single table partition, constraints dictated by the size of its
on-chip memory (SPMEM):

*   ``max_ids_per_partition``: The maximum number of embedding IDs.
*   ``max_unique_ids_per_partition``: The maximum number of unique embedding IDs.

When an input batch requires more lookups for a given partition than these limits
allow, minibatching breaks the batch down into smaller pieces, called
**minibatches**. These minibatches are then processed sequentially on the
SparseCore, and their results are accumulated to produce the final output for
the original batch.

This mechanism offers several advantages:

*   **Prevents errors** caused by exceeding hardware ID limits.
*   Allows users to **train with larger logical batch sizes**, even if parts of the
    input data are highly skewed (i.e., some samples require many more
    embedding ID lookups than others).
*   **Does not affect model quality**, as it is designed to be mathematically
    equivalent to processing the full batch in a single pass.

Minibatching is particularly useful when the number of embedding IDs that need
to be looked up for a single SparseCore core or partition exceeds limits defined
by ``max_ids_per_partition`` or ``max_unique_ids_per_partition``, which are
parameters automatically tuned by :doc:`Feedback-Directed Optimization (FDO) <fdo>`.

Enabling Minibatching
--------------------

Minibatching can be enabled by setting the ``enable_minibatching`` flag to ``True``
in various APIs.

Flax Embedding Layer
^^^^^^^^^^^^^^^^^^^^

If you are using the :doc:`Flax Embedding Layer <../embedding>`, you can
enable minibatching by passing ``enable_minibatching=True`` layer initialization:

.. code-block:: python

    from jax_tpu_embedding.sparsecore.lib.flax.linen import embed as flax_embed
    from jax_tpu_embedding.sparsecore.lib.nn import embedding

    embed_layer = flax_embed.SparseCoreEmbed(
        feature_specs=features,
        enable_minibatching=True,
        mesh=mesh,
    )
    variables = embed_layer.init(jax.random.PRNGKey(0), ...)

    # For multi-host minibatching, create all_reduce_interface
    all_reduce_interface = embedding.get_all_reduce_interface(...)

    # Preprocess inputs
    preprocessed_inputs = embed_layer.preprocess_inputs(
        step, features, features_weights, all_reduce_interface)

    # During forward pass
    activations = embed_layer.apply(variables, preprocessed_inputs)

    # During backward pass
    updated_variables = embed_layer.apply_gradient(
        gradients,
        preprocessed_inputs,
    )

Low-level API
-------------

If using the :mod:`embedding` module directly, pass ``enable_minibatching=True`` to
:func:`preprocess_sparse_dense_matmul_input` and :func:`tpu_sparse_dense_matmul`:

.. code-block:: python

    from jax_tpu_embedding.sparsecore.lib.nn import embedding

    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        ...,
        enable_minibatching=True,
        all_reduce_interface=all_reduce_interface,
    )

    activations = embedding.tpu_sparse_dense_matmul(
        preprocessed_input,
        embedding_vars,
        ...,
        enable_minibatching=True,
    )

Note that for multi-host minibatching, you need to initialize and pass an
``all_reduce_interface`` object to :func:`preprocess_sparse_dense_matmul_input`. This
can be obtained via :func:`get_all_reduce_interface`.

Performance Considerations
--------------------------

Enabling minibatching introduces some overhead:

  * **Preprocessing**: The bucketization and cross-host synchronization steps
    add latency to input preprocessing.
  * **Execution**: Executing SparseCore lookups in a loop for multiple
    minibatches increases TPU execution time compared to a single lookup for the
    entire batch.
  * **Communication**: In multi-host settings, gRPC-based AllReduce operations
    introduce communication overhead.

Despite this overhead, minibatching is essential for stability when dealing with
large or skewed inputs that would otherwise exceed hardware limits for
``max_ids_per_partition`` or ``max_unique_ids_per_partition``. If your model runs
without exceeding these limits and does not report ID dropping with minibatching
disabled, you might achieve better performance by leaving it disabled. However,
if you experience ID dropping or errors due to these limits, enabling
minibatching is the recommended solution.

How it works
------------

When minibatching is enabled, the input preprocessing pipeline performs the
following steps:

1.  **Check for minibatching requirement**: For each table in the input batch,
    the preprocessing step checks if the number of embedding IDs or unique
    embedding IDs destined for any SparseCore partition exceeds the limits
    (``max_ids_per_partition`` and ``max_unique_ids_per_partition``). If these
    limits are exceeded for any table, that table is marked as requiring
    minibatching on the current host.

2.  **Cross-host synchronization**: In a multi-host environment, if at least one
    host requires minibatching for any table, all hosts must agree to use
    minibatching for the current step. This is achieved via a cross-host
    ``AllReduce`` operation implemented using gRPC, which aggregates the
    minibatching requirement status from all hosts. If any host requires
    minibatching, all hosts will proceed with it.

3.  **Bucketization**: If minibatching is required, all embedding IDs in tables
    that require minibatching are assigned to one of 64 buckets based on a hash
    of the embedding ID.

4.  **Minibatch creation**: The 64 buckets are grouped into minibatches. The
    goal is to create minibatches such that each minibatch fits within the
    memory constraints of the SparseCore. The division of buckets into
    minibatches is determined by another ``AllReduce`` operation across hosts,
    ensuring all hosts use the same minibatching strategy for the current step.
    This division is represented by a bitmask called ``MinibatchingSplit``.

5.  **Sequential processing**: During the embedding lookup (forward pass) and
    gradient update (backward pass), if minibatching is active
    (``num_minibatches > 1``), the SparseCore operation
    (``sparse_dense_matmul``) is executed in a loop, once for each minibatch. In
    the forward pass, embedding lookups are accumulated into the activation
    tensors based on the feature's combiner (e.g., 'sum'). In the backward pass,
    gradients are computed for each minibatch and applied sequentially to update
    the embedding tables in-place using the configured optimizer.

Cross-Host Synchronization Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned in steps 2 and 4 above, cross-host synchronization is performed
using a gRPC-based ``AllReduce`` operation. This is used first to
synchronize a boolean ``minibatching_required`` flag, and then to synchronize the
``MinibatchingSplit`` bitmask across all hosts. Both reductions use a logical OR
operation. The synchronization within ``AllReduce`` involves two main
phases:

1.  **Local Reduction**: Within each host, all participating threads (K) call
    ``InitializeOrUpdateState``, and their values are combined (OR-ed) into a
    single host-level value. The last thread to contribute its value is
    responsible for initiating the global reduction, as at this point, all local
    contributions are guaranteed to be aggregated into the host-level value.
2.  **Global Reduction**: The locally-reduced values from all hosts (N) are
    combined via an all-to-all gRPC exchange into a single globally-reduced
    value, which is then made available to all threads on all hosts.

The diagram below illustrates this flow, showing how ``SendLocalData`` executes in
parallel with the ``ContributeData`` RPC handler.

.. graphviz::

  digraph G {
    node [shape=box, style="rounded,filled", fillcolor=white, fontsize=10];
      edge [fontsize=9];
      compound=true;
      newrank=true;
      rankdir=TB;
      nodesep=0.3;
      ranksep=0.6;

      label="AllReduce Synchronization Flow for Minibatching";
      labelloc=t;
      fontsize=16;

      Host_i_T [label="Threads 0..k-1", shape=ellipse];
      Host_j_T [label="Threads 0..k-1", shape=ellipse];

      subgraph cluster_host_i {
          label = "Host i";
          bgcolor="#E6F0FA";
          i_Init [label="1. InitializeOrUpdateState\n(K threads reduce to local value L_i)"];
          i_Send [label="2a. SendLocalData(L_i)\n(Last contributing thread sends L_i to peers)"];
          i_Handler [label="2b. ContributeData(L_peer)\n(RPC handler reduces peer data into L_i)", fillcolor="#FEF9E7"];
          i_GetRes [label="3. GetResult\n(K threads wait for local and peer data, then read result)"];

          i_Init -> i_Send -> i_GetRes;
          i_Handler -> i_GetRes [style=dashed, label="contributes to result"];
      }

      subgraph cluster_host_j {
          label = "Host j";
          bgcolor="#F5E6FA";
          j_Init [label="1. InitializeOrUpdateState\n(K threads reduce to local value L_j)"];
          j_Send [label="2a. SendLocalData(L_j)\n(Last contributing thread sends L_j to peers)"];
          j_Handler [label="2b. ContributeData(L_peer)\n(RPC handler reduces peer data into L_j)", fillcolor="#FEF9E7"];
          j_GetRes [label="3. GetResult\n(K threads wait for local and peer data, then read result)"];

          j_Init -> j_Send -> j_GetRes;
          j_Handler -> j_GetRes [style=dashed, label="contributes to result"];
      }

      Host_i_T -> i_Init;
      i_GetRes -> Host_i_T;

      Host_j_T -> j_Init;
      j_GetRes -> Host_j_T;

      // RPCs
      i_Send -> j_Handler [label="gRPC", style=dashed, constraint=false];
      j_Send -> i_Handler [label="gRPC", style=dashed, constraint=false];

      // Ranks
      {rank=same; Host_i_T; Host_j_T;}
      {rank=same; i_Init; j_Init;}
      {rank=same; i_Send; j_Send; i_Handler; j_Handler;}
      {rank=same; i_GetRes; j_GetRes;}
  }

Explanation of Global Reduction Parallelism
"""""""""""""""""""""""""""""""""""""""""""

Stage 1 ``InitializeOrUpdateState`` performs local reduction among K threads, with
the last contributing thread (the "last contributing thread"), identified by
causing a ``local_contributions_counter`` to reach zero, emerging with the host's
locally-reduced value (e.g., L\ :sub:`i` for Host i).

Stage 2 is the global reduction, which involves parallel send and receive
operations:

* **2a. SendLocalData**: The last contributing thread on Host ``i`` calls
  ``SendLocalData``, which sends L\ :sub:`i` to Host ``j`` (and all other peers) via
  *asynchronous* gRPC calls. This function initiates the sends but does not
  wait for responses.
* **2b. ContributeData**: This is an RPC handler running on Host ``i``'s gRPC
  server. When Host ``j`` calls ``SendLocalData``, its RPC arrives at Host ``i`` and
  invokes ``ContributeData(L_j)``. This handler incorporates L\ :sub:`j` into Host
  ``i``'s state via an OR-reduction and decrements a counter tracking
  pending contributions from peers.
* **Parallelism**: Because ``SendLocalData`` sends RPCs asynchronously, and
  ``ContributeData`` is an RPC handler that reacts to incoming RPCs, these two
  operations occur concurrently. Host ``i`` can be sending L\ :sub:`i` to Host ``j`` at
  the same time as its ``ContributeData`` handler is processing L\ :sub:`k` received
  from Host ``k``.

**Synchronization and Result Retrieval**:

**Stage 3: Synchronization and Result Retrieval**:
Synchronization occurs when threads call ``GetResult``, which blocks until
both of the following conditions are met:

1. All K local threads have completed Stage 1 (``local_reduction_countdown``).
2. All N-1 peers have sent their data, which is processed by ``ContributeData``
   on the local gRPC server (``global_values_countdown``).

Once both local and global reduction are complete, ``GetResult`` unblocks and
provides the final result to all waiting threads.