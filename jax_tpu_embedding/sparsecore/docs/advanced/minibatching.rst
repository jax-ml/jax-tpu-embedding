Minibatching for SparseCore Embedding Lookups
=============================================

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
---------------------

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
^^^^^^^^^^^^^

If using the ``embedding`` module directly, pass ``enable_minibatching=True`` to
``preprocess_sparse_dense_matmul_input`` and ``tpu_sparse_dense_matmul``:

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
``all_reduce_interface`` object to ``preprocess_sparse_dense_matmul_input``. This
can be obtained via ``embedding.get_all_reduce_interface(...)``.

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

When minibatching is enabled, the preprocessing and execution pipeline coordinates
across all hosts to split the batch if the statically allocated buffer limits are
exceeded.

Here is the high-level algorithm for minibatching and synchronization:

.. code-block:: python

    # --- Host Side (Preprocessing) ---
    def preprocess_minibatched_inputs(features, weights, max_sc_limits):
      # 1. Determine if Minibatching is needed (Sync across hosts)
      local_needs_mb = check_if_limits_exceeded(features, max_sc_limits)
      global_needs_mb = cross_host_all_reduce_or(local_needs_mb)

      if not global_needs_mb:
        return preprocess_normal(features, weights)

      # 2. Bucketize and Estimate Load
      buckets = bucketize_by_hash(features, weights, num_buckets=64)
      local_bucket_loads = estimate_memory_per_bucket(buckets)

      # 3. Sync load and Decide splits (Sync across hosts)
      global_bucket_loads = cross_host_all_reduce_sum(local_bucket_loads)
      minibatch_splits = partition_buckets(global_bucket_loads, max_sc_memory_limit)

      # 4. Pack into minibatched buffers
      return pack_minibatches(buckets, minibatch_splits)


    # --- Device Side (TPU Lookup) ---
    # This runs inside JAX JIT/shard_map
    def execute_lookup(preprocessed_inputs, embedding_tables):
      activations = zeros(batch_size, embedding_dim)

      # The compiler/hardware loops over minibatches sequentially
      for mb_idx in range(preprocessed_inputs.num_minibatches):
        # 1. Slice CSR pointers for the current minibatch
        sliced_row_pointers = slice_csr(preprocessed_inputs.row_pointers, mb_idx)

        # 2. Perform local lookup and accumulate
        activations += local_sparse_dense_matmul(
            sliced_row_pointers,
            preprocessed_inputs.embedding_ids,
            preprocessed_inputs.sample_ids,
            preprocessed_inputs.gains,
            embedding_tables
        )

      return activations

Cross-Host Synchronization Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cross-host synchronization (used for ``cross_host_all_reduce_or`` and
``cross_host_all_reduce_sum`` in the pseudocode above) is implemented using a
gRPC-based ``AllReduce`` operation. This ensures all hosts remain synchronized
and use identical minibatching splits.

The synchronization involves two main phases:

1.  **Local Reduction**: Within each host, all local threads combine their
    data into a single host-level value.
2.  **Global Reduction**: The host-level values are exchanged and combined
    across all hosts via gRPC.

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

Here is the simplified pseudocode for this coordination:

.. code-block:: python

    # Shared state on each host, initialized for the step
    global_value = initial_value

    def all_reduce(my_thread_data, reduction_op):
      # 1. Local Reduction: Combine data from all local threads on this host
      host_value = contribute_local(my_thread_data, reduction_op)

      # The last local thread to contribute triggers the global exchange
      if is_last_local_thread():
        for peer in peer_hosts:
          # Send this host's data to peers asynchronously
          send_data_async(peer, host_value)

      # 2. Synchronization: All threads block until local reduction is done
      # and the RPC handler (below) has received and combined all peer data.
      wait_for_all_local_and_peer_contributions()

      return global_value

    # RPC Handler: Executed on the local server when peer data arrives
    def on_peer_data_received(peer_data, reduction_op):
      global global_value
      # Combine peer data into the global result
      global_value = reduction_op(global_value, peer_data)

The synchronization blocks execution until all local threads have completed local
reduction and all peer hosts have contributed their data, ensuring a consistent
global state before proceeding.
