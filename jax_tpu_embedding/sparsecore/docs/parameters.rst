Parameters for SparseCore Input
===============================

Introduction
------------

.. note::

    "Host" refers to CPU and "Device" refers to TPU (SparseCore + TensorCore)
    in the following discussion.

For training `sparse inputs <https://en.wikipedia.org/wiki/Sparse_matrix>`__ on
SparseCore with varying formats such as
`CSR <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`__,
`COO <https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)>`__, we
need to pack it into a `predefined format <https://openxla.org/xla/sparsecore#3_conversion_to_coo_tensors>`__
to be ingestible by XLA. Due to limitations on JAX not supporting variable size inputs,
we need to pad the sparse input (with varying embedding IDs) into a fixed sized buffer.

The following sparse dense matmul (or grad) operation requires an all-to-all
communication due to:

1. Sharding of the input data across devices.
2. Sharding of the embedding table across SparseCores.

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.core.input_preprocessing

The input preprocessing provided by the function :func:`preprocess_sparse_dense_matmul_input`
groups embedding IDs from the input and partitions into queries between pairs of
SparseCores. For a given SparseCore A, a **partition** refers to collection of embedding
IDs that would be queried to another SparseCore B.

Due to the input data distribution and device/host topology, this would result
in the output buffer for each SparseCore containing only some non-empty
partitions with varying sizes. This prevents us from statically determine the
size of the input buffer (without padding). The varying input data distribution
can affect two things:

1. Partition sizes: How many embedding IDs belong to a partition from
   SparseCore A to SparseCore B?
2. Partition counts: How many non-empty partitions can be formed from the input
   data at SparseCore A?

Max (unique) IDs per partition
------------------------------

The format requires separating the embedding IDs into buckets or partitions due
to (2) above. Since we only have static sized buffers, we need to bound these
using ``max_ids_per_partition`` and ``max_unique_ids_per_partition`` (also together
referred to as ``limits``). The former is required because we also need to map
back the combined IDs to the corresponding sample. Using a very large value
would waste memory (or even lead to an OOM) whereas using a very small value
would lead to dropping of IDs [#f1]_ that can in turn affect the model quality.

Suggested COO buffer size
-------------------------

After we pack all the partitions (with HBM granularity/alignment), we may end up
with variable partition counts and sizes that further require alignment - the
final size per SparseCore is ``suggested_coo_buffer_size_per_sc`` (and
``suggested_coo_buffer_size_per_device = suggested_coo_buffer_size_per_sc * num_sc_per_device``).

.. note::

    **Buffer Sizing Invariant:** Because the SparseCore buffer slice on each
    core must accommodate all incoming partitions combined, the per-core buffer
    size must always be at least as large as the maximum partition limit:

    .. code-block:: text

        max_ids_per_partition <= suggested_coo_buffer_size_per_sc

Choosing a value for the parameters
-----------------------------------

The appropriate values for these parameters depend on model architecture
(valency, table stacking), hardware topology (chips, SparseCores per chip), and
input data distribution.

Batch sizes specified in ``FeatureSpec`` shapes are typically global batch sizes
across all devices. Buffer sizing parameters depend on local per-device or
per-SparseCore quantities:

* **Device batch size**: ``device_batch_size = global_batch_size // global_device_count``
* **SparseCore batch size**: ``sc_batch_size = device_batch_size // num_sc_per_device``
* **Global SparseCore count**: ``total_sparse_cores = global_device_count * num_sc_per_device``
* **Effective valency** (``valency``): Total average embedding lookups per sample across all stacked features.

Recommended Rules of Thumb
^^^^^^^^^^^^^^^^^^^^^^^^^^

If optimal dataset statistics are not yet known, use the following closed-form
formulas [#f2]_ (where ``sc_total_lookups = sc_batch_size * valency``):

1. **Max IDs per Partition** (accounts for ~20% hot-shard concentration under Zipfian skew):

   * ``hot_shard_lookups = 0.20 * sc_total_lookups``
   * ``background_avg_lookups = 0.80 * sc_total_lookups / total_sparse_cores``
   * ``tail_fluctuation = 2.0 * sqrt(sc_total_lookups / total_sparse_cores)``
   * ``expected_peak_lookups = hot_shard_lookups + background_avg_lookups + tail_fluctuation``
   * ``peak_partition_ids = headroom_factor * expected_peak_lookups`` (where ``headroom_factor = 1.25``)
   * ``max_ids_per_partition = round_up_to_8(peak_partition_ids)``
   * ``max_unique_ids_per_partition = round_up_to_8(0.5 * peak_partition_ids)``

2. **Suggested COO Buffer Size per Device** (must be at least ``num_sc_per_device * max_ids_per_partition``):

   * ``min_device_buffer = num_sc_per_device * max_ids_per_partition``
   * ``estimated_device_lookups = 1.2 * device_batch_size * valency + 14 * total_sparse_cores``
   * ``suggested_coo_buffer_size_per_device = round_up_to_alignment(max(min_device_buffer, estimated_device_lookups), 8 * num_sc_per_device)``

.. tip::

    You can compute these parameters programmatically using ``jax_tpu_embedding.sparsecore.utils.estimate_preprocessing_parameters(...)``.

Common Pitfalls & Troubleshooting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Compiler Invariant Assertion Failure**:

   .. code-block:: text

       max_ids_per_partition <= coo_buffer_size_per_sc

   *Cause:* ``max_ids_per_partition > suggested_coo_buffer_size_per_sc`` (e.g., testing small batch sizes while passing production partition limits).
   *Fix:* Ensure ``suggested_coo_buffer_size_per_sc >= round_up_to(max_ids_per_partition, 8)`` (or ``suggested_coo_buffer_size_per_device >= num_sc_per_device * round_up_to(max_ids_per_partition, 8)``).

2. **Observed ID Dropping Warning/Error**:

   .. code-block:: text

       Observed max ids per partition: 320 for table: user_table is greater than the set max ids per partition: 256...

   *Cause:* An input partition (or minibatching bucket) exceeded ``max_ids_per_partition``.
   *Fix:* Increase ``max_ids_per_partition``, enable :doc:`minibatching <advanced/minibatching>`, or set ``allow_id_dropping = True`` during initial warm-up to collect FDO statistics.

3. **Compiler Out-of-Memory (OOM)**:

   .. code-block:: text

       No viable logical replica count for...

   *Cause:* ``max_ids_per_partition`` is too large for available TileSpmem.
   *Fix:* Decrease batch size, reduce overly conservative limits, or use :doc:`minibatching <advanced/minibatching>`.

Tuning via FDO (Feedback-Directed Optimization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended production workflow is to use :doc:`FDO <advanced/fdo>`:

.. code:: python

    _, stats = embedding.preprocess_sparse_dense_matmul_input(...)
    embedding.update_preprocessing_parameters(feature_specs, stats, num_sc_per_device)

Terminology
-----------

* ``sample`` / ``example``: A training example or a sample from an input batch.
* ``partition``: Each SparseCore corresponds to a partition of the input batch
  data (subset of embedding IDs) originating from all other SparseCores.
* ``max_ids_per_partition``: Maximum number of embedding IDs that a SparseCore
  receives for its share of sharded embedding table. This depends on input
  data and topology.
* ``max_unique_ids_per_partition``: Maximum number of unique embedding IDs that
  a SparseCore receives for its share of the sharded embedding table. This is
  because an input batch may not have all the IDs from the vocabulary. This
  depends on input data and topology.
* ``suggested_coo_buffer_size_per_device``: The final size of the COO buffer per
  device (multiple SparseCores). This is the size of the HBM buffer that will
  be allocated on all SparseCores. This depends on number and size of
  partition each SparseCore ends up with.

.. [#f1] If ``allow_id_dropping=True``, otherwise would throw an error.
.. [#f2] Assumes a Zipfian / power-law distribution typical of real-world
   recommendation systems (e.g., DLRM, content recommenders, search ranking)
   and sequence models, where popular items and default fallback tokens
   (e.g. ``0`` or ``UNK``) concentrate ~20% of batch traffic onto the hottest
   destination SparseCore shard. Under a strictly uniform distribution,
   ``max_ids_per_partition`` can be scaled closer to
   ``sc_batch_size * valency / total_sparse_cores``, but uniform sizing severely
   under-provisions real-world workloads leading to ID dropping.
