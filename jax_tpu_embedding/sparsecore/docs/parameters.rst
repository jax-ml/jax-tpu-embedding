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
final size per SparseCore is ``suggested_coo_buffer_size_per_sc``.

Choosing a value for the parameters
-----------------------------------

The appropriate values for these parameters depend on the model size and input
training data distribution. However, there are some guidelines to estimate and
tune these values.

Batch sizes specified in ``FeatureSpec`` input and output shapes are typically
global batch sizes (i.e., across all devices). However, buffer size parameters
like ``max_ids_per_partition`` are estimated based on data distribution on
each SparseCore, which depends on the batch size per device or per SparseCore.
When using heuristics like the ones below, ensure that ``batch_size`` refers to
the batch size processed by a single SparseCore.

Firstly, if not much is known, start with the following:

.. code:: python

    max_ids_per_partition = 0.4 * global_batch_size
    max_unique_ids_per_partition = 0.1 * global_batch_size
    suggested_coo_buffer_size_per_sc = 0.4 * global_batch_size

If these are too low, then ids will be dropped during input preprocessing step
of training, leading to an error like the following:

    Observed max ids per partition: 320 for table: user_table is greater than the
    set max ids per partition: 256...

Next, set ``allow_id_dropping = true`` in
``embedding.preprocess_sparse_dense_matmul_input(...)``. This will get past the
above error and continue training with dropping any extra ids. While this will
degrade the model quality, it will allows the trainer to analyze more input
batches leading to better estimates of the table limits.

To avoid dropping ids, now increase the ``max_ids_per_partition`` etc. by using
the reported extra ids count in error message above. Note that when
``allow_id_dropping`` is true, the above error message is logged as a warning so
you can still see the observed limits in logs.

The main function that you will use for preprocessing the input would be
``preprocess_sparse_dense_matmul_input`` in ``embedding.py``. It returns the
preprocessed inputs as well as the input statistics (for all the above
parameters). These can also be used to directly update the feature specs as
follows. Note, this direct approach to updating the ``feature_specs`` should not
be used in a multi-host setup as different processes will observe different
stats leading to different buffer sizes. The correct way to update these stats
is to use the same values across all processes. You can learn more about this
using :doc:`FDO <advanced/fdo>`.

.. code:: python

    _, stats = embedding.preprocess_sparse_dense_matmul_input(...)
    embedding.update_preprocessing_parameters(feature_specs, stats)

Another common scenario is when the max ids per partition is very high leading
to the following compiler error which means it was unable to allocate the
requested buffer sizes.

    No viable logical replica count for...

This is an indicator that the max ids per partition setting is too high for that
batch size and topology. It is recommended to decrease the batch size,
``max_ids_per_partition`` or both to get to compiling stage again.

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
