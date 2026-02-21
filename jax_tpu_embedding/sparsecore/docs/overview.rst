Quick Overview for Users
========================

In this section, we provide a high-level overview of the steps required to utilize the
JAX SparseCore API for large embedding models. The specifics of these steps are detailed
in separate pages.

Embedding Specification
-----------------------

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding_spec

Configuring your embedding is done through two primary :class:`TableSpec`s and
:class:`FeatureSpec`s.

A :class:`TableSpec` specifies things like the embedding table size (vocabulary
and embedding sizes) and optimizer.

A :class:`FeatureSpec` specifies which table an embedding feature uses (multiple
features can use the same table) and the input/output shape of the feature
lookup.

Details of the embedding specification are described in the :doc:`Embedding Specification <embedding>`
page.

Input Preprocessing
-------------------

The SparseCore accepts sparse inputs (ragged/list of list) packed into a
`COO <https://openxla.org/xla/sparsecore#3_conversion_to_coo_tensors>`__ format.
To convert sparse inputs into this format we provide an API that performs the
conversion in a highly efficient way. See :doc:`Input Preprocessing <input_processing>`
section for more details.

Embedding Lookup and Gradient Update
------------------------------------

We provide two APIs for using large embeddings. These APIs have essentially identical
performance and which you choose depends on your modeling preference.

**Flax API**: Using a Flax layer is often the preferred choice as it's a more natural
fit for Flax based models. Here, the details of performing the embedding lookup and
gradient based weight update are implemented by the SparseCoreEmbed Flax layer. An example
of using this can be found in the [Shakespeare on Flax APIs] Colab example.

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding

**Primitive API**: Using this API, you make direct calls to perform the embedding lookup
:func:`tpu_sparse_dense_matmul` and gradient based weight update :func:`tpu_sparse_dense_matmul_grad`.
Typically this is done using JAX's shard_map feature inside of a :func:`jax.jit` function.
A working example of this can be seen in the Shakespeare on Primitive APIs Colab example.
The primitive API is mostly of interest to JAX SparseCore developers for implementing
higher level features like the Flax and embedding pipelining APIs.

Next Steps
----------

The above three steps account for the essentials of using the JAX SparseCore embedding
API. Once these are in place, additional features can be used to maximize performance
and productionize the integration with your model and infrastructure.

* :doc:`FDO <advanced/fdo>`: Dynamically adapt TPU buffer sizes to optimize memory usage.
* :doc:`Table and Feature Stacking <advanced/stacking>`: Make fewer, larger lookups to
  reduce memory usage and increase performance.
* Embedding Pipelining: Overlap SparseCore and TensorCore compute to maximize TPU
  performance.
* :doc:`Checkpointing <advanced/checkpointing>`: Save and restore from checkpoints for
  increased robustness.
