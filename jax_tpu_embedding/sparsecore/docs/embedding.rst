Embedding Specification
=======================

See Google's page about `Embedding <https://developers.google.com/machine-learning/crash-course/embeddings>`__
for a definition and examples.

Terminology
-----------

* **(Embedding) Table**: The lower-dimensional representations of sparse/categorical data.
  For each token in the vocabulary we have a vector with a size of the embedding dimension.
* **Embedding ID (Token)**: Represents an element of the embedding vocabulary.
* **Vocabulary Size**: The total number of unique embedding IDs. This is the number
  of rows in the embedding table.
* **Embedding Dimension**: The size of the lower dimensional space for the embeddings.
  This is the number of columns in the embedding table.
* **Sample (Example)**: Represents a single training example with multiple tokens.
* **Feature (Input)**: Represents a collection of samples.
* **Max Sequence Length**: Defines the maximum number of tokens that a sample can have
  in a given feature.
* **Weight/Gain**: The weight of each Embedding ID in a given sample.
* **Combiner**: The aggregation function for combining the embeddings for a given sample.
  For instance, sum or mean.
* **(Feature) Activations**: The weighted aggregation calculated with the Combiner for
  each sample in a given Input Feature.
* **(Feature) Gradients**: Gradients (of the feature activations) with respect to
  the loss function.
* **(Embedding Table) Optimizer**: The update function for the Model parameters
  and Embedding Table.

API
---

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding_spec

.. autoclass:: TableSpec
   :members:

.. autoclass:: FeatureSpec
   :members:

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.flax.linen.embed

.. autoclass:: SparseCoreEmbed
   :members:

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.flax.nnx.embed

.. autoclass:: SparseCoreEmbed
   :members:

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding

.. autofunction:: tpu_sparse_dense_matmul

.. autofunction:: tpu_sparse_dense_matmul_grad

.. autofunction:: preprocess_sparse_dense_matmul_input

.. autofunction:: preprocess_sparse_dense_matmul_input_from_sparse_tensor

Multivalent (Unordered/Pooled) Features
---------------------------------------

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding_spec

For multivalent features, each sample is represented by an unordered set of embedding IDs.
The embeddings corresponding to these IDs are aggregated or "pooled" into a single embedding
vector for the sample. This is done using the combiner (e.g., sum, mean) specified in the :class:`TableSpec`.

For example, if a sample has IDs ``[10, 21, 32]`` and the combiner is mean, the output activation
will be ``mean(embedding(10), embedding(21), embedding(32))``.

The input shape for a batch of such features is ``[batch_size, max_ids_per_sample]``, where
``max_ids_per_sample`` is the valency. The output shape is ``[batch_size, embedding_dim]``.

Sequence (Ordered/Concatenated) Features
----------------------------------------

For sequence features, each sample is an ordered sequence of items, where each
item can be one or more embedding IDs. The embeddings for each item in the
sequence are computed and then concatenated to form the final output.

To handle sequence features, you will need to flatten the sequence dimension
into the batch dimension before passing the features to the embedding layer. For
an input of shape ``[batch_size, sequence_length, valency]``, you should reshape
it to ``[batch_size * sequence_length, valency]``. The embedding lookups and
combinations (if ``valency > 1``) are performed on this flattened tensor,
resulting in an output of shape ``[batch_size * sequence_length, embedding_dim]``.
You can then reshape this output back to ``[batch_size, sequence_length,
embedding_dim]``, which is equivalent to concatenating the embeddings for each
item in the sequence.

If you have variable sequence lengths, you will need to pad your inputs to a
``max_sequence_length``.

Optimizers
----------

See the :doc:`Optimizers <advanced/optimizers>` page for more details on the available
optimizers and how to configure them.

Flax Embedding Layer
--------------------

`Flax <https://flax-linen.readthedocs.io/en/latest/>`__ is the most commonly used JAX
neural network library. The JAX SparseCore API provides a Flax layer that uses the primitive
APIs to support large embeddings.

Flax comes in two flavors:

`Linen <https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen>`__
(now deprecated) and the more recent `NNX <https://flax.readthedocs.io/en/stable/>`__.
The Flax project provides a guide for `migrating from Linen to NNX <https://flax.readthedocs.io/en/stable/migrating/linen_to_nnx.html>`__.
SparseCore project provides both Linen and NNX layers for large embedding models
that can be used without the need for modification or extension. These layers
are built on the primitive API, use the same :doc:`Embedding Specification <embedding>`
objects to configure the embedding and accept inputs from the :doc:`Input Preprocessing <input_processing>` API.

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.flax

You can find the Linen module here: :class:`linen.embed.SparseCoreEmbed`.
The newer NNX module is here: :class:`nnx.embed.SparseCoreEmbed`.

Caveats
^^^^^^^

**Caveat 1:** As with the primitive API and due to the size of embedding tables,
the embedding tables are updated in-place during the gradient calculation. As
such, gradients of the embeddings can't be extracted in the same way as they are
with dense layers.
