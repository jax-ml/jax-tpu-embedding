Feature and Table Stacking
==========================

This document is only relevant to JAX SparseCore devs or advanced users.
The features described here are optional and not required to use the JAX SparseCore API.

Introduction
------------

Feature Stacking and Table Stacking are two similar but different features that
can be used to improve the efficiency of models with many tables and features.

* **Feature Stacking**: Multiple features that reference the same table can be
  combined into a single lookup by combining the samples. This results in
  fewer, larger lookups which is generally more efficient.
* **Table Stacking**: Multiple small tables can be combined into a single
  stacked table. This results in fewer, larger lookups as well and can also
  be more memory efficient for storing the sharded tables.

.. note::

    In what follows, the term "row" refers to a training sample/example and
    "column" refers to an embedding ID in a given sample. The column maps to a
    vocabulary in an embedding table.

Feature Stacking
----------------

Stacking multiple features requires stacking along the batch/sample dimension,
this is recorded in the `FeatureIdTransformation` structure using these fields:

* Row Offset: records the offset along the batch dimension.
* Col Shift: rotation of the vocabulary across the embedding table shards to
  distribute hot embedding IDs evenly.

.. note::

    The above explanation is for ``STACK_THEN_SPLIT`` strategy, but there's an
    additional interleaving of the sample dimension when using ``SPLIT_THEN_STACK``
    (the default). This interleaving helps distribute embedding IDs evenly across
    SparseCores during embedding lookup and update. This is because we split the
    stacked samples along the batch dimension.

Table Stacking (Optional)
-------------------------

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding_spec

Table stacking can help in decreasing training time by combining smaller
embedding tables to create larger ones there by reducing the number of embedding
table lookups and updates in forward and backward pass respectively. To do table
stacking, define the :class:`TableSpec` and :class:`FeatureSpec` as usual and then call
``auto_stack_tables`` which will update the feature specs and the
referenced tables specs with required stacking information. All the downstream
apis for training refer to the feature specs and account for stacking as
necessary. You do not need to do anything special with regard stacking in
preparing the inputs. For instance, define ``TableSpecs`` for the embedding tables.

.. code:: python

    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_b',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )


Define the ``FeatureSpecs`` that would use these tables:

.. code:: python

    feature_specs = [
      embedding_spec.FeatureSpec(
          table_spec=table_spec_a,
          input_shape=(16, 1),
          output_shape=(
              16,
              table_spec_a.embedding_dim,
          ),
          name='feature_spec_a',
      ),
      embedding_spec.FeatureSpec(
          table_spec=table_spec_b,
          input_shape=(16, 1),
          output_shape=(
              16,
              table_spec_b.embedding_dim,
          ),
          name='feature_spec_b',
      ),
      embedding_spec.FeatureSpec(
          table_spec=table_spec_b,
          input_shape=(16, 1),
          output_shape=(
              16,
              table_spec_b.embedding_dim,
          ),
          name='feature_spec_c',
      ),
    ]

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.table_stacking

If you want to use table stacking call :func:`auto_stack_tables` as follows:

.. code:: python

    from jax_tpu_embedding.sparsecore.lib import embedding

    # Optional, only needed if you want to stack tables.
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=4, # 4 for TPU v5, 2 for TPU v6e
    )
    # Required, this will populate feature stacking related info when more than
    # one feature use same table. It will also do some basic validations on the
    # feature specs.
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=4, # 4 for TPU v5, 2 for TPU v6e
    )

There is also an API to manually stack tables: :func:`stack_tables`.

API
---

.. autofunction:: auto_stack_tables

.. autofunction:: stack_tables

References
----------

* https://openxla.org/xla/sparsecore#understanding_table_stacking
