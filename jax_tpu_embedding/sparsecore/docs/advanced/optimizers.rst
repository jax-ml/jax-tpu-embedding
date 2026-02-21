.. currentmodule:: jax_tpu_embedding.sparsecore.lib.nn.embedding_spec

Optimizers
==========

JAX SparseCore supports several optimizers for training embedding tables. These optimizers
are specified in the :class:`TableSpec` for each table.

Introduction
------------

Optimizers are algorithms or methods used to change the attributes of the neural network
such as weights and learning rate in order to reduce the losses. JAX SparseCore supports
several common optimizers used for training large-scale recommendation models. You can specify
the optimizer for each embedding table via the ``optimizer`` field in the :class:`TableSpec`.

Choosing an Optimizer: ``OptimizerSpec`` vs. ``optax``
------------------------------------------------------

When training a model with JAX SparseCore, you will encounter two approaches to optimization:
the ``...OptimizerSpec`` classes provided by this library, and standard ``optax`` optimizers.

For training SparseCore embedding tables, we strongly encourage you to use the
``...OptimizerSpec`` classes detailed in this document (e.g., :class:`SGDOptimizerSpec`,
:class:`AdamOptimizerSpec`). These specs configure highly efficient optimizers where the
update logic is fused directly into the backward pass on the SparseCore hardware. This avoids
costly round-trips of gradients to the host and provides the best performance.

While ``optax`` is a powerful and flexible library for JAX, its role in a SparseCore model is
for any other non-embedding parameters your model might have. JAX SparseCore provides a helper
function, :func:`create_optimizer_for_sc_model`, which applies a given ``optax`` optimizer to your
model's other parameters, while ensuring the specialized ``...OptimizerSpec`` logic is used for
the embedding tables.

API
---

.. autosummary::
    OptimizerSpec
    AdagradMomentumOptimizerSpec
    AdagradOptimizerSpec
    AdamOptimizerSpec
    FTRLOptimizerSpec
    LaPropOptimizerSpec
    SGDOptimizerSpec

.. autoclass:: OptimizerSpec
   :members:

.. autoclass:: AdagradMomentumOptimizerSpec
   :members:

.. autoclass:: AdagradOptimizerSpec
   :members:

.. autoclass:: AdamOptimizerSpec
   :members:

.. autoclass:: FTRLOptimizerSpec
   :members:

.. autoclass:: LaPropOptimizerSpec
   :members:

.. autoclass:: SGDOptimizerSpec
   :members:
