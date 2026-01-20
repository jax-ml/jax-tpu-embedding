JAX TPU Embedding documentation
===============================

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Essentials

   overview
   embedding
   input_processing
   parameters

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Advanced Topics

   advanced/checkpointing
   advanced/fdo
   advanced/optimizers
   advanced/stacking

JAX SparseCore provides support for leveraging the SparseCore accelerators present in
TPU generations starting with TPU v5. SparseCores are specialized processors designed
to accelerate workloads with sparse data access patterns, particularly large-scale
embedding lookups common in deep learning recommendation models and other areas.

Installation
------------

.. code:: sh

   pip install jax-tpu-embedding
