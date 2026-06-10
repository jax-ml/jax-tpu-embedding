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
   advanced/minibatching
   advanced/embedding_pipelining

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/shakespeare_flax
   tutorials/shakespeare_primitives

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Contributor Guide

   developer

JAX SparseCore provides support for leveraging the SparseCore accelerators present in
TPU generations starting with TPU v5. SparseCores are specialized processors designed
to accelerate workloads with sparse data access patterns, particularly large-scale
embedding lookups common in deep learning recommendation models and other areas.

Installation
------------

Stable Release
^^^^^^^^^^^^^^

You can install JAX TPU Embedding from PyPI:

.. code-block:: sh

   pip install jax-tpu-embedding

Nightly Builds
^^^^^^^^^^^^^^

To install the latest nightly builds from the public JAX registry:

.. code-block:: sh

   pip install --pre \
     --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
     jax-tpu-embedding

Building from Source
^^^^^^^^^^^^^^^^^^^^

For building from source and developer instructions, see the :doc:`developer` guide.
