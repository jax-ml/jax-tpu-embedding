Building from source
====================

This guide explains how to build JAX TPU Embedding from source, run tests, and manage dependencies.

To build the project, you need to have `Bazel <https://bazel.build/>`_ (or `Bazelisk <https://github.com/bazelbuild/bazelisk>`_) installed in your environment.

Cloning the repository
----------------------

First, clone the repository from GitHub:

.. code-block:: bash

   git clone https://github.com/jax-ml/jax-tpu-embedding.git
   cd jax-tpu-embedding

Building the wheel
------------------

You can build the Python wheels using the provided helper scripts.

Option A: Local Build
^^^^^^^^^^^^^^^^^^^^^

To build the wheel locally using your system's Bazel installation:

.. code-block:: bash

   # Build wheel for Python 3.12 (default)
   ./tools/local_build_wheel.sh

   # Build wheel for a specific Python version
   HERMETIC_PYTHON_VERSION=3.13 ./tools/local_build_wheel.sh

This will generate the wheel files in the ``dist/`` directory.

Option B: Docker Build (Recommended for releases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build reproducible wheels targeting Linux in a clean Docker container:

.. code-block:: bash

   ./tools/docker_build_wheel.sh

Installing the wheel
--------------------

Once the wheel is built, you can install it using ``pip``:

.. code-block:: bash

   pip install dist/jax_tpu_embedding-*.whl

Running Tests
-------------

You can run the unit tests using Bazel:

.. code-block:: bash

   # Run all tests
   bazel test //jax_tpu_embedding/sparsecore/...

   # Run a specific test
   bazel test //jax_tpu_embedding/sparsecore/lib/nn/tests:embedding_spec_test

.. note::
   Some tests might require access to TPU hardware and will be skipped or fail if run in a CPU-only environment.

Managing Python Dependencies
----------------------------

JAX TPU Embedding uses hermetic Python for Bazel builds and tests to ensure reproducibility. Dependencies are pinned in lock files under the ``third_party/py/`` directory (e.g., ``requirements_lock_3_12.txt``).

To update these dependencies:

1.  Modify the direct dependencies list in ``third_party/py/requirements.in``.
2.  Run the following command to update the lock file for your target Python version (e.g., 3.12):

    .. code-block:: bash

       bazel run //third_party/py:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
