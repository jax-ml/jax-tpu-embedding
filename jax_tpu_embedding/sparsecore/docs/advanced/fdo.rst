Feedback Directed Optimization
==============================

FDO can be used to adjust the static input buffer parameters (e.g., ``limits``)
as training progresses. It provides a framework to automate the process of
estimating and updating these parameters for next training epoch.

Architecture
------------

.. currentmodule:: jax_tpu_embedding.sparsecore.lib.fdo.fdo_client

Each process records the observed input parameters in a local FDO client during
the course of training and publishes the recorded parameters from local memory
to a persistent storage on a periodic basis. Currently we have an implementation
for file storage but users are encouraged to create a custom :class:`FDOClient`
implementation to better integrate with their training infrastructure.

At some frequency, usually end of epoch, all processes synchronize and load the
limits from the storage and update the :class:`FeatureSpec`s to use the new
limits. When the limits are updated in the :class:`FeatureSpec`s the jitted
train step needs to be recompiled. This can happen automatically if the
:class:`FeatureSpec`s object is a static argument to the train function. See
this `JAX document
<https://docs.jax.dev/en/latest/jit-compilation.html#marking-arguments-as-static>`__
that explains how the :func:`jax.jit` triggers recompilation when a static
argument changes.

All the above steps are customizable and a typical flow looks as follows:

.. code:: python

    # Import the fdo client
    from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client

    # Train function: Make sure it takes feature specs as a static argument so that
    # when its changes, jit triggers a recompilation.
    def train_step(
        feature_spec: Nested[FeatureSpec]
        ...
    ):
        ...

    jit_train_step = jax.jit(train_step, static_argnums=0, ...)

    # Create an instance of fdo client
    fdo_dir = "/tmp/fdo_dumps/"
    fdo_client = file_fdo_client.NPZFileFDOClient(fdo_dir)

    for step in range(100):
        # Record stats returned from preprocessing step
        preprocessed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
            ...
        )
        # Record the stats returned by inputs preprocessing
        fdo_client.record(stats)
        jit_train_step(...)
        # At some frequency, publish and update stats.
        if step % 10 == 0:
            fdo_client.publish()
            # Add a barrier here so that all processes can finish publishing
            # their stats and so all the processes can read the same data.
            jax.experimental.multihost_utils.sync_global_devices("FDO_publish_barrier")
            # Load FDO stats dumps and update feature specs.
            loaded_stats: embedding.SparseDenseMatmulInputStats = fdo_client.load()
            # Any custom code to adjust the stats can go here. `transform` is a
            # user-defined function to modify the FDO statistics. For example:
            def transform(stats):
                # Alter stats as desired:
                # stats.required_buffer_size_per_sc *= 2
                return stats
            updated_stats = transform(loaded_stats)
            # Update the feature specs.
            embedding.update_preprocessing_parameters(
                feature_specs, updated_stats, num_sc_per_device
            )

API
---

.. autoclass:: FDOClient
   :members:
