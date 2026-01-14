Checkpointing
=============


Checkpoints are the serialized state of training. In order to save and restore
training the recommended library for checkpointing in JAX is
`Orbax <https://orbax.readthedocs.io/en/latest/#checkpointing>`__.

.. code:: python

    import orbax.checkpoint as ocp

    chkpt_mgr = ocp.CheckpointManager(
        directory='/path/to/checkpoint',
        options=ocp.CheckpointManagerOptions(
            max_to_keep=5,
            ...
        ),
    )
    ## Saving:
    chkpt_mgr.save(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeSave(train_state),
            embedding=ocp.args.PyTreeSave(embedding_variables)),
    )
    # Restore embedding from checkpoint
    restored = chkpt_mgr.restore(
        step,
        args=ocp.args.Composite(
        train_state=ocp.args.PyTreeRestore(init_train_state),
        embedding=ocp.args.PyTreeRestore()),
    )
    train_state = restored['train_state']
    emb_variables = {}
    for k, v in restored['embedding'].items():
        emb_variables[k] = embedding.EmbeddingVariables(
            table=v['table'], slot=v['slot'])

The embedding is usually saved as a separate ``item`` and it can be restored and
used in continued training. For a complete example see
`shakespeare_jit <https://github.com/jax-ml/jax-tpu-embedding/tree/main/jax_tpu_embedding/sparsecore/examples/shakespeare/jax_sc_shakespeare_jit.py>`__.
