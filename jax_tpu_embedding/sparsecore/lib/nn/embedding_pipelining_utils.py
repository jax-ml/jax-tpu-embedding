# Copyright 2024 The JAX SC Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Jax SparseCore Embedding Pipelining API.

internal link:jax-sc-embedding-pipelining

Passing n batches (0 based indexing)
NOTE: output from `SC bwd (i-2)` is used in `SC fwd (i)`, i.e. `[SC BWD (i-2) ->
  SC FWD i] (SC)` and `[TC FWD BWD (i-1)] (TC)` is run in parallel

| Cycles | SC fwd | TC fwd/bwd | SC bwd | SC bwd -> SC fwd |
| -------|--------|------------|--------|------------------|
| 0      | 0      | -          | -      | SC fwd(0)        |
| 1      | 1      | 0          | -      | SC fwd(1)        |
| 2      | 2      | 1          | 0      | 0 -> 2           |
| 3      | 3      | 2          | 1      | 1 -> 3           |
| ...    | ...    | ...        | ...    | ... -> ...       |
| n-1    | n-1    | n-2        | n-3    | n-3 -> n-1       |
| n   (E)| -      | n-1        | n-2    | SC bwd(n-2)      |
| n+1 (E)| -      | -          | n-1    | SC bwd(n-1)      |

E = Extra Steps

Usage:

```
import embedding_pipelining_utils as ep_utils
import flax
import jax
import jax.numpy as jnp
import functools

@flax.struct.dataclass
class TrainState:
  ...


@flax.struct.dataclass
class PipelineData:
  input0: jax.Array | None = None
  input1: jax.Array | None = None
  input2: jax.Array | None = None
  emb_act: jax.Array | None = None
  emb_grad: jax.Array | None = None
  ...


PipelineState = ep_utils.PipelineState[PipelineData, TrainState]

def sc_fwd(args: PipelineData, emb_vars):
  ...
  return args.replace(emb_act=emb_act, ...)

def tc_fwd_bwd(args: PipelineData, train_state):
  ...
  return args.replace(emb_grad=emb_grad, ...), train_state

def sc_bwd(args: PipelineData, emb_vars):
  ...
  return args.replace(emb_grad=..., ...), emb_vars

def user_train_step(
    pipeline_state: PipelineState,
    pipeline_input: PipelineData,
    pipeline_step: jax.Array,
    num_steps: jax.Array,
    ...
  ) -> tuple[PipelineState, PipelineData, PipelineData, PipelineData]:
  ...
  (
      pipeline_state,
      step_i_sc_fwd_out,
      step_im1_tc_out,
      step_im2_sc_bwd_out,
  ) = ep_utils.step(
      pipeline_state=pipeline_state,
      pipeline_input=pipeline_input,
      sc_fwd_function=sc_fwd,
      tc_function=tc_fwd_bwd,
      sc_bwd_function=sc_bwd,
      pipeline_step=pipeline_step,
      num_steps=num_steps,
  )
  ...
  return (
      pipeline_state,
      step_i_sc_fwd_out,
      step_im1_tc_out,
      step_im2_sc_bwd_out,
      ...,
  )


def run_model():
  ...
  # NOTE: donate arguments according to your use case (if you don't need them
  #   anymore) to avoid unnecessary copies
  jitted_train_step = jax.jit(
      user_train_step,
      donated_argnums=(0, 1, 2, 3),
  )
  ...
  num_steps = ep_utils.get_pipeline_train_steps(num_steps)
  ...
  last_input = None
  ...
  for step in range(start_step,num_steps):
    ...
    pipeline_input = (
        PipelineData(...batches[step])
        if step < num_steps
        else ep_utils.get_dummy_input(last_input)
    )
    last_input = pipeline_input
    ...
    if step == start_step:
      pipeline_state = ep_utils.get_initial_state(
          train_state=train_state,
          embedding_variables=embedding_variables,
          pipeline_input=pipeline_input,
          tc_function=tc_fwd_bwd,
          sc_fwd_function=sc_fwd,
      )
    ...
    (
        pipeline_state,
        step_i_sc_fwd_out,
        step_im1_tc_out,
        step_im2_sc_bwd_out,
    ) = jitted_train_step(
        pipeline_state=pipeline_state,
        pipeline_input=pipeline_input,
        pipeline_step=jnp.array([step]),
        num_steps=jnp.array([num_steps]),
        ...
    )
    ...
    if ep_utils.is_sc_fwd_valid(step, num_steps):
      # Use SC fwd outputs
    if ep_utils.is_tc_valid(step, num_steps):
      # Use TC outputs
    if ep_utils.is_sc_bwd_valid(step, num_steps):
      # Use SC BWD outputs
    ...
```
"""

import dataclasses
import functools
from typing import Any, Callable, Generic, TypeVar

from flax import struct
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


NestedArray = embedding.Nested[jax.Array]
NestedFeatureSpecs = embedding.Nested[embedding_spec.FeatureSpec]
NestedEmbeddingVariables = embedding.Nested[embedding.EmbeddingVariables]

# stubs for user defined types (internal only, users should not use these types)

# NOTE: should be an instance of flax.struct.dataclass
_PipelineData = TypeVar('_PipelineData', bound=Any)

_TrainState = TypeVar('_TrainState', bound=Any)

# user defined function types
ScFwdStageFun = Callable[
    [_PipelineData, NestedEmbeddingVariables], _PipelineData
]
TcStageFun = Callable[
    [_PipelineData, _TrainState], tuple[_PipelineData, _TrainState]
]
ScBwdStageFun = Callable[
    [
        _PipelineData,
        NestedEmbeddingVariables,
    ],
    # NOTE: no stage after this so pipeline_data can be stopped here, but user
    #   may want to return some additional data, in that case we can change the
    #   return type
    tuple[_PipelineData, NestedEmbeddingVariables],
]


@struct.dataclass
class PipelineState(Generic[_PipelineData, _TrainState]):
  train_state: _TrainState
  embedding_variables: NestedEmbeddingVariables
  step_i_sc_fwd_inputs: _PipelineData
  step_im1_tc_inputs: _PipelineData
  step_im2_sc_bwd_inputs: _PipelineData


def get_pipeline_train_steps(num_steps: int) -> int:
  """Get the number of pipeline train steps."""
  return num_steps + 2


def is_sc_fwd_valid(pipeline_step: int, num_steps: int) -> bool:
  return 0 <= pipeline_step <= num_steps - 1


def is_tc_valid(pipeline_step: int, num_steps: int) -> bool:
  return 1 <= pipeline_step <= num_steps


def is_sc_bwd_valid(pipeline_step: int, num_steps: int) -> bool:
  return 2 <= pipeline_step <= num_steps + 1


@struct.dataclass
class DefaultPipelineData:
  """Default PipelineData."""

  # sc fwd in
  lhs_row_pointers: NestedArray | None = None
  lhs_local_embedding_ids: NestedArray | None = None
  lhs_local_sample_ids: NestedArray | None = None
  lhs_gains: NestedArray | None = None
  # sc fwd out
  emb_act: jax.Array | None = None
  # tc out
  emb_grad: jax.Array | None = None


def get_default_sc_fwd_function(
    feature_specs: NestedFeatureSpecs, global_mesh: jax.sharding.Mesh
) -> ScFwdStageFun:
  """Get the default SC fwd function."""

  if len(global_mesh.shape) != 1:
    raise ValueError(f'global_mesh must be 1d, got {len(global_mesh.shape)}')

  def sc_fwd_default_function(
      args: DefaultPipelineData, embedding_variables: NestedEmbeddingVariables
  ) -> DefaultPipelineData:
    """Default SC fwd function."""

    pd = jax.sharding.PartitionSpec(global_mesh.axis_names[0])
    pe = jax.sharding.PartitionSpec(global_mesh.axis_names[0], None)

    tpu_sparse_dense_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        global_device_count=global_mesh.size,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    tpu_sparse_dense_matmul = jax.experimental.shard_map.shard_map(
        f=tpu_sparse_dense_matmul,
        mesh=global_mesh,
        in_specs=(pd, pd, pd, pd, pe),
        out_specs=pd,
        check_rep=False,
    )
    emb_act = tpu_sparse_dense_matmul(
        args.lhs_row_pointers,
        args.lhs_local_embedding_ids,
        args.lhs_local_sample_ids,
        args.lhs_gains,
        embedding_variables,
    )
    return dataclasses.replace(args, emb_act=emb_act)

  return sc_fwd_default_function


def get_default_sc_bwd_function(
    feature_specs: NestedFeatureSpecs, global_mesh: jax.sharding.Mesh
) -> ScBwdStageFun:
  """Get the default SC bwd function."""

  if len(global_mesh.shape) != 1:
    raise ValueError(f'global_mesh must be 1d, got {len(global_mesh.shape)}')

  def sc_bwd_default_function(
      args: DefaultPipelineData, embedding_variables: NestedEmbeddingVariables
  ) -> tuple[DefaultPipelineData, NestedEmbeddingVariables]:
    """SparseCore backward pass - embedding update."""
    pd = jax.sharding.PartitionSpec(global_mesh.axis_names[0])
    pe = jax.sharding.PartitionSpec(global_mesh.axis_names[0], None)

    tpu_sparse_dense_matmul_grad = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    tpu_sparse_dense_matmul_grad = jax.experimental.shard_map.shard_map(
        f=tpu_sparse_dense_matmul_grad,
        mesh=global_mesh,
        in_specs=(pd, pd, pd, pd, pd, pe),
        out_specs=pe,
        check_rep=False,
    )
    updated_embedding_variables = tpu_sparse_dense_matmul_grad(
        args.emb_grad,
        args.lhs_row_pointers,
        args.lhs_local_embedding_ids,
        args.lhs_local_sample_ids,
        args.lhs_gains,
        embedding_variables,
    )

    return args, updated_embedding_variables

  return sc_bwd_default_function


def get_initial_state(
    train_state: _TrainState,
    embedding_variables: NestedEmbeddingVariables,
    pipeline_input: _PipelineData,
    tc_function: TcStageFun,
    sc_fwd_function: ScFwdStageFun,
) -> PipelineState:
  """Get the initial pipeline state."""

  dummy_sc_fwd_out = jax.tree.map(
      jnp.zeros_like,
      jax.eval_shape(sc_fwd_function, pipeline_input, embedding_variables),
  )
  dummy_tc_out, _ = jax.tree.map(
      jnp.zeros_like,
      jax.eval_shape(
          tc_function,
          dummy_sc_fwd_out,  # require chaining for tracing
          train_state,
      ),
  )

  return PipelineState(
      train_state=train_state,
      embedding_variables=embedding_variables,
      step_i_sc_fwd_inputs=pipeline_input,
      step_im1_tc_inputs=dummy_sc_fwd_out,
      step_im2_sc_bwd_inputs=dummy_tc_out,
  )


def get_dummy_input(
    pipeline_input: _PipelineData,
) -> _PipelineData:
  """Get the dummy input for the pipeline."""
  return jax.tree.map(jnp.zeros_like, pipeline_input)


def step(
    pipeline_input: _PipelineData,
    pipeline_state: PipelineState,
    pipeline_step: jax.Array,
    num_steps: jax.Array,
    sc_fwd_function: ScFwdStageFun,
    tc_function: TcStageFun,
    sc_bwd_function: ScBwdStageFun,
) -> tuple[PipelineState, _PipelineData, _PipelineData, _PipelineData]:
  """Step the pipeline."""

  # feed the input (redundant in case of initial state, but needed thereafter)
  pipeline_state: PipelineState = dataclasses.replace(
      pipeline_state, step_i_sc_fwd_inputs=pipeline_input
  )

  # Step i-2/SC bwd
  # NOTE: short circuit the embedding variables instead of passing through the
  #   sc fwd and tc stage for faster training, otherwise there's a lag of 3
  #   stages before the new embedding variables from this stage is used again at
  #   this stage
  #
  # (cycle 3) SC_BWD i-2 -> SC_FWD i+1 -> TC i -> SC_BWD i-1
  # (cycle 1) SC_BWD i-2 -> (SC_FWD i+1, SC_BWD i-1) directly to both stages
  # NOTE: this at step 1 uses zero emb gradients from TC, so will be effectively
  #   same as no-op, however with optimizer that use momentum, this will
  #   probably do a small update, at step 0, it uses zero grad from
  #   get_initial_state
  with jax.named_scope('pipelined_sc_bwd'):
    step_im2_sc_bwd_out, step_im2_sc_bwd_embedding_variables = sc_bwd_function(
        pipeline_state.step_im2_sc_bwd_inputs,
        pipeline_state.embedding_variables,
    )

  # Step i/SC fwd
  with jax.named_scope('pipelined_sc_fwd'):
    step_i_sc_fwd_out = sc_fwd_function(
        pipeline_state.step_i_sc_fwd_inputs,
        step_im2_sc_bwd_embedding_variables,
        # ^ # use newer embedding variables from step i-2 SC bwd
    )

  # Step i-1/TC fwd/bwd
  # NOTE: short circuit the train state instead of passing through the sc_fwd
  #   stage for faster training, otherwise there's a lag of 2 stages before the
  #   new train state from this stage is used again at this stage
  #
  # (cycle 2) TC i-1 -> SC_FWD i+1 -> TC i
  # (cycle 1) TC i-1 -> (SC_FWD i+1, TC i) directly to both stages
  run_tc = jnp.logical_and(
      jnp.all(2 <= pipeline_step), jnp.all(pipeline_step <= num_steps)
  )
  with jax.named_scope('pipelined_tc'):
    step_im1_tc_out, step_im1_train_state = jax.lax.cond(
        run_tc,
        tc_function,
        # NOTE: when this stage is not run (pipeline_step < 2), the output of
        #   this stage is not used, so we can use the previous step's input (to
        #   make both cond branches have the same output structure), when
        #   (pipeline_step > num_steps), that is the last step, so the output is
        #   not used either
        lambda tc_in, train_state: (
            pipeline_state.step_im2_sc_bwd_inputs,
            train_state,
        ),
        pipeline_state.step_im1_tc_inputs,
        pipeline_state.train_state,
    )

  # NOTE: does not replace the step_i_sc_fwd_inputs to None since that will
  #   cause re-tracing
  next_pipeline_state = dataclasses.replace(
      pipeline_state,
      step_im1_tc_inputs=step_i_sc_fwd_out,
      step_im2_sc_bwd_inputs=step_im1_tc_out,
      train_state=step_im1_train_state,
      embedding_variables=step_im2_sc_bwd_embedding_variables,
  )

  # NOTE: to check for retracing
  assert jax.tree.structure(next_pipeline_state) == jax.tree.structure(
      pipeline_state
  ) and jax.tree.map(lambda x: x.shape, next_pipeline_state) == jax.tree.map(
      lambda y: y.shape, pipeline_state
  ), 'next_pipeline_state and pipeline_state must have the same structure'

  return (
      next_pipeline_state,
      step_i_sc_fwd_out,
      step_im1_tc_out,
      step_im2_sc_bwd_out,
  )
