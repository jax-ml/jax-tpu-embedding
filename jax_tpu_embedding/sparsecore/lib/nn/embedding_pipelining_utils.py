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
"""

from collections.abc import Callable
import functools
import types
from typing import Generic, ParamSpec, Protocol, TypeVar

from flax import struct
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


# stubs for user defined types (internal only, users should not use these types)

# Input to the pipeline stages (SC fwd, TC fwd/bwd, SC bwd)
_SparseInput = TypeVar('_SparseInput')
_DenseInput = TypeVar('_DenseInput')
_EmbeddingActivations = TypeVar('_EmbeddingActivations')
_EmbeddingGradients = TypeVar('_EmbeddingGradients')
_TcAux = TypeVar('_TcAux')
_ScFwdAux = TypeVar('_ScFwdAux')
_ScBwdAux = TypeVar('_ScBwdAux')

# Output from the pipeline stages (TC fwd/bwd)
_PipelineOutput = TypeVar('_PipelineOutput')

# Train state for the TensorCore
_TcTrainState = TypeVar('_TcTrainState')

# Embedding variables for the SparseCore
_EmbeddingVariables = TypeVar('_EmbeddingVariables')

_T = TypeVar('_T')
_P = ParamSpec('_P')


def _eval_shape(
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
  """Similar to jax.eval_shape, but also returns the output shardings."""

  fn = jax.jit(fn)
  out = fn.eval_shape(*args, **kwargs)
  sharding = fn.lower(*args, **kwargs).compile().output_shardings

  return jax.tree.map(
      lambda x, s: _replace_shape_dtype_struct(x, sharding=s),
      out,
      sharding,
  )


def _replace_shape_dtype_struct(
    obj: jax.ShapeDtypeStruct, **kwargs
) -> jax.ShapeDtypeStruct:
  """Replaces the attributes of a ShapeDtypeStruct with the given values."""
  new_kwargs = {
      'shape': obj.shape,
      'dtype': obj.dtype,
      'sharding': obj.sharding,
      'weak_type': obj.weak_type,
  }
  new_kwargs.update(kwargs)
  return jax.ShapeDtypeStruct(**new_kwargs)


def _create_output_arrays(spec: jax.ShapeDtypeStruct):
  """Takes a spec and creates a zero array of the same sharding and shape."""
  if (
      jnp.dtype(spec) == jax.dtypes.float0
      or spec.sharding is None
      or not spec.shape
      or spec.sharding.is_fully_replicated
  ):
    return jnp.zeros_like(spec)
  local_arrays = []
  local_indices_map = spec.sharding.addressable_devices_indices_map(spec.shape)
  for local_device in spec.sharding.addressable_devices:
    if local_device in local_indices_map:
      local_slice = local_indices_map[local_device]

      local_shape = []
      for i, s in enumerate(local_slice):
        if s.stop is None or s.start is None:
          # Replicated dimension
          local_shape.append(spec.shape[i])
        else:
          local_shape.append(s.stop - s.start)
      local_data = jnp.zeros(local_shape, dtype=spec.dtype, device=local_device)
      local_arrays.append(local_data)
    else:
      return jnp.zeros_like(spec)
  return jax.make_array_from_single_device_arrays(
      spec.shape, spec.sharding, local_arrays
  )


# generic functions that user need to instantiate with their own input and
# output types, along with train state and embedding variables. using a protocol
# to use generics with callable
class ScFwdStageFun(
    Protocol[
        _SparseInput, _EmbeddingActivations, _EmbeddingVariables, _ScFwdAux
    ]
):
  """Protocol for the SparseCore forward pass function."""

  def __call__(
      self,
      sparse_inputs: _SparseInput,
      embedding_variables: _EmbeddingVariables,
  ) -> tuple[_EmbeddingActivations, _ScFwdAux]:
    """SparseCore forward pass function.

    Args:
      sparse_inputs: Sparse inputs for the SparseCore forward pass.
      embedding_variables: Embedding variables for the SparseCore.

    Returns:
      A tuple of embedding activations and auxiliary data to TC.
    """
    ...


class TcStageFun(
    Protocol[
        _EmbeddingActivations,
        _DenseInput,
        _EmbeddingGradients,
        _PipelineOutput,
        _TcTrainState,
        _ScFwdAux,
        _TcAux,
    ]
):
  """Protocol for the TensorCore forward/backward pass function."""

  def __call__(
      self,
      embedding_activations: _EmbeddingActivations,
      dense_inputs: _DenseInput,
      train_state: _TcTrainState,
      sc_fwd_aux: _ScFwdAux = None,
  ) -> tuple[_EmbeddingGradients, _PipelineOutput, _TcTrainState, _TcAux]:
    """TensorCore forward/backward pass function.

    Args:
      embedding_activations: Embedding activations from the SparseCore.
      dense_inputs: Dense inputs for the TensorCore.
      train_state: Train state for the TensorCore.
      sc_fwd_aux: Auxiliary data from the SparseCore forward pass.

    Returns:
      A tuple of embedding gradients, pipeline output, updated train state, and
      auxiliary data to SC BWD.
    """
    ...


class ScBwdStageFun(
    Protocol[
        _SparseInput,
        _EmbeddingGradients,
        _EmbeddingVariables,
        _TcAux,
        _ScBwdAux,
    ]
):
  """Protocol for the SparseCore backward pass function."""

  def __call__(
      self,
      sparse_inputs: _SparseInput,
      embedding_gradients: _EmbeddingGradients,
      embedding_variables: _EmbeddingVariables,
      tc_aux: _TcAux = None,
  ) -> tuple[_EmbeddingVariables, _ScBwdAux]:
    """SparseCore backward pass function.

    Args:
      sparse_inputs: Sparse inputs for the SparseCore backward pass.
      embedding_gradients: Embedding gradients from the TensorCore.
      embedding_variables: Embedding variables for the SparseCore.
      tc_aux: Auxiliary data from the TensorCore forward/backward pass.

    Returns:
      A tuple of updated embedding variables and auxiliary output.
    """
    ...


class CurrentStepInput(Generic[_SparseInput, _DenseInput], struct.PyTreeNode):
  """Step i inputs."""

  sparse_inputs: _SparseInput  # used by sc fwd
  dense_inputs: _DenseInput  # used by tc in next step


class LastStepInput(
    Generic[_SparseInput, _DenseInput, _EmbeddingActivations, _ScFwdAux],
    struct.PyTreeNode,
):
  """Step i-1 inputs."""

  sparse_inputs: _SparseInput  # used by sc bwd in next step
  embedding_activations: _EmbeddingActivations  # used by tc fwd/bwd
  dense_inputs: _DenseInput  # used by tc fwd/bwd
  sc_fwd_aux: _ScFwdAux


class StepBeforeLastStepInput(
    Generic[_SparseInput, _EmbeddingGradients, _TcAux],
    struct.PyTreeNode,
):
  """Step i-2 inputs."""

  sparse_inputs: _SparseInput  # used by sc bwd
  embedding_gradients: _EmbeddingGradients  # used by sc bwd
  tc_aux: _TcAux


class PipelineState(
    Generic[
        _SparseInput,
        _DenseInput,
        _EmbeddingActivations,
        _EmbeddingGradients,
        _TcAux,
        _ScFwdAux,
        _PipelineOutput,
    ],
    struct.PyTreeNode,
):
  """Pipeline state.

  Attributes:
    pipeline_step: Current pipeline step (0 based indexing). This is the step
      number of the pipeline, not the training step.
    placeholder_output: Placeholder output from the current pipeline step (TC)
      when run_tc=False and jax.lax.cond needs a placeholder value.
    placeholder_tc_aux: Placeholder auxiliary data from the current pipeline
      step (TC) when run_tc=False and jax.lax.cond needs a placeholder value.
  """

  pipeline_step: jax.Array

  last_step_inputs: LastStepInput[
      _SparseInput, _DenseInput, _EmbeddingActivations, _ScFwdAux
  ]

  step_before_last_step_inputs: StepBeforeLastStepInput[
      _SparseInput, _EmbeddingGradients, _TcAux
  ]

  placeholder_output: _PipelineOutput
  placeholder_tc_aux: _TcAux


def get_pipeline_train_steps(num_steps: int) -> int:
  """Get the number of pipeline train steps."""
  return num_steps + 2


def is_output_valid(pipeline_step: int, num_steps: int) -> bool:
  """Check if the output of the pipeline is valid.

  Output is only valid when the TensorCore (TC) runs which is from step 1 to
  num_steps.

  Args:
    pipeline_step: Current pipeline step (0 based indexing).
    num_steps: Total number of steps.

  Returns:
    True if the output is valid, False otherwise.
  """
  return 1 <= pipeline_step <= num_steps


NestedArray = embedding.Nested[jax.Array]
NestedFeatureSpecs = embedding.Nested[embedding_spec.FeatureSpec]
NestedEmbeddingVariables = embedding.Nested[embedding.EmbeddingVariables]

# You could make these a PyTree and include more information to be passed
# between SC and TC stagest.
DefaultSparseInputs = embedding.PreprocessedInput
DefaultEmbeddingActivations = jax.Array | None
DefaultEmbeddingGradients = jax.Array | None
DefaultScFwdAux = types.NoneType
DefaultTcAux = types.NoneType
DefaultScBwdAux = types.NoneType


def get_default_sc_fwd_function(
    feature_specs: NestedFeatureSpecs, global_mesh: jax.sharding.Mesh
) -> ScFwdStageFun[
    DefaultSparseInputs,
    DefaultEmbeddingActivations,
    NestedEmbeddingVariables,
    DefaultScFwdAux,
]:
  """Get the default SC fwd function.

  Args:
    feature_specs: Feature specs for the embedding layer.
    global_mesh: Global mesh for the embedding layer.

  Returns:
    A function that takes default pipeline stage input/output and embedding
    variables as input and returns updated default pipeline stage input/output.
  """

  if len(global_mesh.shape) != 1:
    raise ValueError(f'global_mesh must be 1d, got {len(global_mesh.shape)}')

  def sc_fwd_default_function(
      sparse_inputs: DefaultSparseInputs,
      embedding_variables: NestedEmbeddingVariables,
  ) -> tuple[DefaultEmbeddingActivations, None]:
    """Default SC fwd function."""

    pd = jax.sharding.PartitionSpec(global_mesh.axis_names[0])
    pe = jax.sharding.PartitionSpec(global_mesh.axis_names[0], None)

    tpu_sparse_dense_matmul = functools.partial(
        embedding.tpu_sparse_dense_matmul,
        global_device_count=global_mesh.size,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    tpu_sparse_dense_matmul = jax.shard_map(
        tpu_sparse_dense_matmul,
        mesh=global_mesh,
        in_specs=(pd, pe),
        out_specs=pd,
        check_vma=False,
    )
    emb_act = tpu_sparse_dense_matmul(sparse_inputs, embedding_variables)
    return emb_act, None

  return sc_fwd_default_function


def get_default_sc_bwd_function(
    feature_specs: NestedFeatureSpecs, global_mesh: jax.sharding.Mesh
) -> ScBwdStageFun[
    DefaultSparseInputs,
    DefaultEmbeddingGradients,
    NestedEmbeddingVariables,
    DefaultTcAux,
    DefaultScBwdAux,
]:
  """Get the default SC bwd function.

  Args:
    feature_specs: Feature specs for the embedding layer.
    global_mesh: Global mesh for the embedding layer.

  Returns:
    A function that takes default pipeline stage input/output and embedding
    variables as input and
    returns updated embedding variables.
  """

  if len(global_mesh.shape) != 1:
    raise ValueError(f'global_mesh must be 1d, got {len(global_mesh.shape)}')

  def sc_bwd_default_function(
      sparse_inputs: DefaultSparseInputs,
      emb_grad: DefaultEmbeddingGradients,
      embedding_variables: NestedEmbeddingVariables,
      tc_aux: DefaultTcAux,
  ) -> tuple[NestedEmbeddingVariables, DefaultScBwdAux]:
    """SparseCore backward pass - embedding update."""
    del tc_aux
    pd = jax.sharding.PartitionSpec(global_mesh.axis_names[0])
    pe = jax.sharding.PartitionSpec(global_mesh.axis_names[0], None)

    tpu_sparse_dense_matmul_grad = functools.partial(
        embedding.tpu_sparse_dense_matmul_grad,
        feature_specs=feature_specs,
        sharding_strategy='MOD',
    )
    tpu_sparse_dense_matmul_grad = jax.shard_map(
        tpu_sparse_dense_matmul_grad,
        mesh=global_mesh,
        in_specs=(pd, pd, pe),
        out_specs=pe,
        check_vma=False,
    )
    updated_embedding_variables = tpu_sparse_dense_matmul_grad(
        emb_grad, sparse_inputs, embedding_variables
    )

    return updated_embedding_variables, None

  return sc_bwd_default_function


def get_pipeline_state_sharding(
    pipeline_state_cls: type[
        PipelineState[
            _SparseInput,
            _DenseInput,
            _EmbeddingActivations,
            _EmbeddingGradients,
            _TcAux,
            _ScFwdAux,
            _PipelineOutput,
        ]
    ],
    dense_input_sharding: jax.sharding.Sharding,
    sparse_input_sharding: jax.sharding.Sharding,
    pipeline_output_sharding: jax.sharding.Sharding,
    tc_aux_sharding: jax.sharding.Sharding,
) -> PipelineState[
    _SparseInput,
    _DenseInput,
    _EmbeddingActivations,
    _EmbeddingGradients,
    _TcAux,
    _ScFwdAux,
    _PipelineOutput,
]:
  """Get the sharding for the pipeline state.

  Args:
    pipeline_state_cls: The class of the pipeline state instantiated with user
      type parameters.
    dense_input_sharding: The sharding of the dense inputs.
    sparse_input_sharding: The sharding of the sparse inputs.
    pipeline_output_sharding: The sharding of the output from the TensorCore.
    tc_aux_sharding: The sharding of the auxiliary data from the TensorCore.

  Returns:
    The sharding of the pipeline state.
  """
  return pipeline_state_cls(
      pipeline_step=None,  # pytype: disable=wrong-arg-types
      last_step_inputs=LastStepInput(
          sparse_inputs=sparse_input_sharding,
          embedding_activations=dense_input_sharding,
          dense_inputs=dense_input_sharding,
          sc_fwd_aux=dense_input_sharding,
      ),
      step_before_last_step_inputs=StepBeforeLastStepInput(
          sparse_inputs=sparse_input_sharding,
          embedding_gradients=dense_input_sharding,
          tc_aux=tc_aux_sharding,
      ),
      placeholder_output=pipeline_output_sharding,
      placeholder_tc_aux=tc_aux_sharding,
  )


def get_initial_state(
    pipeline_input: CurrentStepInput[_SparseInput, _DenseInput],
    tc_train_state: _TcTrainState,
    embedding_variables: _EmbeddingVariables,
    sc_fwd_function: ScFwdStageFun[
        _SparseInput, _EmbeddingActivations, _EmbeddingVariables, _ScFwdAux
    ],
    tc_function: TcStageFun[
        _EmbeddingActivations,
        _DenseInput,
        _EmbeddingGradients,
        _PipelineOutput,
        _TcTrainState,
        _ScFwdAux,
        _TcAux,
    ],
    dense_input_sharding: jax.sharding.Sharding | None = None,
    sparse_input_sharding: jax.sharding.Sharding | None = None,
) -> PipelineState[
    _SparseInput,
    _DenseInput,
    _EmbeddingActivations,
    _EmbeddingGradients,
    _TcAux,
    _ScFwdAux,
    _PipelineOutput,
]:
  """Get the initial pipeline state.

  Args:
    pipeline_input: Input to the sc_fwd, tc and sc_bwd for the current step.
    tc_train_state: Training state for the TensorCore.
    embedding_variables: Embedding variables for the SparseCore.
    sc_fwd_function: SparseCore forward pass function.
    tc_function: TensorCore forward/backward pass function.
    dense_input_sharding: The sharding of the dense inputs.
    sparse_input_sharding: The sharding of the sparse inputs.

  Returns:
    Initial pipeline state.
  """
  sparse_init_fn = jnp.zeros_like
  dense_init_fn = jnp.zeros_like

  # Wrap input initialization functions with sharding if provided.
  # TODO(b/436649494): Try sharding the placeholder inputs implicitly.
  def _create_sharded_array(sharding, x):
    if (
        isinstance(x, int)
        or (hasattr(x, 'ndim') and x.ndim == 0)
        or sharding.is_fully_replicated
    ):
      return x
    return jax.make_array_from_callback(
        x.shape,
        sharding,
        lambda index: jnp.zeros(
            [x.shape[0] // sharding.num_devices, *x.shape[1:]],
            dtype=x.dtype,
        ),
    )

  if sparse_input_sharding is not None:
    sparse_init_fn = functools.partial(
        _create_sharded_array, sparse_input_sharding
    )
  if dense_input_sharding is not None:
    dense_init_fn = functools.partial(
        _create_sharded_array, dense_input_sharding
    )

  dense_placeholder_inputs = jax.tree.map(
      dense_init_fn,
      pipeline_input.dense_inputs,
  )
  sparse_placeholder_inputs = jax.tree.map(
      sparse_init_fn,
      pipeline_input.sparse_inputs,
  )

  placeholder_pipeline_input = CurrentStepInput(
      sparse_inputs=sparse_placeholder_inputs,
      dense_inputs=dense_placeholder_inputs,
  )

  emb_act_shapes, sc_fwd_aux_shapes = jax.eval_shape(
      sc_fwd_function, sparse_placeholder_inputs, embedding_variables
  )
  placeholder_sc_fwd_aux = jax.tree.map(
      jnp.zeros_like,
      sc_fwd_aux_shapes,
  )
  placeholder_emb_act = jax.tree.map(
      sparse_init_fn,
      emb_act_shapes,
  )

  placeholder_emb_grads: _EmbeddingGradients
  placeholder_output: _PipelineOutput
  emb_grads_shapes, output_shapes, _, aux_shapes = _eval_shape(
      tc_function,
      placeholder_emb_act,
      dense_placeholder_inputs,
      tc_train_state,
      placeholder_sc_fwd_aux,
  )

  placeholder_emb_grads = jax.tree.map(
      sparse_init_fn,
      emb_grads_shapes,
  )
  placeholder_output = jax.tree.map(_create_output_arrays, output_shapes)
  aux = jax.tree.map(jnp.zeros_like, aux_shapes)

  # need a few copies for donation, as we reuse some inputs
  return PipelineState(
      pipeline_step=jnp.array(0),
      last_step_inputs=LastStepInput(
          sparse_inputs=placeholder_pipeline_input.sparse_inputs,
          embedding_activations=placeholder_emb_act,
          dense_inputs=placeholder_pipeline_input.dense_inputs,
          sc_fwd_aux=placeholder_sc_fwd_aux,
      ),
      step_before_last_step_inputs=StepBeforeLastStepInput(
          sparse_inputs=jax.tree.map(
              jnp.copy, placeholder_pipeline_input.sparse_inputs
          ),
          embedding_gradients=placeholder_emb_grads,
          tc_aux=aux,
      ),
      placeholder_output=placeholder_output,
      placeholder_tc_aux=jax.tree.map(jnp.copy, aux),
  )


def step(
    # input
    pipeline_input: CurrentStepInput[_SparseInput, _DenseInput],
    # states
    tc_train_state: _TcTrainState,
    embedding_variables: _EmbeddingVariables,
    pipeline_state: PipelineState[
        _SparseInput,
        _DenseInput,
        _EmbeddingActivations,
        _EmbeddingGradients,
        _TcAux,
        _ScFwdAux,
        _PipelineOutput,
    ],
    # functions
    sc_fwd_function: ScFwdStageFun[
        _SparseInput, _EmbeddingActivations, _EmbeddingVariables, _ScFwdAux
    ],
    tc_function: TcStageFun[
        _EmbeddingActivations,
        _DenseInput,
        _EmbeddingGradients,
        _PipelineOutput,
        _TcTrainState,
        _ScFwdAux,
        _TcAux,
    ],
    sc_bwd_function: ScBwdStageFun[
        _SparseInput,
        _EmbeddingGradients,
        _EmbeddingVariables,
        _TcAux,
        _ScBwdAux,
    ],
    *,
    fake_tc_step: bool,
) -> tuple[
    _PipelineOutput,
    _ScBwdAux,
    _TcTrainState,
    _EmbeddingVariables,
    PipelineState[
        _SparseInput,
        _DenseInput,
        _EmbeddingActivations,
        _EmbeddingGradients,
        _TcAux,
        _ScFwdAux,
        _PipelineOutput,
    ],
]:
  """Step the pipeline.

  Args:
    pipeline_input: Inputs for the current step (sc_fwd, tc, sc_bwd).
    tc_train_state: Training state for the TensorCore.
    embedding_variables: Embedding variables for the SparseCore.
    pipeline_state: Internal pipeline state that contains the inputs and outputs
      for and from the three stages that is sequentially shifted by one step at
      each pipeline execution.
    sc_fwd_function: SparseCore forward pass function.
    tc_function: TensorCore forward/backward pass function.
    sc_bwd_function: SparseCore backward pass function.
    fake_tc_step: If true, fake the TC step by copying the embedding gradients
      to the pipeline output.

  Returns:
    A tuple of (pipeline_output, auxiliary output from SC BWD, updated
    train_state, updated embedding_variables, updated pipeline_state).
  """
  # Step i-2/SC bwd
  # NOTE: this at step 1 uses zero emb gradients from TC, so will be effectively
  #   same as no-op, however with optimizer that use slot variables, this will
  #   probably do a small update. For step 0, it uses zero grad from
  #   get_initial_state and hence is also a no-op.
  with jax.named_scope('pipelined_sc_bwd'):
    embedding_variables, sc_bwd_aux = sc_bwd_function(
        pipeline_state.step_before_last_step_inputs.sparse_inputs,
        pipeline_state.step_before_last_step_inputs.embedding_gradients,
        embedding_variables,
        pipeline_state.step_before_last_step_inputs.tc_aux,
    )

  # Step i/SC fwd
  with jax.named_scope('pipelined_sc_fwd'):
    emb_act, sc_fwd_aux = sc_fwd_function(
        pipeline_input.sparse_inputs,
        embedding_variables,
        # ^ # use newer embedding variables from step i-2 SC bwd
    )

  # Step i-1/TC fwd/bwd
  with jax.named_scope('pipelined_tc'):
    if not fake_tc_step:
      emb_grad, pipeline_out, updated_train_state, tc_aux = tc_function(
          pipeline_state.last_step_inputs.embedding_activations,
          pipeline_state.last_step_inputs.dense_inputs,
          tc_train_state,
          pipeline_state.last_step_inputs.sc_fwd_aux,
      )
    else:
      emb_grad, pipeline_out, updated_train_state, tc_aux = (
          pipeline_state.step_before_last_step_inputs.embedding_gradients,
          pipeline_state.placeholder_output,
          tc_train_state,
          pipeline_state.placeholder_tc_aux,
      )

  pipeline_state = pipeline_state.replace(
      pipeline_step=pipeline_state.pipeline_step + 1,
      last_step_inputs=LastStepInput(
          embedding_activations=emb_act,
          sparse_inputs=pipeline_input.sparse_inputs,
          dense_inputs=pipeline_input.dense_inputs,
          sc_fwd_aux=sc_fwd_aux,
      ),
      step_before_last_step_inputs=StepBeforeLastStepInput(
          sparse_inputs=pipeline_state.last_step_inputs.sparse_inputs,
          embedding_gradients=emb_grad,
          tc_aux=tc_aux,
      ),
  )
  return (
      pipeline_out,
      sc_bwd_aux,
      updated_train_state,
      embedding_variables,
      pipeline_state,
  )
