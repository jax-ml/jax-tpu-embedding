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
"""SparseCore embedding layer."""

import functools
from typing import Any, Callable, Mapping, TypeVar

import flax
from flax import linen as nn
from flax import typing
import jax
from jax.experimental import layout
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


if jax.__version_info__ >= (0, 6, 3):
  DLL = layout.Layout
else:
  DLL = layout.DeviceLocalLayout  # type: ignore
Layout = layout.Format
LogicalNames = typing.LogicalNames
shard_map = jax.experimental.shard_map.shard_map
Nested = embedding.Nested
EmbeddingLookupInput = embedding.PreprocessedInput

EMBEDDING_PARAM_NAME = 'sc_embedding_variables'


A = TypeVar('A')


class WithSparseCoreLayout(nn.Partitioned[A]):

  def get_sharding(self, _: jax.sharding.Mesh) -> jax.sharding.Sharding:
    assert self.mesh is not None
    return Layout(  # pytype: disable=bad-return-type
        DLL(
            major_to_minor=(0, 1),
            tiling=((8,),),
        ),
        jax.sharding.NamedSharding(self.mesh, self.get_partition_spec()),
    )


def with_sparsecore_layout(
    fn: Callable[..., Any],
    names: LogicalNames,
    mesh: jax.sharding.Mesh,
) -> Callable[..., Any]:
  """Wraps a function to add a SparseCore layout."""
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return WithSparseCoreLayout(fn(*args, **kwargs), names, mesh=mesh)

  return wrapper


class SparseCoreEmbed(nn.Module):
  """SparseCore embedding layer.

  ## Pipelining

  This layer supports pipelining of SparseCore computations if
  `enable_pipelining` is set to True. Pipelining decouples the SC computation
  with TC computation by processing multiple batches concurrently stored in
  internal state (variables). This allows for greater SC-TC overlap and
  generally better performance at the cost of higher memory usage.
  There's however a comparitively slower convergence which is tolerable in most
  cases. See internal link:jax-sc-embedding-pipelining for more information.

  When pipelining is enabled, it implements a two-stage pipeline: embedding
  lookups for batch `i` run concurrently with TensorCore computations for batch
  `i-1` and embedding gradient updates for batch `i-2`. This results in
  activations being delayed by one step and gradient updates by two steps
  relative to the inputs.

  NOTE for pipelining:
  * The first two steps return zero activations (warm-up), therefore user needs
  to run two additional steps. The dense input for first(0) and last(N+1) could
  be dummy input.
  * If pipelining is enabled, user will have to pass
  `mutable=['sparsecore_pipeline_state']` to `.apply()` to
  update internal pipeline state.
  """

  # A sequence of FeatureSpecs to specify the configurations for the
  # input feature.
  feature_specs: embedding.Nested[embedding_spec.FeatureSpec]
  # Axis in the mesh to use for sharding.
  sharding_axis: str = 'sparsecore_sharding'
  # Mesh to use for the embedding layer.
  mesh: jax.sharding.Mesh = None  # type: ignore  # initialized in __post_init__
  # Sharding strategy for embedding tables.
  table_sharding_strategy: str = 'MOD'
  enable_minibatching: bool = False
  enable_pipelining: bool = False

  num_sc_per_device: int = -1  # Initialized in __post_init__.

  def __post_init__(self):
    if not self.mesh:
      self.mesh = jax.sharding.Mesh(jax.devices(), [self.sharding_axis])

    self.num_sc_per_device = utils.num_sparsecores_per_device(
        self.mesh.devices.item(0)
    )

    super().__post_init__()

  def setup(self):
    self.embedding_table_partition = jax.sharding.PartitionSpec(
        self.sharding_axis, None
    )
    self.data_partition = jax.sharding.PartitionSpec(self.sharding_axis)
    self.num_shards = self.mesh.shape[self.sharding_axis]

    initializer = functools.partial(
        embedding.init_embedding_variables,
        table_specs=embedding.get_table_specs(self.feature_specs),
        global_sharding=jax.sharding.NamedSharding(
            self.mesh, self.embedding_table_partition
        ),
        num_sparsecore_per_device=self.num_sc_per_device,
        # We need to by-pass the mesh check if not using all
        # JAX devices (build-in assumption to the check).
        bypass_mesh_check=len(self.mesh.devices) != jax.device_count(),
    )
    self.embedding_table = self.param(
        EMBEDDING_PARAM_NAME,
        self._wrap_initializer(initializer),
    )

  def _wrap_initializer(
      self, initializer: Callable[[jax.Array], tuple[jax.Array, ...]]
  ):
    return with_sparsecore_layout(
        fn=initializer,
        names=(self.sharding_axis, None),
        mesh=self.mesh,
    )

  def preprocess_inputs(
      self,
      step: int,
      features: embedding.Nested[np.ndarray],
      features_weights: embedding.Nested[np.ndarray],
      all_reduce_interface: Any | None = None,
  ) -> embedding.PreprocessedInput:
    """Preprocesses the input for sparse dense matmul.

    This method do not need to be invoked with module.apply().

    Args:
      step: The current step
      features: The input features for the current process. The features are
        expected to be Nested type (defined above). Concretely each leaf node
        should be either a 2D numpy array or a 1D list or numpy array of numpy
        arrays with dtype object (in the ragged tensor case).
      features_weights: The input feature weights. The structure must be
        identical to the features.
      all_reduce_interface: The all reduce interface for minibatching. This can
        be generated using the `get_all_reduce_interface` function. Not required
        for single-host minibatching.

    Returns:
      The processed data for embedding lookup.
    """
    return embedding.preprocess_sparse_dense_matmul_input(
        features,
        features_weights,
        self.feature_specs,
        local_device_count=self.mesh.local_mesh.size,
        global_device_count=self.mesh.size,
        num_sc_per_device=self.num_sc_per_device,
        sharding_strategy=self.table_sharding_strategy,
        batch_number=step,
        enable_minibatching=self.enable_minibatching,
        all_reduce_interface=all_reduce_interface,
    )[0]

  @nn.compact
  def __call__(
      self, embedding_lookup_inputs: EmbeddingLookupInput
  ) -> embedding.Nested[jax.Array]:
    """Computes the embedding activations.

    Args:
      embedding_lookup_inputs: The preprocessed data for embedding lookup.

    Returns:
      The activations structure with the same structure as feature_specs.
    """
    if self.enable_pipelining:
      return self._pipelined_call(embedding_lookup_inputs)
    return _emb_lookup(
        self,
        embedding_lookup_inputs,
        self.embedding_table,
    )

  def _get_embedding_table_path(self) -> str:
    """Returns the path to the embedding table within the module."""
    return '/'.join(self.path + (EMBEDDING_PARAM_NAME,))

  @nn.compact
  def apply_gradient(
      self,
      gradients: embedding.Nested[jax.Array],
      embedding_lookup_inputs: EmbeddingLookupInput,
  ) -> Mapping[str, Mapping[str, jax.Array]]:
    """Apply the gradients to the embedding variables.

    Args:
      gradients: The activation gradients.
      embedding_lookup_inputs: The preprocessed data for embedding lookup.

    Returns:
      The updated activation embedding tables.
    """
    if self.enable_pipelining:
      return self._pipelined_apply_gradient(gradients, embedding_lookup_inputs)
    _, embed_table = _emb_lookup_bwd(
        self,
        (embedding_lookup_inputs, self.embedding_table),
        gradients,
    )
    return {self._get_embedding_table_path(): embed_table}

  ##############################################################################
  # Pipelining implementation
  ##############################################################################
  def _pipelined_call(
      self, embedding_lookup_inputs: embedding.PreprocessedInput
  ) -> embedding.Nested[jax.Array]:
    """Pipelined version of __call__."""
    step_im1_sparse_activations = self._get_step_im1_sparse_activations()
    step_im2_sparse_inputs = self._get_step_im2_sparse_inputs(
        embedding_lookup_inputs
    )
    step_im2_sparse_gradients = self._get_step_im2_sparse_gradients()

    # Update embedding table using values from step i-2.
    # Perform the update using a custom_vjp to avoid differentiating the
    # optimizer step.
    updated_table_val = _perform_update(
        self,
        step_im2_sparse_inputs.value,
        step_im2_sparse_gradients.value,
        self.embedding_table,
    )
    if self.is_mutable_collection('params'):
      self.scope.variables()['params'][
          EMBEDDING_PARAM_NAME
      ] = self.scope.variables()['params'][EMBEDDING_PARAM_NAME].replace_boxed(
          updated_table_val
      )

    updated_table = updated_table_val

    # The activations for the current step's forward pass are from step i-1.
    result_activations = step_im1_sparse_activations.value

    # Now, perform the lookup for the current step (i) using the newly updated
    # embedding table and store the inputs and resulting activations for future
    # steps.
    self._get_step_im1_sparse_inputs(embedding_lookup_inputs).value = (
        embedding_lookup_inputs
    )
    step_im1_sparse_activations.value = _emb_lookup(
        self, embedding_lookup_inputs, updated_table
    )

    return result_activations

  def _pipelined_apply_gradient(
      self,
      gradients: embedding.Nested[jax.Array],
      embedding_lookup_inputs: embedding.PreprocessedInput,
  ) -> Mapping[str, Mapping[str, jax.Array]]:
    """Pipelined version of apply_gradient."""
    step_im1_sparse_inputs = self._get_step_im1_sparse_inputs(
        embedding_lookup_inputs
    )
    step_im2_sparse_inputs = self._get_step_im2_sparse_inputs(
        embedding_lookup_inputs
    )
    step_im2_sparse_gradients = self._get_step_im2_sparse_gradients()

    # Store sparse inputs and gradients for use on next step.
    step_im2_sparse_inputs.value = step_im1_sparse_inputs.value
    step_im2_sparse_gradients.value = gradients

    return {}

  def _get_unfrozen_feature_specs(
      self,
  ) -> embedding.Nested[embedding_spec.FeatureSpec]:
    return (
        flax.core.unfreeze(self.feature_specs)
        if isinstance(self.feature_specs, flax.core.FrozenDict)
        else self.feature_specs
    )

  def _get_step_im1_sparse_inputs(
      self, embedding_lookup_inputs: embedding.PreprocessedInput
  ) -> flax.core.scope.Variable[embedding.PreprocessedInput]:
    return self.variable(
        SPARSECORE_PIPELINE_STATE_COLLECTION,
        'step_im1_sparse_inputs',
        lambda: jax.tree.map(jnp.zeros_like, embedding_lookup_inputs),
    )

  def _get_step_im1_sparse_activations(
      self,
  ) -> flax.core.scope.Variable[embedding.Nested[jax.Array]]:
    return self.variable(
        SPARSECORE_PIPELINE_STATE_COLLECTION,
        'step_im1_sparse_activations',
        lambda: jax.tree.map(
            lambda f: jnp.zeros(f.output_shape, dtype=jnp.float32),
            self._get_unfrozen_feature_specs(),
        ),
    )

  def _get_step_im2_sparse_inputs(
      self, embedding_lookup_inputs: embedding.PreprocessedInput
  ) -> flax.core.scope.Variable[embedding.PreprocessedInput]:
    return self.variable(
        SPARSECORE_PIPELINE_STATE_COLLECTION,
        'step_im2_sparse_inputs',
        lambda: jax.tree.map(jnp.zeros_like, embedding_lookup_inputs),
    )

  def _get_step_im2_sparse_gradients(
      self,
  ) -> flax.core.scope.Variable[embedding.Nested[jax.Array]]:
    return self.variable(
        SPARSECORE_PIPELINE_STATE_COLLECTION,
        'step_im2_sparse_gradients',
        lambda: jax.tree.map(
            lambda f: jnp.zeros(f.output_shape, dtype=jnp.float32),
            self._get_unfrozen_feature_specs(),
        ),
    )


################################################################################
# Define embedding lookup.
################################################################################
@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _emb_lookup(
    embedding_layer: SparseCoreEmbed,
    embedding_lookup_inputs: EmbeddingLookupInput,
    emb_table: Mapping[str, tuple[jax.Array, ...]],
):
  pt = embedding_layer.embedding_table_partition
  pd = embedding_layer.data_partition
  return shard_map(
      functools.partial(
          embedding.tpu_sparse_dense_matmul,
          global_device_count=embedding_layer.mesh.size,
          feature_specs=embedding_layer.feature_specs,
          sharding_strategy=embedding_layer.table_sharding_strategy,
          enable_minibatching=embedding_layer.enable_minibatching,
      ),
      mesh=embedding_layer.mesh,
      in_specs=(pd, pt),
      out_specs=pd,
      check_rep=False,
  )(
      embedding_lookup_inputs,
      emb_table,
  )


def _emb_lookup_fwd(
    embedding_layer: SparseCoreEmbed,
    embedding_lookup_inputs: EmbeddingLookupInput,
    emb_table: Mapping[str, tuple[jax.Array, ...]],
):
  return _emb_lookup(
      embedding_layer,
      embedding_lookup_inputs,
      emb_table,
  ), (
      embedding_lookup_inputs,
      emb_table,
  )


def _emb_lookup_bwd(embedding_layer, res, gradients):
  """Backward pass for embedding lookup."""
  (embedding_lookups, emb_table) = res

  pt = embedding_layer.embedding_table_partition
  pd = embedding_layer.data_partition
  updated_emb_table = shard_map(
      functools.partial(
          embedding.tpu_sparse_dense_matmul_grad,
          feature_specs=embedding_layer.feature_specs,
          sharding_strategy=embedding_layer.table_sharding_strategy,
          enable_minibatching=embedding_layer.enable_minibatching,
      ),
      mesh=embedding_layer.mesh,
      in_specs=(pd, pd, pt),
      out_specs=pt,
      check_rep=False,
  )(
      gradients,
      embedding_lookups,
      emb_table,
  )

  # > Reinterpret emb table as grads (unused).
  # tpu_sparse_dense_matmul_grad returns a general Mapping (usually a dict).
  # It may not be the same type as the embedding table (e.g. FrozenDict).
  # Here we use flatten / unflatten to ensure the types are the same.
  emb_table_grads = jax.tree.unflatten(
      jax.tree.structure(emb_table), jax.tree.leaves(updated_emb_table)
  )

  return (
      None,
      emb_table_grads,
  )


_emb_lookup.defvjp(_emb_lookup_fwd, _emb_lookup_bwd)


SPARSECORE_PIPELINE_STATE_COLLECTION = 'sparsecore_pipeline_state'


################################################################################
# Define custom VJP for embedding update for pipelining.
# This is used to prevent autodiff from differentiating through the optimizer
# step.
################################################################################
@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _perform_update(
    module: 'SparseCoreEmbed',
    im2_inputs: embedding.PreprocessedInput,
    im2_grads: embedding.Nested[jax.Array],
    emb_table: embedding.Nested[jax.Array],
) -> embedding.Nested[jax.Array]:
  """Performs the embedding update, but is opaque to autodiff."""
  _, updated_table = _emb_lookup_bwd(  # pylint: disable=protected-access
      module,
      (im2_inputs, emb_table),
      im2_grads,
  )
  return updated_table


def _perform_update_fwd(
    module: 'SparseCoreEmbed',
    im2_inputs: embedding.PreprocessedInput,
    im2_grads: embedding.Nested[jax.Array],
    emb_table: embedding.Nested[jax.Array],
):
  """Forward pass for _perform_update."""
  updated_table = _perform_update(module, im2_inputs, im2_grads, emb_table)
  # Return inputs as residuals for backward pass.
  return updated_table, (im2_inputs, im2_grads, emb_table)


def _perform_update_bwd(
    module: 'SparseCoreEmbed',
    res: tuple[
        embedding.PreprocessedInput,
        embedding.Nested[jax.Array],
        embedding.Nested[jax.Array],
    ],
    g: embedding.Nested[jax.Array],
) -> tuple[
    embedding.PreprocessedInput,
    embedding.Nested[jax.Array],
    embedding.Nested[jax.Array],
]:
  """Backward pass for _perform_update."""
  # g is the gradient w.r.t. the output (updated_table).
  # We want this to flow back to the original emb_table as if this function
  # was an identity function.
  im2_inputs, im2_grads, emb_table = res
  del module, emb_table
  return (
      jax.tree.map(jnp.zeros_like, im2_inputs),
      jax.tree.map(jnp.zeros_like, im2_grads),
      g,  # Pass gradient through to the original embedding table.
  )


_perform_update.defvjp(_perform_update_fwd, _perform_update_bwd)
