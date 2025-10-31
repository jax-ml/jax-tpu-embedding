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
from typing import Any

from flax import nnx
import jax
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import optax


Nested = embedding.Nested


class EmbeddingVariablesParam(nnx.Param):
  """A Param subclass for embedding variables, used to filter model params."""

  pass


def is_embedding_variables(_, value) -> bool:
  """NNX filter function to select only embedding variables."""
  return isinstance(value, EmbeddingVariablesParam)


def is_non_embedding_variables(_, value) -> bool:
  """NNX filter function to select only non-embedding variables."""
  return not isinstance(value, EmbeddingVariablesParam)


def get_named_sharding(
    node: nnx.Module | nnx.Optimizer, mesh: jax.sharding.Mesh
):
  """Returns the named sharding for the model or optimizer for use with nnx.jit.

  Args:
    node: The model or optimizer.
    mesh: The mesh to use for the sharding.

  Returns:
    The named sharding for the model or optimizer.

  Example usage:
    model_sharding = embed.get_named_sharding(model, mesh)
    optimizer_sharding = embed.get_named_sharding(optimizer, mesh)
    @nnx.jit(
        in_shardings=(
            model_sharding,
            optimizer_sharding,
            data_sharding,
            data_sharding,
        ),
        donate_argnames=[
            'model',
            'optimizer',
        ],
    )
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        embedding_lookup_inputs: embedding.PreprocessedInput,
        labels: jax.Array,
    ):
      ...
  """

  def fn(_, x):
    if isinstance(x, EmbeddingVariablesParam):
      return x.embedding_sharding
    else:
      return nnx.get_named_sharding(x, mesh)

  return nnx.StateSharding(nnx.map_state(fn, nnx.state(node)))


class PartitionedOptimizer(nnx.Optimizer):
  """A partitioned optimizer for models that use the SparseCoreEmbed layer.

  This is a wrapper around the nnx.Optimizer that splits the gradients into
  embedding and non-embedding variables.

  For effficiency, JAX SparseCore returns the fully updated embedding variables
  from the gradient computation so they must be handled separately so the
  "update" is simply a copy. The non-embedding (dense) variables are updated by
  an nnx.Optimizer using the provided Optax optimizer.
  """

  def __init__(
      self,
      model: nnx.Module,
      dense_optimizer: optax.GradientTransformation,
  ):
    super().__init__(model, dense_optimizer, wrt=is_non_embedding_variables)

  def update(self, model: nnx.Module, grads: Any):
    """Compute and apply optimzier updates for the model."""

    # JAX SparseCore returns the updated embedding variables as the gradients so
    # we just update the model directly.
    updated_embedding_variables = nnx.state(grads, is_embedding_variables)
    nnx.update(model, updated_embedding_variables)

    # The rest of the gradients are applied by the nnx.Optimizer.
    super().update(model, grads)


class SparseCoreEmbed(nnx.Module):
  """SparseCore embedding layer."""

  def __init__(
      self,
      feature_specs: embedding.Nested[embedding_spec.FeatureSpec],
      sharding_axis: str,
      mesh: jax.sharding.Mesh,
      rngs: nnx.Rngs,
      *,
      enable_minibatching: bool = False,
      table_sharding_strategy: str = 'MOD',
  ):
    """Creates a SparseCore embedding layer.

    Args:
      feature_specs: A sequence of FeatureSpecs to specify the configurations
        for the input feature.
      sharding_axis: Axis in the mesh to use for sharding.
      mesh: Mesh to use for the embedding layer.
      rngs: The random number generators for the embedding layer.
      enable_minibatching: Whether to enable minibatching.
      table_sharding_strategy: Sharding strategy for embedding tables.
    """
    self.feature_specs = feature_specs
    self.sharding_axis = sharding_axis
    self.mesh = mesh
    self.table_sharding_strategy = table_sharding_strategy
    self.enable_minibatching = enable_minibatching
    self.num_sc_per_device = utils.num_sparsecores_per_device(
        self.mesh.devices.item(0)
    )

    self.embedding_table_partition = jax.sharding.PartitionSpec(
        self.sharding_axis, None
    )
    self.data_partition = jax.sharding.PartitionSpec(self.sharding_axis)
    self.num_shards = self.mesh.shape[self.sharding_axis]

    # Table initializers are specified in the TableSpec of the FeatureSpec.
    initializer = functools.partial(
        embedding.init_embedding_variables,
        rng=rngs.params(),
        table_specs=embedding.get_table_specs(self.feature_specs),
        global_sharding=jax.sharding.NamedSharding(
            self.mesh, self.embedding_table_partition
        ),
        num_sparsecore_per_device=self.num_sc_per_device,
        # We need to by-pass the mesh check if not using all
        # JAX devices (build-in assumption to the check).
        bypass_mesh_check=len(self.mesh.devices) != jax.device_count(),
    )
    # Note, we stash the embedding table format in the metadata so it can be
    # used by the get_named_sharding() function provided above. This is critical
    # for setting the in_shardings for the nnx.jit annotation to avoid
    # unnecesary copy/reshape operations.
    self.embedding_table = EmbeddingVariablesParam(
        initializer(),
        embedding_sharding=utils.embedding_table_format(
            self.mesh, self.embedding_table_partition
        ),
    )

  def __call__(
      self, embedding_lookup_inputs: embedding.PreprocessedInput
  ) -> embedding.Nested[jax.Array]:
    """Computes the embedding activations.

    Args:
      embedding_lookup_inputs: The preprocessed data for embedding lookup.

    Returns:
      The activations structure with the same structure as feature_specs.
    """
    return embedding_lookup(
        self,
        embedding_lookup_inputs,
    )


################################################################################
# Define embedding lookup.
################################################################################
@nnx.custom_vjp
def embedding_lookup(
    embedding_layer: SparseCoreEmbed,
    embedding_lookup_inputs: embedding.PreprocessedInput,
):
  pt = embedding_layer.embedding_table_partition
  pd = embedding_layer.data_partition
  return nnx.shard_map(
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
      check_vma=False,
  )(
      embedding_lookup_inputs,
      embedding_layer.embedding_table.value,
  )


def _embedding_lookup_fwd(
    embedding_layer: SparseCoreEmbed,
    embedding_lookup_inputs: embedding.PreprocessedInput,
):
  return embedding_lookup(
      embedding_layer,
      embedding_lookup_inputs,
  ), (
      embedding_layer,
      embedding_lookup_inputs,
  )


def embedding_lookup_bwd(res, g):
  """Backward pass for embedding lookup."""
  (embedding_layer, embedding_lookup_inputs) = res
  # input_updates_g, out_g = g
  (m_updates_g, unused_), out_g = g
  m_g = jax.tree.map(lambda x: x, m_updates_g)  # create a copy

  pt = embedding_layer.embedding_table_partition
  pd = embedding_layer.data_partition
  emb_grad_result = nnx.shard_map(
      functools.partial(
          embedding.tpu_sparse_dense_matmul_grad,
          feature_specs=embedding_layer.feature_specs,
          sharding_strategy=embedding_layer.table_sharding_strategy,
          enable_minibatching=embedding_layer.enable_minibatching,
      ),
      mesh=embedding_layer.mesh,
      in_specs=(pd, pd, pt),
      out_specs=pt,
      check_vma=False,
  )(
      out_g,
      embedding_lookup_inputs,
      embedding_layer.embedding_table.value,
  )

  # tpu_sparse_dense_matmul_grad returns a general Mapping (usually a dict).
  # It may not be the same type as the embedding table (e.g. FrozenDict).
  # Here we use flatten / unflatten to ensure the types are the same.
  emb_grad_result = jax.tree.unflatten(
      jax.tree.structure(embedding_layer.embedding_table.value),
      jax.tree.leaves(emb_grad_result),
  )

  m_g['embedding_table'].value = emb_grad_result

  return (m_g, unused_)


embedding_lookup.defvjp(_embedding_lookup_fwd, embedding_lookup_bwd)
