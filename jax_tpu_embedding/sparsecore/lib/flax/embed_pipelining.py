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
"""SparseCore layer to pipeline computations with TensorCore."""

import functools
from typing import Mapping

from absl import logging
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


SPARSECORE_PIPELINE_STATE_COLLECTION = 'sparsecore_pipeline_state'


class PipelinedSparseCoreEmbed(embed.SparseCoreEmbed):
  """A SparseCore embedding layer with pipelining support.

  This decouples the SC computation with TC computation by processing multiple
  batches concurrently stored in internal state (variables). This allows for
  greater SC-TC
  overlap and generally better performance at the cost of higher memory usage.
  There's however a comparitively slower convergence which is tolerable in most
  cases. See internal link:jax-sc-embedding-pipelining for more information.

  It implements a two-stage pipeline: embedding lookups for batch `i` run
  concurrently with TensorCore computations for batch `i-1` and embedding
  gradient updates for batch `i-2`. This results in activations being
  delayed by one step and gradient updates by two steps relative to the inputs.

  TODO {{bugsnag;a:manuadg;p:2;s:2;t:fr;c:1846895}} - Enable SC pipelining for
  Flax layer.

  There are couple of missing optimizations that would be later
  added (requires cl/811416653):

  * Store stacked activations and load stacked gradients to avoid reshaping
  during SC operation (Currently we unstack during lookup and stack during
  gradient update.)

  NOTE:
  * The first two steps return zero activations (warm-up), therefore user needs
  to run two additional steps. The dense input for first(0) and last(N+1) could
  be dummy input.
  * User will have to pass `mutable=True` to `.apply()/.apply_gradient()` to
  update internal pipeline state.
  """

  @nn.compact
  def __call__(
      self, embedding_lookup_inputs: embedding.PreprocessedInput
  ) -> embedding.Nested[jax.Array]:

    step_im1_sparse_activations = self._get_step_im1_sparse_activations()
    step_im2_sparse_inputs = self._get_step_im2_sparse_inputs(
        embedding_lookup_inputs
    )
    step_im2_sparse_gradients = self._get_step_im2_sparse_gradients()

    # Update embedding table using values from step i-2.
    # Perform the update using a custom_vjp to avoid differentiating the
    # optimizer step.
    updated_table = _perform_update(
        self,
        step_im2_sparse_inputs.value,
        step_im2_sparse_gradients.value,
        self.embedding_table,
    )

    # The activations for the current step's forward pass are from step i-1.
    result_activations = step_im1_sparse_activations.value

    # Now, perform the lookup for the current step (i) using the newly updated
    # embedding table and store the inputs and resulting activations for future
    # steps.
    self._get_step_im1_sparse_inputs(embedding_lookup_inputs).value = (
        embedding_lookup_inputs
    )
    step_im1_sparse_activations.value = embed._emb_lookup(
        self, embedding_lookup_inputs, updated_table
    )

    return result_activations

  @nn.compact
  def apply_gradient(
      self,
      gradients: embedding.Nested[jax.Array],
      embedding_lookup_inputs: embedding.PreprocessedInput,
  ) -> Mapping[str, Mapping[str, jax.Array]]:

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

  ##############################################################################
  # Variables
  ##############################################################################

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
# Define custom VJP for embedding update.
# This is used to prevent autodiff from differentiating through the optimizer
# step.
################################################################################
@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _perform_update(
    module: 'PipelinedSparseCoreEmbed',
    im2_inputs: embedding.PreprocessedInput,
    im2_grads: embedding.Nested[jax.Array],
    emb_table: embedding.Nested[jax.Array],
) -> embedding.Nested[jax.Array]:
  """Performs the embedding update, but is opaque to autodiff."""
  _, updated_table = embed._emb_lookup_bwd(  # pylint: disable=protected-access
      module,
      (im2_inputs, emb_table),
      im2_grads,
  )
  return updated_table


def _perform_update_fwd(
    module: 'PipelinedSparseCoreEmbed',
    im2_inputs: embedding.PreprocessedInput,
    im2_grads: embedding.Nested[jax.Array],
    emb_table: embedding.Nested[jax.Array],
):
  """Forward pass for _perform_update."""
  updated_table = _perform_update(module, im2_inputs, im2_grads, emb_table)
  # Return inputs as residuals for backward pass.
  return updated_table, (im2_inputs, im2_grads, emb_table)


def _perform_update_bwd(
    module: 'PipelinedSparseCoreEmbed',
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
