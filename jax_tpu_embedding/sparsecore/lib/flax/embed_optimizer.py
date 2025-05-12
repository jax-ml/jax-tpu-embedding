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
"""Optimizer for models with SparseCore modules."""

from typing import Any

import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed
import optax


def _is_emb_path(path: list[Any]) -> bool:
  return any(
      isinstance(level, jax.tree_util.DictKey)
      and level.key == embed.EMBEDDING_PARAM_NAME
      for level in path
  )


def create_optimizer_for_sc_model(
    params: Any, tc_optimizer: optax.GradientTransformation
) -> optax.GradientTransformation:
  """Create the optimizer for the model.

  Args:
    params: A PyTree of model parameters.
    tc_optimizer: The optimizer for the TensorCore part of the model.

  Returns:
    An optax.GradientTransformation that applies updates to the model.
  """
  embedding_params_tree = jax.tree_util.tree_map_with_path(
      lambda path, v: (
          'tc_optimizer' if not _is_emb_path(path) else 'sc_optimizer'
      ),
      params,
  )

  # Create optimizer for the model.
  return optax.multi_transform(
      {
          'tc_optimizer': tc_optimizer,
          'sc_optimizer': _get_optimizer_for_optax(),
      },
      embedding_params_tree,
  )


def apply_updates_for_sc_model(params, updates):
  """Apply the updates to the params for models with SparseCore modules."""

  def apply_update_to_params(path, param, update):
    if not _is_emb_path(path):
      return jnp.asarray(param + update).astype(jnp.asarray(update).dtype)
    else:
      return _apply_update(param, update)

  return jax.tree_util.tree_map_with_path(
      apply_update_to_params,
      params,
      updates,
  )


def _get_optimizer_for_optax() -> optax.GradientTransformation:
  # For now, the optimizer is part of the SC grad op.
  # We create a trivial optimizer to simply return the new embedding table.
  #
  # For the long run, we'd like to have SC grad op to return the real gradients,
  # and this function would need to create the real optimizer for SC.
  return optax.GradientTransformation(
      init=lambda params: optax.EmptyState(),
      update=lambda grads, state, params: (grads, state),
  )


def _apply_update(params, updates):
  # For now, since the grad op and the SC dummy optimizer are
  # returning the updated embedding table as the "update", we just need to
  # use the updated embedding table here.
  #
  # For the long run, we'd like to implement logic to apply the real
  # embedding table updates to embedding tables.
  del params
  return updates
