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
"""Internal utilities for embedding lookup and update."""

import collections
import functools
from typing import Mapping, Sequence, TypeAlias, TypeVar, Union

import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import tree

ArrayLike = jnp.ndarray | np.ndarray

T: TypeAlias = TypeVar("T")
Nested: TypeAlias = Union[T, Sequence[T], Mapping[str, T]]


def sharding_strategy_to_int(sharding_strategy: str) -> int:
  if sharding_strategy == "MOD":
    return 1
  else:
    raise ValueError(
        f"Unsupported sharding strategy: {sharding_strategy}. Only MOD is"
        " supported."
    )


def _get_activation_for_feature(
    feature: embedding_spec.FeatureSpec,
    activations: dict[str, jax.Array],
    global_device_count: int,
) -> jax.Array:
  """Gets the activation slice for a given feature."""
  assert feature.table_spec.stacked_table_spec is not None
  if feature.id_transformation is None:
    raise ValueError(
        "FeatureIdTransformation cannot be None. It is None for"
        f" {feature.name}",
    )
  per_device_offset = (
      feature.id_transformation.row_offset // global_device_count
  )
  if feature.output_shape[-1] > feature.table_spec.embedding_dim:
    raise ValueError(
        f"Feature {feature.name} has output shape {feature.output_shape} and"
        f" embedding dim {feature.table_spec.embedding_dim}. The output shape"
        " must be at least same as the (original, unpadded)embedding dim."
    )
  return jax.lax.slice(
      activations[feature.table_spec.stacked_table_spec.stack_name],
      (per_device_offset, 0),
      (
          per_device_offset + feature.output_shape[0] // global_device_count,
          feature.output_shape[-1],
      ),
  )


def unstack_embedding_activations(
    activations: dict[str, jax.Array],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
) -> Nested[jax.Array]:
  """Unstacks the activations to match the feature specs."""

  get_activation_for = functools.partial(
      _get_activation_for_feature,
      activations=activations,
      global_device_count=global_device_count,
  )

  return jax.tree_util.tree_map(get_activation_for, feature_specs)


def sharding_strategy_to_enum(sharding_strategy: str) -> int:
  """Converts the sharding strategy string to the enum."""
  if sharding_strategy.upper() == "MOD":
    return 1
  else:
    raise ValueError(
        f"Unsupported sharding strategy: {sharding_strategy}. Only MOD is"
        " supported."
    )


def stack_embedding_gradients(
    activation_gradients: Nested[jax.Array],
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> Mapping[str, jax.Array]:
  """Stacks the gradients for update to embedding variables."""
  stacked_table_to_features = collections.defaultdict(list)
  for gradient, feature in zip(
      tree.flatten(activation_gradients), tree.flatten(feature_specs)
  ):
    assert feature.table_spec.stacked_table_spec is not None
    if feature.id_transformation is None:
      raise ValueError(
          "FeatureIdTransformation cannot be None here. It is None for"
          f" {feature.name}"
      )
    stacked_table_to_features[
        feature.table_spec.stacked_table_spec.stack_name
    ].append((feature, gradient))
  stacked_table_to_gradients = collections.defaultdict(list)
  for stacked_table_name, stacked_features in stacked_table_to_features.items():
    stacked_features.sort(key=lambda x: x[0].id_transformation.row_offset)
    for f, g in stacked_features:
      # feature.table_spec.embedding_dim is the original table dim, before
      # padding
      gradient = g.reshape([-1, f.table_spec.embedding_dim])
      # Add padding for extra cols
      extra_cols = (
          f.table_spec.setting_in_stack.padded_embedding_dim
          - f.table_spec.embedding_dim
      )
      if extra_cols != 0:
        gradient = jax.lax.pad(gradient, 0.0, [(0, 0, 0), (0, extra_cols, 0)])
      stacked_table_to_gradients[stacked_table_name].append(gradient)
  return {
      t: jax.lax.concatenate(grads, dimension=0)
      for t, grads in stacked_table_to_gradients.items()
  }


