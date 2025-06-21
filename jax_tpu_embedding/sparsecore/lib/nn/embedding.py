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
"""List of functions for embedding lookup."""

import collections
import functools
from typing import List, Mapping, NamedTuple, Sequence, Tuple, TypeAlias, TypeVar, Union

from absl import logging
from flax import struct
import jax
from jax.experimental import shard_map
from jax.experimental.layout import DeviceLocalLayout as DLL
from jax.experimental.layout import Format
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import tree


ArrayLike = jnp.ndarray | np.ndarray

T: TypeAlias = TypeVar("T")
Nested: TypeAlias = Union[T, Sequence[T], Mapping[str, T]]
LimitsCallable: TypeAlias = table_stacking.LimitsCallable
get_default_limits = table_stacking.get_default_limits


class EmbeddingVariablesInitializer(NamedTuple):
  table: embedding_spec.CallableTableInitializer
  slot: tuple[embedding_spec.CallableTableInitializer, ...]


class EmbeddingVariables(NamedTuple):
  table: jax.Array
  slot: tuple[jax.Array, ...]


class SparseDenseMatmulInput(NamedTuple):
  """The result of preprocessing sparse dense matmul input."""

  lhs_row_pointers: Mapping[str, np.ndarray]
  lhs_embedding_ids: Mapping[str, np.ndarray]
  lhs_sample_ids: Mapping[str, np.ndarray]
  lhs_gains: Mapping[str, np.ndarray]


@struct.dataclass
class SparseDenseMatmulInputStats:
  """The stats of preprocessing sparse dense matmul input."""

  max_ids_per_partition: Mapping[str, np.ndarray]
  max_unique_ids_per_partition: Mapping[str, np.ndarray]
  required_buffer_size_per_sc: Mapping[str, np.ndarray]

  @classmethod
  def from_cc(
      cls, stats: pybind_input_preprocessing.SparseDenseMatmulInputStats
  ) -> "SparseDenseMatmulInputStats":
    return cls(
        max_ids_per_partition=stats.max_ids_per_partition,
        max_unique_ids_per_partition=stats.max_unique_ids_per_partition,
        required_buffer_size_per_sc=stats.required_buffer_sizes,
    )

  @classmethod
  def from_dict(
      cls, stats: Mapping[str, Mapping[str, np.ndarray]]
  ) -> "SparseDenseMatmulInputStats":
    return cls(
        max_ids_per_partition=stats["max_ids"],
        max_unique_ids_per_partition=stats["max_unique_ids"],
        required_buffer_size_per_sc=stats["required_buffer_size"],
    )


# TODO: b/346873239 - Add more checks for the feature specs to ensure all the
# fields are valid.
def _verify_feature_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> None:
  """Ensures all the fields in the feature specs are correctly defined."""
  visited_feature_names = set()
  for feature_spec in tree.flatten(feature_specs):
    if feature_spec.name in visited_feature_names:
      raise ValueError(f"Feature spec {feature_spec.name} is already defined.")
    visited_feature_names.add(feature_spec.name)


# TODO: b/346873239 - Add more checks for the table specs.
def _verify_table_specs(table_specs: Nested[embedding_spec.TableSpec]) -> None:
  """Ensures all the fields in the table specs are correctly defined."""
  visited_table_names = set()
  for table_spec in tree.flatten(table_specs):
    if table_spec.name in visited_table_names:
      raise ValueError(f"Table spec {table_spec.name} is already defined.")
    visited_table_names.add(table_spec.name)


def get_table_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> Mapping[str, embedding_spec.TableSpec]:
  """Get the flattened table specs from the feature specs.

  Multiple feature specs can share the same table spec.

  Args:
    feature_specs: A collection of feature specs.

  Returns:
    A map from table names to table specs. These table specs are the feature
    specs pointing to. These table specs are not the stacked table specs.

  Raises:
    ValueError: if there is duplicate table/feature name.
  """
  _verify_feature_specs(feature_specs)
  table_specs = {
      feature_spec.table_spec.name: feature_spec.table_spec
      for feature_spec in tree.flatten(feature_specs)
  }
  _verify_table_specs(table_specs)
  return table_specs


def get_stacked_table_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> Mapping[str, embedding_spec.StackedTableSpec]:
  """Get the flattened stacked table specs from the feature specs.

  Stacked table specs are stacked representation of the table specs. Multiple
  feature specs can share the same stacked table spec.

  Args:
    feature_specs: A collection of feature specs.

  Returns:
    A map from stacked table names to their specs. These table specs are the
    top-most table specs each feature spec pointing to.

  Raises:
    ValueError: if there is duplicate table/feature name.
  """
  _verify_feature_specs(feature_specs)
  stacked_table_specs: list[embedding_spec.StackedTableSpec] = [
      feature_spec.table_spec.stacked_table_spec
      for feature_spec in tree.flatten(feature_specs)
  ]
  if any(s is None for s in stacked_table_specs):
    raise ValueError(
        "Looks like embedding.prepare_feature_specs_for_training was not"
        " called."
    )
  return {
      stacked_table_specs.stack_name: stacked_table_specs
      for stacked_table_specs in stacked_table_specs  # pytype: disable=annotation-type-mismatch
  }


# TODO(b/376860403): Move this to preprocessing/forward/backward pass ops.
def prepare_feature_specs_for_training(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int,
) -> None:
  """Prepares the feature specs for training by populating missing fields.

  Checks that all the feature specs and corresponding table specs are
  populated correctly for training. This is a no-op if all the tables are
  already stacked. For any unstacked tables, this will populate the stacked
  table spec field. The feature specs are updated in place.

  Args:
    feature_specs: Input feature specs.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: Number of sparse cores per device.

  Raises:
    ValueError: If there is duplicate table/feature name or if there is
      invalid table stacking.
  """
  not_stacked = [
      feature
      for feature in tree.flatten(feature_specs)
      if feature.table_spec.stacked_table_spec is None
  ]
  # Amongst the not explicitly stacked features, collect the ones that point
  # to same table.
  not_stacked_tables = collections.defaultdict(list)
  for feature in not_stacked:
    not_stacked_tables[feature.table_spec.name].append(feature)

  feature_to_row_offset = {}
  for table_name, features in not_stacked_tables.items():
    if len(features) > 1:
      logging.info(
          "Will stack the following features for table %s: %s",
          table_name,
          [feature.name for feature in features],
      )
    row_offset = 0
    features.sort(key=lambda f: f.name)
    for feature in features:
      feature_to_row_offset[feature.name] = row_offset
      row_offset += np.prod(feature.output_shape[:-1])

  def _populate_stacking_info_in_features(
      feature: embedding_spec.FeatureSpec,
  ) -> None:
    """Updates the feature spec with stacking info populated."""
    if feature not in not_stacked:
      return
    # Stacked table spec is not populated. Check that the table setting in stack
    # relects that the table is not stacked.
    if (
        feature.table_spec.setting_in_stack is not None
        and feature.table_spec.setting_in_stack.stack_name
        != feature.table_spec.name
    ):
      raise ValueError(
          f"Invalid stacking. Table {feature.table_spec.name} does not have"
          " StackedTableSpec populated, but"
          " feature.table_spec.setting_in_stack.stack_name is not"
          f" {feature.table_spec.name}."
      )
    table_to_padded_dim, tables_to_padded_vocab_size = (
        table_stacking.round_up_dim_and_vocab_size(
            {feature.table_spec.name: feature.table_spec},
            num_sc_per_device * global_device_count,
        )
    )
    total_sample_count = sum([
        np.prod(f.output_shape[:-1])
        for f in not_stacked_tables[feature.table_spec.name]
    ])

    feature.table_spec.setting_in_stack = embedding_spec.TableSettingInStack(
        stack_name=feature.table_spec.name,
        padded_embedding_dim=table_to_padded_dim[feature.table_spec.name],
        padded_vocab_size=tables_to_padded_vocab_size[feature.table_spec.name],
        row_offset_in_shard=0,
        shard_rotation=0,
    )
    feature.table_spec.stacked_table_spec = embedding_spec.StackedTableSpec(
        stack_name=feature.table_spec.name,
        stack_vocab_size=tables_to_padded_vocab_size[feature.table_spec.name],
        stack_embedding_dim=table_to_padded_dim[feature.table_spec.name],
        optimizer=feature.table_spec.optimizer,
        combiner=feature.table_spec.combiner,
        total_sample_count=total_sample_count,
        max_ids_per_partition=feature.table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=feature.table_spec.max_unique_ids_per_partition,
        suggested_coo_buffer_size=feature.table_spec.suggested_coo_buffer_size,
        quantization_config=feature.table_spec.quantization_config,
    )
    feature.id_transformation = embedding_spec.FeatureIdTransformation(
        row_offset=feature_to_row_offset.get(feature.name, 0),
        col_offset=0,
        col_shift=0,
    )
    logging.info(
        "Populated feature spec for %s with: %s",
        feature.name,
        feature_specs,
    )
  for feature in tree.flatten(feature_specs):
    _populate_stacking_info_in_features(feature)


def auto_stack_tables(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
) -> None:
  """Computes the stacked tables based on the feature specs.

  Args:
    feature_specs: A collection of feature specs.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparse cores per device.
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.

  Returns:
    None. The feature specs are updated with stacking information.
  """
  table_stacking.auto_stack_tables(
      feature_specs,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
  )


def sharding_strategy_to_enum(sharding_strategy: str) -> int:
  """Converts the sharding strategy string to the enum."""
  if sharding_strategy.upper() == "MOD":
    return 1
  else:
    raise ValueError(
        f"Unsupported sharding strategy: {sharding_strategy}. Only MOD is"
        " supported."
    )


def preprocess_sparse_dense_matmul_input(
    features: Nested[ArrayLike],
    features_weights: Nested[ArrayLike],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    local_device_count: int,
    global_device_count: int,
    num_sc_per_device: int,
    sharding_strategy: str = "MOD",
    has_leading_dimension: bool = False,
    allow_id_dropping: bool = False,
) -> tuple[SparseDenseMatmulInput, SparseDenseMatmulInputStats]:
  """Preprocesses the input for sparse dense matmul.

  Args:
    features: The input features for the current process. The features are
      expected to be Nested type (defined above). Concretely each leaf node
      should be either a 2D numpy array or a 1D list or numpy array of numpy
      arrays with dtype object (in the ragged tensor case).
    features_weights: The input feature weights. The structure must be identical
      to the features.
    feature_specs: The feature specs. This needs to have the same structure as
      features and features_weights (e.g., if one of them is a mapping then all
      of them are).
    local_device_count: The number of local devices (chips). Typically
      `mesh.local_mesh.size`.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparse cores per device.
    sharding_strategy: The sharding strategy (e.g., MOD)
    has_leading_dimension: If set to True, then the first dimension of the
      output will be the number of local devices. This is useful when using the
      output in jax.pmap. If set to False, then the first dimension of the
      output will be the number of local devices * the static buffer size. This
      is useful when using the output in jax.jit. In conclusion, Set it to True
      if using jax.pmap and set it to False if using jax.jit.
    allow_id_dropping: If set to True, then ids will be dropped if they exceed
      the max_ids_per_partition or max_unique_ids_per_partition limits.

  Returns:
    A tuple of PreprocessSparseDenseMatmulInput and SparseDenseMatmulInputStats.
  """
  tree.assert_same_structure(features, feature_specs)
  tree.assert_same_structure(features_weights, feature_specs)

  *preprocessed_inputs, stats = (
      pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
          tree.flatten(features),
          tree.flatten(features_weights),
          tree.flatten(feature_specs),
          local_device_count,
          global_device_count,
          num_sc_per_device,
          sharding_strategy_to_enum(sharding_strategy),
          has_leading_dimension,
          allow_id_dropping=allow_id_dropping,
      )
  )

  return SparseDenseMatmulInput(
      *preprocessed_inputs
  ), SparseDenseMatmulInputStats.from_cc(stats)


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


def _unstack_embedding_activations(
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


@jax.named_call
def tpu_sparse_dense_matmul(
    preprocessed_inputs: SparseDenseMatmulInput,
    embedding_variables: Mapping[str, EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    sharding_strategy: str = "MOD",
) -> Nested[jax.Array]:
  """Computes the sparse dense matmul.

  This function can be used with jax.jit and/or shard_map or as a complete
  standalone computation.

  Example invocation:

  sparse_matmul = functools.partial(
      embedding.tpu_sparse_dense_matmul,
      global_device_count=mesh.size,
      feature_specs=feature_specs,
      sharding_strategy="MOD",
  )
  sparse_matmul = shard_map.shard_map(
      sparse_matmul,
      mesh=mesh,
      in_specs=(
          P(mesh.axis_names[0]),
          P(mesh.axis_names[0]),
      ),
      out_specs=P(mesh.axis_names[0]),
      check_rep=False,
  )
  sparse_matmul = jax.jit(sparse_matmul)
  activations = sparse_matmul(
      preprocessed_inputs=preprocessed_inputs,
      embedding_variables,
  )

  Args:
    preprocessed_inputs: The preprocessed inputs for sparse dense matmul.
    embedding_variables: A tuple of embedding tables and slot variables. The
      first one is always the embedding table, the following ones are slot
      variables. The tree structure must be identical to the lhs_row_pointers.
    feature_specs: The input features for the current process.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    sharding_strategy: The sharding strategy (e.g., MOD)

  Returns:
    The activations structure with the same structure as feature_specs.

  Raises:
    ValueError: The input arrays and tuples are not of the expected structure or
      the sharding strategy is not supported.
  """
  lhs_row_pointers = preprocessed_inputs.lhs_row_pointers
  lhs_embedding_ids = preprocessed_inputs.lhs_embedding_ids
  lhs_sample_ids = preprocessed_inputs.lhs_sample_ids
  lhs_gains = preprocessed_inputs.lhs_gains

  assert lhs_row_pointers.keys() == embedding_variables.keys()

  stacked_table_specs = get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  sharding_strategy = sharding_strategy_to_enum(sharding_strategy)

  activations = {}
  for stacked_table_name in stacked_table_specs:
    row_pointer = lhs_row_pointers[stacked_table_name]
    embedding_id = lhs_embedding_ids[stacked_table_name]
    sample_id = lhs_sample_ids[stacked_table_name]
    gain = lhs_gains[stacked_table_name]
    embedding_variable = embedding_variables[stacked_table_name]
    stacked_table = stacked_table_specs[stacked_table_name]
    quantization_config = stacked_table.quantization_config
    quantization_config_tuple = (
        quantization_config.as_tuple() if quantization_config else None
    )
    activations[stacked_table.stack_name] = (
        sparse_dense_matmul_csr.tpu_sparse_dense_matmul_csr_primitive.bind(
            row_pointer,
            embedding_id,
            sample_id,
            gain,
            embedding_variable[0],  # [0] is the embedding table
            device_batch_size=stacked_table.total_sample_count
            // global_device_count,
            max_ids_per_partition=stacked_table.max_ids_per_partition,
            max_unique_ids_per_partition=stacked_table.max_unique_ids_per_partition,
            sharding_strategy=sharding_strategy,
            quantization_config=quantization_config_tuple,
        )
    )

  return _unstack_embedding_activations(
      activations, feature_specs, global_device_count
  )


def _stack_embedding_gradients(
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


@jax.named_call
def tpu_sparse_dense_matmul_grad(
    activation_gradients: Nested[jax.Array],
    preprocessed_inputs: SparseDenseMatmulInput,
    embedding_variables: Mapping[str, EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    sharding_strategy: str = "MOD",
    label: str = "",
    step: jax.Array | int | None = None,
) -> Mapping[str, EmbeddingVariables]:
  """Computes the updated embedding variables based on the activation gradients.

  Example invocation with jit + shard_map:

  grad_update = functools.partial(
      embedding.tpu_sparse_dense_matmul_grad,
      feature_specs=feature_specs,
      sharding_strategy="MOD",
  )
  grad_update = shard_map.shard_map(
      grad_update,
      mesh=mesh,
      in_specs=(
          P(mesh.axis_names[0]),
          P(mesh.axis_names[0]),
          P(mesh.axis_names[0]),
      ),
      out_specs=P(mesh.axis_names[0]),
      check_rep=False,
  )

  grad_update = jax.jit(grad_update)
  updated_embedding_variables = grad_update(
      activations_grad,
      preprocessed_inputs=preprocessed_inputs,
      embedding_variables,
  )

  Args:
    activation_gradients: The activation gradients.
    preprocessed_inputs: The preprocessed inputs for sparse dense matmul.
    embedding_variables: A tuple of embedding tables and slot variables. The
      first one is always the embedding table, the following ones are slot
      variables. The tree structure must be identical to the lhs_row_pointers.
    feature_specs: The input features for the current process.
    sharding_strategy: The sharding strategy (e.g., MOD)
    label: The label for the optimizer computation.
    step: The current step number.

  Returns:
    The updated activation embedding variables.
  """
  # Verify the input structures and lengths.
  lhs_row_pointers = preprocessed_inputs.lhs_row_pointers
  lhs_embedding_ids = preprocessed_inputs.lhs_embedding_ids
  lhs_sample_ids = preprocessed_inputs.lhs_sample_ids
  lhs_gains = preprocessed_inputs.lhs_gains

  assert lhs_row_pointers.keys() == embedding_variables.keys()
  # Activations match the feature specs structure
  tree.assert_same_structure(feature_specs, activation_gradients)

  stacked_table_specs = get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  gradients = _stack_embedding_gradients(activation_gradients, feature_specs)
  assert lhs_row_pointers.keys() == gradients.keys()

  sharding_strategy = sharding_strategy_to_enum(sharding_strategy)

  updated_embedding_variables = {}
  for stacked_table_name in stacked_table_specs:
    row_pointer = lhs_row_pointers[stacked_table_name]
    embedding_id = lhs_embedding_ids[stacked_table_name]
    sample_id = lhs_sample_ids[stacked_table_name]
    gain = lhs_gains[stacked_table_name]
    embedding_variable = embedding_variables[stacked_table_name]
    activation_gradient = gradients[stacked_table_name]
    stack_table_spec = stacked_table_specs[stacked_table_name]
    hyper_params = stack_table_spec.optimizer.get_hyperparameters(step)
    # The MLIR computation symbol names need to be different. We attach the
    # table name to the symbol name to ensure that.
    symbol_name = "{}-{}{}".format(
        stack_table_spec.optimizer.short_name(),
        stack_table_spec.stack_name,
        label,
    )
    optimizer_primitive = stack_table_spec.optimizer.get_optimizer_primitive()

    flatten_variables, treedef = jax.tree.flatten(embedding_variable)
    updated_variables = optimizer_primitive.bind(
        row_pointer,
        embedding_id,
        sample_id,
        gain,
        *flatten_variables,
        activation_gradient,
        *hyper_params,
        max_ids_per_partition=stack_table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=stack_table_spec.max_unique_ids_per_partition,
        computation_name=symbol_name,
        sharding_strategy=sharding_strategy,
    )

    updated_embedding_variables[stacked_table_name] = jax.tree.unflatten(
        treedef,
        jax.tree.leaves(updated_variables),
    )

  return updated_embedding_variables


def _init_embedding_variables_shard(
    rng: jax.Array,
    tspec: embedding_spec.TableSpec,
    num_sparsecore: int,
) -> EmbeddingVariables:
  """Initializes a shard of an embedding variable for a single SparseCore.

  Args:
    rng: The random number generator.
    tspec: The table spec for the embedding variable.
    num_sparsecore: The number of sparsecore per device.

  Returns:
    A tuple of initializers for the embedding variable and slot variables.
  """
  shape = (
      tspec.setting_in_stack.padded_vocab_size // num_sparsecore,
      tspec.setting_in_stack.padded_embedding_dim,
  )

  initializers = EmbeddingVariablesInitializer(
      table=tspec.initializer,
      slot=tspec.optimizer.slot_variables_initializers(),
  )
  return EmbeddingVariables(
      *jax.tree.map(
          lambda init: init(rng, shape), initializers
      )
  )


def _init_stacked_embedding_table_shard(
    rng: jax.Array,
    table_specs: List[embedding_spec.TableSpec],
    num_global_shards: int,
    num_sparsecore_per_device: int,
) -> EmbeddingVariables:
  """Initializes a shard of a stacked table.

  Args:
    rng: The random number generator.
    table_specs: A list of table specs that are in the same stack.
    num_global_shards: The number of global shards.
    num_sparsecore_per_device: Number of sparsecores per TPU device.

  Returns:
    A tuple of embedding table shards for all the tables in the stack.
  """
  # Each device has `num_sparsecore_per_device` sparsecores. An embedding table
  # shard for a device is constructed by initializing sparsecore shards for
  # each table and then stacking the sparsecore shards.
  num_sparsecore = num_sparsecore_per_device * num_global_shards
  table_shards = []
  for i in range(num_sparsecore_per_device):
    for tspec in table_specs:
      table_shards.append(
          _init_embedding_variables_shard(rng[i], tspec, num_sparsecore)
      )
  return jax.tree.map(
      lambda *shards: jax.numpy.concatenate(list(shards)), *table_shards
  )


def _init_stacked_embedding_table(
    rng: jax.Array,
    stack_name: str,
    table_specs: List[embedding_spec.TableSpec],
    global_sharding: jax.sharding.NamedSharding,
    sharding_axis: str | Tuple[str, ...],
    num_sparsecore_per_device: int,
) -> EmbeddingVariables:
  """Initializes a stacked embedding table."""
  logging.info(
      "Creating embedding variable for stack: %s with tables: %s",
      stack_name,
      [t.name for t in table_specs],
  )

  settings = [tspec.setting_in_stack for tspec in table_specs]
  # dim of all the tables in the stack should be the same because that's how
  # the stack is defined.
  stack_dim = settings[0].padded_embedding_dim
  assert all(s.padded_embedding_dim == stack_dim for s in settings)

  P: TypeAlias = jax.sharding.PartitionSpec
  num_global_shards = global_sharding.mesh.size
  init_func = lambda rng: _init_stacked_embedding_table_shard(
      rng=rng,
      table_specs=table_specs,
      num_global_shards=num_global_shards,
      num_sparsecore_per_device=num_sparsecore_per_device,
  )

  use_pmap = len(global_sharding.spec) == 3

  if not use_pmap:
    embedding_table = jax.jit(
        shard_map.shard_map(
            init_func,
            mesh=global_sharding.mesh,
            in_specs=P(sharding_axis),
            out_specs=global_sharding.spec,
        ),
        out_shardings=Format(
            DLL(
                major_to_minor=(0, 1),
                _tiling=((8,),),
            ),
            global_sharding,
        ),
    )(
        rng,
    )
    # TODO: b/377517742 - Fix layout issue for PMAP cases.
  else:
    local_device_count = jax.local_device_count()
    embedding_table = jax.pmap(init_func)(
        rng.reshape(
            (local_device_count, rng.shape[0] // local_device_count, -1)
        )
    )
  return embedding_table


def init_embedding_variables(
    rng: jax.Array,
    table_specs: Nested[embedding_spec.TableSpec],
    global_sharding: jax.sharding.NamedSharding,
    num_sparsecore_per_device: int = -1,
    bypass_mesh_check: bool = False,
) -> Mapping[str, EmbeddingVariables]:
  """Generates the initial embedding variables.

  Args:
    rng: the random number generator.
    table_specs: a collection of table specs.
    global_sharding: defines how the embedding variables are laid out across
      devices. When using with jit, the partition spec should be 2 dimensional,
      where the 0th dimension denotes the vocab dimension, the 1st dimension
      denotes the embedding dimension. Since the sharding should happen along
      the vocab dimension, the partition spec should be
      sharding.PartitionSpec("x", None). When using with pmap, the parititon
      spec should be 3 dimensional where the 0th dimension represents the device
      dimension, the 1st dimension represents the vocab dimension and the 2nd
      dimension represents the embedding dimension. Since the sharding should
      happen along the device dimension, the partition spec should be
      sharding.PartitionSpec("x", None, None).
    num_sparsecore_per_device: Number of sparsecore per device. default = -1 to
      query from the TPU type of global_sharding.mesh.devices.item(0).
    bypass_mesh_check: If True, don't require the mesh device order to match
      jax.devices(). This can be used when exporting a model from a host where
      the training devices are not present.

  Returns:
    A dictionary of embedding variable initializers. Each key is the table name
    and each value is the embedding variable initializer.

  Raises:
    ValueError: if there is duplicate table name, or if the number of
                sparsecores could not be determined.
  """
  # When using pmap, the partition spec should be 3 dimensional where
  # * the 0th dimension represents the device dimension
  # * the 1st dimension represents the vocab # dimension
  # * the 2nd dimension represents the embedding dimension
  # Since the sharding should happen along the device dimension, the partition
  # spec should be sharding.PartitionSpec("x", None, None).
  #
  # When using jit, the partition spec should be 2 dimensional, where
  # * the 0th dimension denotes the vocab dimension
  # * the 1st dimension denotes the embedding dimension
  # Since the sharding should happen along the vocab dimension,
  # the partition spec should be sharding.PartitionSpec("x", None).
  sharding_axis = next((s for s in global_sharding.spec if s is not None), None)
  if num_sparsecore_per_device < 0:
    if not isinstance(global_sharding.mesh.devices, np.ndarray):
      raise ValueError(
          "Cannot determine the number of sparsecores from the provided mesh. "
          "The parameter `num_sparsecore_per_device` must be specified."
      )

    num_sparsecore_per_device = utils.num_sparsecores_per_device(
        global_sharding.mesh.devices.item(0)
    )

  if sharding_axis is None or (
      global_sharding.spec != (sharding_axis, None, None)
      and global_sharding.spec != (sharding_axis, None)
  ):
    raise ValueError(
        "PartitionSpec of the global sharding either needs to be in the format"
        " PartitionSpec('x', None, None) or PartitionSpec('x',"
        f" None). Got {global_sharding.spec}"
    )

  if not bypass_mesh_check and (
      global_sharding.mesh.devices is None
      or not np.array_equal(
          global_sharding.mesh.devices.flatten(), jax.devices()
      )
  ):
    raise ValueError(
        "global_sharding needs to be created with default device order from"
        " jax.device(), but "
        f" global_sharding.mesh.devices={global_sharding.mesh.devices!r} and"
        f" jax.devices()={jax.devices()!r}"
    )

  stacks = collections.defaultdict(list)
  for table_spec in tree.flatten(table_specs):
    stacks[table_spec.setting_in_stack.stack_name].append(table_spec)

  # Make sure the table specs are sorted by their position in the stack
  for table_specs in stacks.values():
    table_specs.sort(key=lambda x: x.setting_in_stack.row_offset_in_shard)

  # Initialize the RNG keys for each table/shard once. This is much faster than
  # initializing the RNG keys for each table/shard separately when needed.
  use_pmap = len(global_sharding.spec) == 3
  if use_pmap:
    per_table_width = jax.local_device_count() * num_sparsecore_per_device
    rngs = jax.random.split(rng, per_table_width * len(stacks))
  else:
    per_table_width = (
        global_sharding.mesh.size * num_sparsecore_per_device
    )
    rngs = jax.random.split(
        rng,
        per_table_width * len(stacks),
    )

  stacked_mapping = {}
  for i, (stack_name, table_specs) in enumerate(stacks.items()):
    rngs_for_table = rngs[i * per_table_width : (i + 1) * per_table_width]
    stacked_mapping[stack_name] = _init_stacked_embedding_table(
        rng=rngs_for_table,
        stack_name=stack_name,
        table_specs=table_specs,
        global_sharding=global_sharding,
        sharding_axis=sharding_axis,
        num_sparsecore_per_device=num_sparsecore_per_device,
    )
  return stacked_mapping


def create_proto_from_feature_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sparsecore_per_device: int,
) -> embedding_spec_pb2.EmbeddingSpecProto:
  """Creates a StackedTableSpecProto from a list of FeatureSpec.

  This is used to create the proto for feature sets used for training. The proto
  captures relevant information for the features such that the
  training variables can be unsharded when being loaded from a checkpoint,
  for serving.

  Args:
    feature_specs: A Nested (e.g., list, dict etc.) of FeatureSpec.
    global_device_count: The number of devices in the system.
    num_sparsecore_per_device: The number of sparsecores per device.

  Returns:
    An EmbeddingSpecProto.
  """
  stacked_table_specs: dict[str, embedding_spec_pb2.StackedTableSpecProto] = {}
  stack_to_table_specs: dict[
      str, dict[str, embedding_spec_pb2.TableSpecProto]
  ] = collections.defaultdict(dict)
  # Traverse the feature specs and create the StackedTableSpecProto.
  for feature in tree.flatten(feature_specs):
    assert feature.table_spec.stacked_table_spec is not None
    current_stack_name = feature.table_spec.stacked_table_spec.stack_name
    current_table_name = feature.table_spec.name
    if current_stack_name not in stacked_table_specs:
      stacked_table_specs[current_stack_name] = (
          embedding_spec_pb2.StackedTableSpecProto(
              stack_name=current_stack_name,
              stack_vocab_size=feature.table_spec.stacked_table_spec.stack_vocab_size,
              stack_embedding_dim=feature.table_spec.stacked_table_spec.stack_embedding_dim,
              total_sample_count=feature.table_spec.stacked_table_spec.total_sample_count,
              max_ids_per_partition=feature.table_spec.stacked_table_spec.max_ids_per_partition,
              num_sparsecores=(num_sparsecore_per_device * global_device_count),
              max_unique_ids_per_partition=feature.table_spec.stacked_table_spec.max_unique_ids_per_partition,
          )
      )
    if current_table_name not in stack_to_table_specs[current_stack_name]:
      stack_to_table_specs[current_stack_name][current_table_name] = (
          embedding_spec_pb2.TableSpecProto(
              table_name=current_table_name,
              vocab_size=feature.table_spec.vocabulary_size,
              embedding_dim=feature.table_spec.embedding_dim,
              padded_vocab_size=feature.table_spec.setting_in_stack.padded_vocab_size,
              padded_embedding_dim=feature.table_spec.setting_in_stack.padded_embedding_dim,
              row_offset_in_shard=feature.table_spec.setting_in_stack.row_offset_in_shard,
              shard_rotation=feature.table_spec.setting_in_stack.shard_rotation,
          )
      )
    feature_spec = embedding_spec_pb2.FeatureSpecProto(
        feature_name=feature.name,
        row_offset=feature.id_transformation.row_offset,
        col_offset=feature.id_transformation.col_offset,
        col_shift=feature.id_transformation.col_shift,
        input_shape=feature.input_shape,
        output_shape=feature.output_shape,
    )
    stack_to_table_specs[current_stack_name][
        current_table_name
    ].feature_specs.append(feature_spec)
  for stack_name, specs in stack_to_table_specs.items():
    stacked_table_specs[stack_name].table_specs.extend(specs.values())
  return embedding_spec_pb2.EmbeddingSpecProto(
      stacked_table_specs=stacked_table_specs.values()
  )
