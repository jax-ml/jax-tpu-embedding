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
import dataclasses
import functools
import textwrap
from typing import List, Mapping, NamedTuple, Sequence, TypeAlias, TypeVar, Union
import warnings

from absl import logging
import einops
from flax import struct
import jax
from jax.experimental import shard_map
from jax.experimental.layout import Format
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import pybind_input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


if jax.__version_info__ >= (0, 6, 3):
  from jax.experimental.layout import Layout as DLL  # pylint: disable=g-import-not-at-top
else:
  from jax.experimental.layout import DeviceLocalLayout as DLL  # pylint: disable=g-import-not-at-top  # type: ignore

ArrayLike = jnp.ndarray | np.typing.ArrayLike


FeatureStackingStrategy = pybind_input_preprocessing.FeatureStackingStrategy

T: TypeAlias = TypeVar("T")
Nested: TypeAlias = Union[T, Sequence[T], Mapping[str, T]]
LimitsCallable: TypeAlias = table_stacking.LimitsCallable
get_default_limits = table_stacking.get_default_limits


def _assert_same_structure(a, b, a_name: str = "a", b_name: str = "b"):
  """Asserts that two structures have the same nested structure."""
  a_paths = [path for path, _ in jax.tree_util.tree_leaves_with_path(a)]
  b_paths = [path for path, _ in jax.tree_util.tree_leaves_with_path(b)]
  if a_paths != b_paths:
    raise ValueError(textwrap.dedent(f"""\
        Pytree structures of {a_name} and {b_name} do not match.
        {a_name}: {jax.tree.structure(a)}
        {b_name}: {jax.tree.structure(b)}"""))


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
  """The stats of preprocessing sparse dense matmul input.

  Multiple value indicate per partition or per sparsecore values, whereas a
  single value could also be used in cases where maximum is required.
  """

  max_ids_per_partition: dict[str, np.ndarray] = dataclasses.field(
      metadata={"suffix": "_max_ids"}
  )
  max_unique_ids_per_partition: dict[str, np.ndarray] = dataclasses.field(
      metadata={"suffix": "_max_unique_ids"}
  )
  required_buffer_size_per_sc: dict[str, np.ndarray] = dataclasses.field(
      metadata={"suffix": "_required_buffer_size"}, default_factory=dict
  )
  id_drop_counters: dict[str, int] = dataclasses.field(
      metadata={"suffix": "_id_drop_counters"}, default_factory=dict
  )

  @classmethod
  def from_cc(
      cls, stats: pybind_input_preprocessing.SparseDenseMatmulInputStats
  ) -> "SparseDenseMatmulInputStats":
    return cls(
        max_ids_per_partition=stats.max_ids_per_partition,
        max_unique_ids_per_partition=stats.max_unique_ids_per_partition,
        required_buffer_size_per_sc=stats.required_buffer_sizes,
        id_drop_counters=stats.dropped_id_count,
    )


class PreprocessedInput(struct.PyTreeNode):
  """The result of preprocessing input for sparse dense matmul.

  Attributes:
    sparse_dense_matmul_input: The sparse dense matmul input.
    num_minibatches: The number of minibatches.

  Properties:
    lhs_row_pointers: The row pointers of the sparse matrix.
    lhs_embedding_ids: The embedding ids of the sparse matrix.
    lhs_sample_ids: The sample ids of the sparse matrix.
    lhs_gains: The gains of the sparse matrix.
  """

  sparse_dense_matmul_input: SparseDenseMatmulInput
  num_minibatches: np.ndarray = struct.field(
      default_factory=lambda: np.array(1)
  )

  # Backward compatibility properties and functions. This class acts as a
  # drop-in replacement for SparseDenseMatmulInput.
  @property
  def lhs_row_pointers(self) -> Mapping[str, np.ndarray]:
    return self.sparse_dense_matmul_input.lhs_row_pointers

  @property
  def lhs_embedding_ids(self) -> Mapping[str, np.ndarray]:
    return self.sparse_dense_matmul_input.lhs_embedding_ids

  @property
  def lhs_sample_ids(self) -> Mapping[str, np.ndarray]:
    return self.sparse_dense_matmul_input.lhs_sample_ids

  @property
  def lhs_gains(self) -> Mapping[str, np.ndarray]:
    return self.sparse_dense_matmul_input.lhs_gains

  def __iter__(self):
    warnings.warn(
        "Please use attributed lookup on sparse_dense_matmul_input attribute.",
        DeprecationWarning,
    )
    return iter(self.sparse_dense_matmul_input)


# TODO: b/346873239 - Add more checks for the feature specs to ensure all the
# fields are valid.
def _verify_feature_specs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
) -> None:
  """Ensures all the fields in the feature specs are correctly defined."""
  visited_feature_names = set()
  for feature_spec in jax.tree.leaves(feature_specs):
    if feature_spec.name in visited_feature_names:
      raise ValueError(f"Feature spec {feature_spec.name} is already defined.")
    visited_feature_names.add(feature_spec.name)


# TODO: b/346873239 - Add more checks for the table specs.
def _verify_table_specs(table_specs: Nested[embedding_spec.TableSpec]) -> None:
  """Ensures all the fields in the table specs are correctly defined."""
  visited_table_names = set()
  for table_spec in jax.tree.leaves(table_specs):
    if table_spec.name in visited_table_names:
      raise ValueError(f"Table spec {table_spec.name} is already defined.")
    visited_table_names.add(table_spec.name)


def _get_num_sc_per_device(num_sc_per_device: int | None) -> int:
  """Get the number of sparse cores per device.

  Args:
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.

  Returns:
    The number of sparse cores per device.

  Raises:
    ValueError: If the given number of sparse cores per device is invalid.
  """
  if num_sc_per_device is None:
    return utils.num_sparsecores_per_device()
  elif num_sc_per_device not in utils.NUM_SC_PER_DEVICE_MAP.values():
    raise ValueError(f"Invalid num_sc_per_device: {num_sc_per_device}")
  return num_sc_per_device


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
      for feature_spec in jax.tree.leaves(feature_specs)
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
  if any(
      not feature_spec.table_spec.is_stacked()
      for feature_spec in jax.tree.leaves(feature_specs)
  ):
    raise ValueError(
        "embedding.prepare_feature_specs_for_training was not called"
    )
  stacked_table_specs: list[embedding_spec.StackedTableSpec] = [
      feature_spec.table_spec.stacked_table_spec
      for feature_spec in jax.tree.leaves(feature_specs)
  ]
  return {
      stacked_table_specs.stack_name: stacked_table_specs
      for stacked_table_specs in stacked_table_specs  # pytype: disable=annotation-type-mismatch
  }


# TODO(b/376860403): Move this to preprocessing/forward/backward pass ops.
def prepare_feature_specs_for_training(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int | None = None,
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
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.

  Raises:
    ValueError: If there is duplicate table/feature name or if there is
      invalid table stacking.
  """
  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)
  not_stacked = [
      feature
      for feature in jax.tree.leaves(feature_specs)
      if not feature.table_spec.is_stacked()
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
          " feature.table_spec.setting_in_stack.stack_name"
          f" ({feature.table_spec.setting_in_stack.stack_name}) is not"
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
        suggested_coo_buffer_size_per_device=feature.table_spec.suggested_coo_buffer_size_per_device,
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
        feature,
    )

  for feature in jax.tree.leaves(feature_specs):
    _populate_stacking_info_in_features(feature)


def auto_stack_tables(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int | None = None,
    stack_to_max_ids_per_partition: LimitsCallable = get_default_limits,
    stack_to_max_unique_ids_per_partition: LimitsCallable = get_default_limits,
    use_short_stack_names: bool = True,
) -> None:
  """Computes the stacked tables based on the feature specs.

  Args:
    feature_specs: A collection of feature specs.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.
    stack_to_max_ids_per_partition: Override the max_ids_per_partition for each
      stack.
    stack_to_max_unique_ids_per_partition: Override the
      max_unique_ids_per_partition for each stack.
    use_short_stack_names: If true, use the short stack names.

  Returns:
    None. The feature specs are updated with stacking information.
  """
  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)
  table_stacking.auto_stack_tables(
      feature_specs,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
      stack_to_max_ids_per_partition=stack_to_max_ids_per_partition,
      stack_to_max_unique_ids_per_partition=stack_to_max_unique_ids_per_partition,
      use_short_stack_names=use_short_stack_names,
  )


def sharding_strategy_to_enum(
    sharding_strategy: str,
) -> pybind_input_preprocessing.ShardingStrategy:
  """Converts the sharding strategy string to the enum."""
  if sharding_strategy.upper() == "MOD":
    return pybind_input_preprocessing.ShardingStrategy.Mod
  else:
    raise ValueError(
        f"Unsupported sharding strategy: {sharding_strategy}. Only MOD is"
        " supported."
    )


def get_all_reduce_interface(
    peer_addresses: Sequence[str],
    minibatching_port: int,
    host_id: int | None = None,
    host_count: int | None = None,
) -> pybind_input_preprocessing.AllReduceInterface:
  """Gets an AllReduceInterface for inter-process communication.

  Args:
    peer_addresses: A sequence of addresses for other participating processes.
      These should be in a format compatible with gRPC channel creation, such as
      "host:port", "ipv4:port", or "[ipv6]:port".
    minibatching_port: The port number to be used for minibatching
      communication.
    host_id: The current host's index. If `None`, it will be set to
      `jax.process_index()`.
    host_count: The total number of hosts. If `None`, it will be set to
      `jax.process_count()`.

  Returns:
    An instance of `pybind_input_preprocessing.AllReduceInterface`.
  """
  if host_id is None:
    host_id = jax.process_index()
  if host_count is None:
    host_count = jax.process_count()
  return pybind_input_preprocessing.MinibatchingNode(
      host_id,
      host_count,
      peer_addresses,
      minibatching_port,
  ).get_all_reduce_interface()


def preprocess_sparse_dense_matmul_input(
    features: Nested[ArrayLike],
    features_weights: Nested[ArrayLike],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    local_device_count: int,
    global_device_count: int,
    *,
    num_sc_per_device: int | None = None,
    sharding_strategy: str = "MOD",
    has_leading_dimension: bool = False,
    allow_id_dropping: bool = False,
    feature_stacking_strategy: FeatureStackingStrategy = FeatureStackingStrategy.SPLIT_THEN_STACK,
    batch_number: int = 0,
    enable_minibatching: bool = False,
    all_reduce_interface: (
        pybind_input_preprocessing.AllReduceInterface | None
    ) = None,
) -> tuple[PreprocessedInput, SparseDenseMatmulInputStats]:
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
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.
    sharding_strategy: The sharding strategy (e.g., MOD)
    has_leading_dimension: If set to True, then the first dimension of the
      output will be the number of local devices. This is useful when using the
      output in jax.pmap. If set to False, then the first dimension of the
      output will be the number of local devices * the static buffer size. This
      is useful when using the output in jax.jit. In conclusion, Set it to True
      if using jax.pmap and set it to False if using jax.jit.
    allow_id_dropping: If set to True, then ids will be dropped if they exceed
      the max_ids_per_partition or max_unique_ids_per_partition limits.
    feature_stacking_strategy: The feature stacking strategy.
    batch_number: The batch number.
    enable_minibatching: Whether to enable minibatching.
    all_reduce_interface: Interface to communicate between multiple hosts. This
      can be generated using the `get_all_reduce_interface` function. Not
      required for single-host minibatching.

  Returns:
    A tuple of PreprocessResults and SparseDenseMatmulInputStats.
  """
  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)
  _assert_same_structure(features, feature_specs, "features", "feature_specs")
  _assert_same_structure(
      features_weights, feature_specs, "features_weights", "feature_specs"
  )

  if (
      enable_minibatching
      and all_reduce_interface is None
      and local_device_count < global_device_count
  ):
    raise ValueError(
        "all_reduce_interface must be provided when minibatching is enabled for"
        " multi-host."
    )

  *csr_inputs, num_minibatches, stats = (
      pybind_input_preprocessing.PreprocessSparseDenseMatmulInput(
          jax.tree.leaves(features),
          jax.tree.leaves(features_weights),
          jax.tree.leaves(feature_specs),
          local_device_count,
          global_device_count,
          num_sc_per_device=num_sc_per_device,
          sharding_strategy=sharding_strategy_to_enum(sharding_strategy),
          has_leading_dimension=has_leading_dimension,
          allow_id_dropping=allow_id_dropping,
          feature_stacking_strategy=feature_stacking_strategy,
          batch_number=batch_number,
          enable_minibatching=enable_minibatching,
          all_reduce_interface=all_reduce_interface,
      )
  )

  minibatches_arr = np.full(local_device_count, num_minibatches)
  if has_leading_dimension:
    minibatches_arr.reshape(local_device_count, 1)

  return (
      PreprocessedInput(SparseDenseMatmulInput(*csr_inputs), minibatches_arr),
      SparseDenseMatmulInputStats.from_cc(stats),
  )


def preprocess_sparse_dense_matmul_input_from_sparse_tensor(
    indices: Nested[ArrayLike],
    values: Nested[ArrayLike],
    dense_shapes: Nested[ArrayLike],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    local_device_count: int,
    global_device_count: int,
    *,
    num_sc_per_device: int | None = None,
    sharding_strategy: str = "MOD",
    has_leading_dimension: bool = False,
    allow_id_dropping: bool = False,
    feature_stacking_strategy: FeatureStackingStrategy = FeatureStackingStrategy.SPLIT_THEN_STACK,
    batch_number: int = 0,
    enable_minibatching: bool = False,
    all_reduce_interface: (
        pybind_input_preprocessing.AllReduceInterface | None
    ) = None,
) -> tuple[PreprocessedInput, SparseDenseMatmulInputStats]:
  """Preprocesses the input for sparse dense matmul.

  Args:
    indices: A nested structure of 2-D int64 tensors, where each tensor has
      shape [N, ndims]. It represents the indices of non-zero elements in a
      sparse tensor, with elements being zero-indexed. For instance,
      `indices=[[1,3], [2,4]]` indicates that elements at [1,3] and [2,4] have
      non-zero values.
    values: A nested structure of 1-D tensors, each with shape [N], representing
      the values of non-zero elements corresponding to `indices`. For example,
      with `indices=[[1,3], [2,4]]`, `values=[18, 3.6]` means the element at
      [1,3] is 18 and at [2,4] is 3.6.
    dense_shapes: A nested structure of 2-element 1-D int64 tensors, defining
      the dense shape of the sparse tensor. It specifies the number of elements
      in each dimension. For example, `dense_shape=[3,6]` represents a 3x6
      tensor.
    feature_specs: The feature specs. This needs to have the same structure as
      indices, values and dense_shapes (e.g., if one of them is a mapping then
      all of them are).
    local_device_count: The number of local devices (chips). Typically
      `mesh.local_mesh.size`.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.
    sharding_strategy: The sharding strategy (e.g., MOD)
    has_leading_dimension: If set to True, then the first dimension of the
      output will be the number of local devices. This is useful when using the
      output in jax.pmap. If set to False, then the first dimension of the
      output will be the number of local devices * the static buffer size. This
      is useful when using the output in jax.jit. In conclusion, Set it to True
      if using jax.pmap and set it to False if using jax.jit.
    allow_id_dropping: If set to True, then ids will be dropped if they exceed
      the max_ids_per_partition or max_unique_ids_per_partition limits.
    feature_stacking_strategy: The feature stacking strategy.
    batch_number: The batch number.
    enable_minibatching: Whether to enable minibatching.
    all_reduce_interface: Interface to communicate between multiple hosts. This
      can be generated using the `get_all_reduce_interface` function. Not
      required for single-host minibatching.

  Returns:
    A tuple of PreprocessResults and SparseDenseMatmulInputStats.
  """
  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)
  _assert_same_structure(indices, feature_specs, "indices", "feature_specs")
  _assert_same_structure(values, feature_specs, "values", "feature_specs")
  _assert_same_structure(
      dense_shapes, feature_specs, "dense_shapes", "feature_specs"
  )

  if (
      enable_minibatching
      and all_reduce_interface is None
      and local_device_count < global_device_count
  ):
    raise ValueError(
        "all_reduce_interface must be provided when minibatching is enabled for"
        " multi-host."
    )

  *csr_inputs, num_minibatches, stats = (
      pybind_input_preprocessing.PreprocessSparseDenseMatmulSparseCooInput(
          jax.tree.leaves(indices),
          jax.tree.leaves(values),
          jax.tree.leaves(dense_shapes),
          jax.tree.leaves(feature_specs),
          local_device_count,
          global_device_count,
          num_sc_per_device=num_sc_per_device,
          sharding_strategy=sharding_strategy_to_enum(sharding_strategy),
          has_leading_dimension=has_leading_dimension,
          allow_id_dropping=allow_id_dropping,
          feature_stacking_strategy=feature_stacking_strategy,
          batch_number=batch_number,
          enable_minibatching=enable_minibatching,
          all_reduce_interface=all_reduce_interface,
      )
  )

  minibatches_arr = np.full(local_device_count, num_minibatches)
  if has_leading_dimension:
    minibatches_arr.reshape(local_device_count, 1)

  return (
      PreprocessedInput(SparseDenseMatmulInput(*csr_inputs), minibatches_arr),
      SparseDenseMatmulInputStats.from_cc(stats),
  )


def _get_activation_for_feature(
    feature: embedding_spec.FeatureSpec,
    activations: dict[str, jax.Array],
    global_device_count: int,
    num_feature_slices_per_device: int = 1,
) -> jax.Array:
  """Gets the activation slice for a given feature."""
  if feature.output_shape[-1] > feature.table_spec.embedding_dim:
    raise ValueError(
        f"Feature {feature.name} has output shape {feature.output_shape} and"
        f" embedding dim {feature.table_spec.embedding_dim}. The embedding dim"
        " in output shape cannot be more than the (original,"
        " unpadded) embedding dim in the FeatureSpec."
    )
  stacked_table_activation = activations[
      feature.table_spec.stacked_table_spec.stack_name
  ]
  row_offset_per_stacked_feature_slice = (
      feature.id_transformation.row_offset
      // (global_device_count * num_feature_slices_per_device)
  )
  per_feature_slice_batch_size = feature.output_shape[0] // (
      global_device_count * num_feature_slices_per_device
  )
  _verify_input_batch_size(
      stacked_table_activation.shape,
      num_feature_slices_per_device,
      name=feature.name,
  )
  # d: padded embedding_dim
  # b: padded global (stacked-feature)-slice batch-size
  activation_per_slice = einops.rearrange(
      stacked_table_activation,
      "(f b) d -> f b d",
      f=num_feature_slices_per_device,
  )
  # For each SC, take the subslice corresponding to the current feature.
  feature_slice = activation_per_slice[
      :,
      row_offset_per_stacked_feature_slice : row_offset_per_stacked_feature_slice
      + per_feature_slice_batch_size,
      0 : feature.output_shape[-1],
  ]
  # Merge SC outputs.
  # b1: padded global (per-feature)-slice batch-size
  return einops.rearrange(feature_slice, "f b1 d -> (f b1) d")


StackingStrategy = pybind_input_preprocessing.FeatureStackingStrategy


def unstack_embedding_activations(
    activations: dict[str, jax.Array],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    global_device_count: int,
    num_sc_per_device: int,
    feature_stacking_strategy: StackingStrategy = StackingStrategy.SPLIT_THEN_STACK,
) -> Nested[jax.Array]:
  """Unstacks the activations to match the feature specs."""

  match feature_stacking_strategy:
    case StackingStrategy.STACK_THEN_SPLIT:
      num_feature_slices_per_device = 1
    case StackingStrategy.SPLIT_THEN_STACK:
      num_feature_slices_per_device = num_sc_per_device
    case _:
      raise ValueError(
          f"Unsupported feature stacking strategy: {feature_stacking_strategy}"
      )

  get_activation_for = functools.partial(
      _get_activation_for_feature,
      activations=activations,
      global_device_count=global_device_count,
      num_feature_slices_per_device=num_feature_slices_per_device,
  )
  return jax.tree_util.tree_map(get_activation_for, feature_specs)


@jax.named_call
def tpu_sparse_dense_matmul(
    preprocessed_inputs: PreprocessedInput | SparseDenseMatmulInput,
    embedding_variables: Mapping[str, EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    *,
    global_device_count: int,
    sharding_strategy: str = "MOD",
    feature_stacking_strategy: StackingStrategy = StackingStrategy.SPLIT_THEN_STACK,
    num_sc_per_device: int | None = None,
    enable_minibatching: bool = False,
    perform_unstacking: bool = True,
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
      feature_stacking_strategy=StackingStrategy.SPLIT_THEN_STACK,
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
    feature_stacking_strategy: The feature stacking strategy.
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.
    enable_minibatching: Whether to enable minibatching. Defaults to `False`.
    perform_unstacking: If True, returns per-feature activations by unstacking
      the results. If False, returns raw stacked activations.

  Returns:
    The activations structure with the same structure as feature_specs.

  Raises:
    ValueError: The input arrays and tuples are not of the expected structure or
      the sharding strategy is not supported.
  """
  if isinstance(preprocessed_inputs, SparseDenseMatmulInput):
    warnings.warn(
        "SparseDenseMatmulInput is deprecated. Please use PreprocessedInput"
        " instead.",
        DeprecationWarning,
    )
    # backward compatibility with older input format.
    preprocessed_inputs = PreprocessedInput(preprocessed_inputs)
  (
      lhs_row_pointers,
      lhs_embedding_ids,
      lhs_sample_ids,
      lhs_gains,
  ) = preprocessed_inputs.sparse_dense_matmul_input

  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)

  assert lhs_row_pointers.keys() == embedding_variables.keys()

  stacked_table_specs = get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  # Casting to int since primitives requires JSON serializable value.
  sharding_strategy = int(sharding_strategy_to_enum(sharding_strategy))

  num_minibatches = preprocessed_inputs.num_minibatches
  if num_minibatches.ndim == 1:
    num_minibatches = num_minibatches[0]

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
            num_minibatches,
            embedding_variable.table,
            device_batch_size=stacked_table.total_sample_count
            // global_device_count,
            max_ids_per_partition=stacked_table.max_ids_per_partition,
            max_unique_ids_per_partition=stacked_table.max_unique_ids_per_partition,
            sharding_strategy=sharding_strategy,
            quantization_config=quantization_config_tuple,
            enable_minibatching=enable_minibatching,
        )
    )

  if perform_unstacking:

    activations = unstack_embedding_activations(
        activations,
        feature_specs,
        global_device_count,
        num_sc_per_device,
        feature_stacking_strategy,
    )

  return activations


def _verify_input_batch_size(
    input_shape: tuple[int, ...], feature_slice_per_device: int, name: str
) -> None:
  """Verifies that the feature batch size is divisible by the number of feature slices per device."""
  if input_shape[0] % feature_slice_per_device:
    raise ValueError(
        "The input batch size must be divisible by the number of feature"
        " slices per device. Got input shape"
        f" {input_shape} and {feature_slice_per_device} feature slices"
        f" per device for feature {name}."
    )


def stack_embedding_gradients(
    activation_gradients: Nested[jax.Array],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    num_sc_per_device: int,
    feature_stacking_strategy: StackingStrategy = StackingStrategy.SPLIT_THEN_STACK,
) -> Mapping[str, jax.Array]:
  """Stacks the gradients for update to embedding variables."""
  stacked_table_to_features: dict[
      str, list[tuple[embedding_spec.FeatureSpec, jax.Array]]
  ] = collections.defaultdict(list)
  for gradient, feature in zip(
      jax.tree.leaves(activation_gradients), jax.tree.leaves(feature_specs)
  ):
    if feature.id_transformation is None:
      raise ValueError(
          "FeatureIdTransformation cannot be None here. It is None for"
          f" {feature.name}"
      )
    stacked_table_to_features[
        feature.table_spec.stacked_table_spec.stack_name
    ].append((feature, gradient))
  stacked_table_to_gradients = collections.defaultdict(list)

  if feature_stacking_strategy == StackingStrategy.STACK_THEN_SPLIT:
    feature_slice_per_device = 1
  else:
    feature_slice_per_device = num_sc_per_device

  result: dict[str, jax.Array] = {}

  for stacked_table_name, stacked_features in stacked_table_to_features.items():
    stacked_features.sort(key=lambda x: x[0].id_transformation.row_offset)
    for feature, gradient in stacked_features:
      # feature.table_spec.embedding_dim is the original table dim, before
      # padding
      gradient = gradient.reshape([-1, feature.table_spec.embedding_dim])
      # Add padding for extra cols
      extra_cols = (
          feature.table_spec.setting_in_stack.padded_embedding_dim
          - feature.table_spec.embedding_dim
      )
      if extra_cols != 0:
        gradient = jax.lax.pad(gradient, 0.0, [(0, 0, 0), (0, extra_cols, 0)])
      _verify_input_batch_size(
          gradient.shape, feature_slice_per_device, name=feature.name
      )
      # Slice the feature.
      # b: batch size per slice, d: padded embedding dim
      gradient = einops.rearrange(
          gradient, "(f b) d -> f b d", f=feature_slice_per_device
      )
      stacked_table_to_gradients[stacked_table_name].append(gradient)

    # Concatenate along batch dimension.
    result[stacked_table_name] = jax.lax.concatenate(
        stacked_table_to_gradients[stacked_table_name], dimension=1
    )
    # Merge the feature slice dimension with the batch dimension.
    result[stacked_table_name] = einops.rearrange(
        result[stacked_table_name], "f b d -> (f b) d"
    )

  return result


@jax.named_call
def tpu_sparse_dense_matmul_grad(
    activation_gradients: Nested[jax.Array],
    preprocessed_inputs: PreprocessedInput | SparseDenseMatmulInput,
    embedding_variables: Mapping[str, EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    *,
    sharding_strategy: str = "MOD",
    feature_stacking_strategy: StackingStrategy = StackingStrategy.SPLIT_THEN_STACK,
    label: str = "",
    step: jax.Array | int | None = None,
    num_sc_per_device: int | None = None,
    enable_minibatching: bool = False,
    perform_stacking: bool = True,
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
    feature_stacking_strategy: The feature stacking strategy.
    label: The label for the optimizer computation.
    step: The current step number.
    num_sc_per_device: The number of sparse cores per device. If `None`, it will
      be set to the number of sparse cores on the current host machine.
    enable_minibatching: Whether to use minibatching. Defaults to `False`.
    perform_stacking: If True, expects per-feature gradients and stacks them
      internally. If False, assumes activation_gradients are already stacked.

  Returns:
    The updated activation embedding variables.
  """
  if isinstance(preprocessed_inputs, SparseDenseMatmulInput):
    warnings.warn(
        "SparseDenseMatmulInput is deprecated. Please use PreprocessedInput"
        " instead.",
        DeprecationWarning,
    )
    # backward compatibility with older input format.
    preprocessed_inputs = PreprocessedInput(preprocessed_inputs)
  (
      lhs_row_pointers,
      lhs_embedding_ids,
      lhs_sample_ids,
      lhs_gains,
  ) = preprocessed_inputs.sparse_dense_matmul_input

  # Verify the input structures and lengths.
  assert lhs_row_pointers.keys() == embedding_variables.keys()

  stacked_table_specs = get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  num_sc_per_device = _get_num_sc_per_device(num_sc_per_device)

  if perform_stacking:
    # Activations match the feature specs structure
    _assert_same_structure(
        feature_specs,
        activation_gradients,
        "feature_specs",
        "activation_gradients",
    )

    gradients = stack_embedding_gradients(
        activation_gradients,
        feature_specs,
        num_sc_per_device,
        feature_stacking_strategy,
    )
  else:
    assert isinstance(activation_gradients, Mapping)
    gradients = activation_gradients

  assert lhs_row_pointers.keys() == gradients.keys()

  # Casting to int since primitives requires JSON serializable value.
  sharding_strategy = int(sharding_strategy_to_enum(sharding_strategy))

  num_minibatches = preprocessed_inputs.num_minibatches
  if num_minibatches.ndim == 1:
    num_minibatches = num_minibatches[0]

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
    extra_kwargs = {}
    if isinstance(stack_table_spec.optimizer, embedding_spec.FTRLOptimizerSpec):
      extra_kwargs["multiply_linear_by_learning_rate"] = (
          stack_table_spec.optimizer.multiply_linear_by_learning_rate
      )
    updated_variables = optimizer_primitive.bind(
        row_pointer,
        embedding_id,
        sample_id,
        gain,
        num_minibatches,
        *flatten_variables,
        activation_gradient,
        *hyper_params,
        max_ids_per_partition=stack_table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=stack_table_spec.max_unique_ids_per_partition,
        computation_name=symbol_name,
        sharding_strategy=sharding_strategy,
        enable_minibatching=enable_minibatching,
        **extra_kwargs,
    )

    updated_embedding_variables[stacked_table_name] = jax.tree.unflatten(
        treedef,
        jax.tree.leaves(updated_variables),
    )

  return updated_embedding_variables


def _init_embedding_variables_shard(
    rng: jax.Array, tspec: embedding_spec.TableSpec, num_sparsecore: int
) -> EmbeddingVariables:
  """Initializes a shard of an embedding variable for a single SparseCore.

  Args:
    rng: The random number generator.
    tspec: The table spec for the embedding variable.
    num_sparsecore: The number of sparsecore to shard over.

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
      *jax.tree.map(lambda init: init(rng, shape), initializers)
  )


def _init_stacked_embedding_table_shard(
    rng: jax.Array,
    table_specs: List[embedding_spec.TableSpec],
    num_global_shards: int,
    num_sparsecore_per_device: int | None = None,
) -> EmbeddingVariables:
  """Initializes a shard of a stacked table.

  Args:
    rng: The random number generator.
    table_specs: A list of table specs that are in the same stack.
    num_global_shards: The number of global shards.
    num_sparsecore_per_device: The number of sparse cores per device. If `None`,
      it will be set to the number of sparse cores per device on the current
      host.

  Returns:
    A tuple of embedding table shards for all the tables in the stack.
  """
  num_sparsecore_per_device = _get_num_sc_per_device(num_sparsecore_per_device)
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
    sharding_axis: str | tuple[str, ...],
    num_sparsecore_per_device: int | None = None,
) -> EmbeddingVariables:
  """Initializes a stacked embedding table."""
  num_sparsecore_per_device = _get_num_sc_per_device(num_sparsecore_per_device)
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
                tiling=((8,),),
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
    num_sparsecore_per_device: int | None = None,
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
    num_sparsecore_per_device: The number of sparse cores per device. If `None`,
      it will use the number of sparse cores on the current host machine.
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
  num_sparsecore_per_device = _get_num_sc_per_device(num_sparsecore_per_device)
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
  for table_spec in jax.tree.leaves(table_specs):
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
    per_table_width = global_sharding.mesh.size * num_sparsecore_per_device
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
    global_device_count: int | None,
    num_sparsecore_per_device: int | None = None,
) -> embedding_spec_pb2.EmbeddingSpecProto:
  """Creates a StackedTableSpecProto from a list of FeatureSpec.

  This is used to create the proto for feature sets used for training. The proto
  captures relevant information for the features such that the
  training variables can be unsharded when being loaded from a checkpoint,
  for serving.

  Args:
    feature_specs: A Nested (e.g., list, dict etc.) of FeatureSpec.
    global_device_count: The number of devices in the system.
    num_sparsecore_per_device: The number of sparse cores per device. If `None`,
      it will be set to the number of sparse cores on the current host machine.

  Returns:
    An EmbeddingSpecProto.
  """
  num_sparsecore_per_device = _get_num_sc_per_device(num_sparsecore_per_device)

  stacked_table_specs: dict[str, embedding_spec_pb2.StackedTableSpecProto] = {}
  stack_to_table_specs: dict[
      str, dict[str, embedding_spec_pb2.TableSpecProto]
  ] = collections.defaultdict(dict)
  # Traverse the feature specs and create the StackedTableSpecProto.
  for feature in jax.tree.leaves(feature_specs):
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


def update_preprocessing_parameters(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    updated_params: SparseDenseMatmulInputStats,
    num_sc_per_device: int,
) -> None:
  """Updates the preprocessing parameters in the feature specs.

  All the features/tables must be stacked already.

  This function updates the max_ids_per_partition, max_unique_ids_per_partition
  and suggested_coo_buffer_size_per_device in the feature specs based on the
  updated
  parameters.

  Args:
    feature_specs: A Nested (e.g., list, dict etc.) of FeatureSpec.
    updated_params: The updated preprocessing parameters.
    num_sc_per_device: The number of sparsecores per device.

  Returns:
    None. The feature specs are updated in place.
  """

  def _update_feature_spec_limits(feature: embedding_spec.FeatureSpec) -> None:
    stacked_table_spec = feature.table_spec.stacked_table_spec
    stack_name = stacked_table_spec.stack_name

    max_ids_per_partition = updated_params.max_ids_per_partition.get(
        stack_name, stacked_table_spec.max_ids_per_partition
    )
    max_unique_ids_per_partition = (
        updated_params.max_unique_ids_per_partition.get(
            stack_name, stacked_table_spec.max_unique_ids_per_partition
        )
    )
    if stack_name in updated_params.required_buffer_size_per_sc:
      new_buffer_size_per_device = int(
          np.max(updated_params.required_buffer_size_per_sc[stack_name])
          * num_sc_per_device
      )
    else:
      new_buffer_size_per_device = (
          stacked_table_spec.suggested_coo_buffer_size_per_device
      )

    feature.table_spec.stacked_table_spec = dataclasses.replace(
        stacked_table_spec,
        max_ids_per_partition=int(np.max(max_ids_per_partition)),
        max_unique_ids_per_partition=int(np.max(max_unique_ids_per_partition)),
        suggested_coo_buffer_size_per_device=new_buffer_size_per_device,
    )

  jax.tree_util.tree_map(_update_feature_spec_limits, feature_specs)
