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

from typing import Any, List, Mapping, Sequence, TypeAlias, TypeVar, Union

from absl import logging
import jax
import jax.extend as jex
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing_with_mini_batching_cc
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_csr_with_mini_batching
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad_with_mini_batching
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd_with_mini_batching
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import embedding_utils
import numpy as np
import tree


ArrayLike = jnp.ndarray | np.ndarray

T: TypeAlias = TypeVar("T")
Nested: TypeAlias = Union[T, Sequence[T], Mapping[str, T]]

# Necessary for all configurations and all operations.
CONFIG_MODE = "MINI_BATCH_MODE"

# Necessary for all configurations and all operations. Set to 1 in MODE_NONE.
CONFIG_SIZE = "MINI_BATCH_SIZE"

# Supported modes of preprocessing.
# The definitions must be aligned with the ones in
# input_preprocessing_with_mini_batching.h.
MODE_NONE = 1
MODE_VOCABULARY_DIMENSION = 2
MODE_SAMPLE_DIMENSION = 3
MODE_EXPERIMENTAL_FORCE_VOCABULARY_DIV = 200
MODE_EXPERIMENTAL_FORCE_VOCABULARY_MOD = 201


# SGD optimizer supporting mini-batching.
class SGDOptimizerSpec(embedding_spec.SGDOptimizerSpec):
  """Spec for the Stochastic Gradient Descent (SGD) optimizer.

  An iterative optimization method that updates the weights of the embedding
  variables by taking a step in the direction of the gradient. The step size is
  controlled by the learning rate.
  SGD is a usually a default choice in training setup.

  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
  """

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    """Returns the optimizer primitive for the SGD optimizer."""
    return (
        sparse_dense_matmul_grad_with_sgd_with_mini_batching.
        tpu_sparse_dense_matmul_grad_with_sgd_with_mini_batching_primitive
    )


# Adagrad optimizer supporting mini-batching.
class AdagradOptimizerSpec(embedding_spec.AdagradOptimizerSpec):
  """Spec for the Adagrad optimizer.

  An Adagrad optimizer is an adaptive optimizer that adjusts the learning rate
  for each embedding variable based on its past gradients. This helps in
  reducing the number of steps needed for convergence, especially for sparse
  data.
  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
    initial_accumulator_value: The initial value for the accumulator slot
      variable. This constant is used to initialize the accumulator slot
      variable.
  """

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_adagrad_with_mini_batching.
        tpu_sparse_dense_matmul_grad_with_adagrad_with_mini_batching_primitive
    )


def preprocess_sparse_dense_matmul_input(
    local_batch_size: int,
    indices: Sequence[Sequence[Sequence[int]]],
    values: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]] | Sequence[None],
    feature_specs: List[embedding_spec.FeatureSpec],
    mini_batching_config: Mapping[str, Any],
    local_device_count: int,
    global_device_count: int,
    static_buffer_size_multiplier: int = 0,
    num_sc_per_device: int = 4,
    sparsecore_register_width: int = 8,
    sharding_strategy: str = "MOD",
    has_leading_dimension: bool = False,
) -> tuple[
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    int,
    Mapping[str, np.ndarray],
]:
  """Preprocesses the input for sparse dense matmul.

  Args:
    local_batch_size: The number of samples in this batch. This is called a
      'local' batch because it is the combined batch size for all local devices.
    indices: The indices to values and weights. The first dimension is over the
      features. The second dimension is over the samples. All elements are
      expected to be 64bit integers.
    values: The values to process. The outer list is over the features. All
      elements are expected to be 32bit integers.
    weights: The weights associated with the values. The outer list is over
      the features. All elements are expected to be 32bit floats. If the weights
      are None for some or all features, the computation would assume the
      weights are 1.0.
    feature_specs: The feature specs. The order of this list must be aligned
      with the order of the indices, values, and weights lists.
    mini_batching_config: The mini-batching config. This is a dictionary
      containing the mini-batching mode and the mini-batch size. More
      configuration items will be added in the future.
    local_device_count: The number of local devices (chips). Typically
      `mesh.local_mesh.size`.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    static_buffer_size_multiplier: If larger than 0, this is the multiplier that
      is used to determine the size of the static buffers (lhs_embedding_ids,
      lhs_sample_ids and lhs_gains). The size of the buffer returned is
      static_buffer_size_multiplier x batch_size. If less than or equal to 0,
      the size of the buffer is determined based off of the
      max_ids_per_partition limits.
    num_sc_per_device: The number of sparse cores per device.
    sparsecore_register_width: The width of the sparsecore registers. This is
      hardware dependent and would change based on the underlying TPU
      generations.
    sharding_strategy: The sharding strategy (e.g., MOD)
    has_leading_dimension: If set to True, then the first dimension of the
      output will be the number of local devices. This is useful when using the
      output in jax.pmap. If set to False, then the first dimension of the
      output will be the number of local devices * the static buffer size. This
      is useful when using the output in jax.jit. In conclusion, Set it to True
      if using jax.pmap and set it to False if using jax.jit. Currently,
      only jax.pmap is supported.

  Returns:
    A tuple of four dictionaries mapping the stacked table names to the
    preprocessed inputs for the corresponding table. The four dictionaries are
    lhs_row_pointers, lhs_embedding_ids, lhs_sample_ids, and lhs_gains. The
    tuple also contains the resulting mini-batch size and a dictionary
    containing the statistics. The mini-batch size could be different from the
    the input mini-batch config, and should be used as the actual mini-batch
    size in later lookup and update calls. The statistics dictionary contains
    the max ids per table, the max unique ids per table, and the id drop
    counters per table.
  """

  return (
      input_preprocessing_with_mini_batching_cc.PreprocessSparseDenseMatmulInputWithBCOO(
          local_batch_size,
          indices,
          values,
          weights,
          feature_specs,
          mini_batching_config,
          local_device_count,
          global_device_count,
          static_buffer_size_multiplier,
          num_sc_per_device,
          sparsecore_register_width,
          embedding_utils.sharding_strategy_to_int(sharding_strategy),
          has_leading_dimension,
      )
  )


def flatten_features_and_weights(
    features: Mapping[str, ArrayLike],
    weights: Mapping[str, ArrayLike],
    flatten_feature_specs: Sequence[embedding_spec.FeatureSpec],
) -> tuple[
    int,
    Sequence[Sequence[Sequence[int]]],
    Sequence[Sequence[int]],
    Sequence[Sequence[float]],
]:
  """Transforms features and weights from numpy arrays to sparse BCOO format.

  This function is used to transform the features and weights from numpy arrays
  to sparse BCOO format. The sparse BCOO format is used to store the features
  and weights in a way that is efficient for sparsecore API. The returned tuple
  is suitable for the sparse dense matmul API. Note that this function is not
  performant and should only be used for testing purposes. The expectation is
  that the input features and weights should already be in the sparse BCOO
  format.

  Args:
    features: The features to process. The keys are the feature names.
    weights: The weights associated with the values. The keys are the feature
      names. Weights can be None for some or all features. In this case, the
      resulting flattened weights would also be None.
    flatten_feature_specs: The feature specs. The resulting flattened indices
      and values would be aligned with the feature order of this list.

  Returns:
    A tuple containing the local batch size, the flattened indices, the
    flattened values, and the flattened weights, to be fed to
    preprocess_sparse_dense_matmul_input.
  """
  local_batch_size = 0
  flatten_indices = []
  flatten_values = []
  flatten_weights = []

  assert flatten_feature_specs, "Feature specs must not be empty."
  assert features, "Features must not be empty."
  assert weights, "Weights must not be empty."

  assert len(features) == len(weights), (
      "Features and weights must have the same length."
  )
  assert len(features) == len(flatten_feature_specs), (
      "Features and feature specs must have the same length."
  )

  for feature_spec in flatten_feature_specs:
    feature_name = feature_spec.name
    current_feature = features[feature_name]
    current_weights = weights[feature_name]
    if local_batch_size == 0:
      local_batch_size = current_feature.shape[0]
      assert local_batch_size > 0, "Batch size must be greater than 0."
    else:
      assert (
          local_batch_size == current_feature.shape[0]
      ), "Batch size must be the same for all features."

    # Create the indices array to point to all values and weights.
    index_size = 0
    for row in current_feature:
      index_size += len(row)
    indices = np.empty((index_size, 2), dtype=np.int64)

    # Create the values array to store all the values (embedding ids).
    concatenated_values = np.empty(index_size, dtype=np.int32)

    # Optionally create the weights array to store all the weights.
    concatenated_weights = None
    if current_weights is not None:
      concatenated_weights = np.empty(index_size, dtype=np.float32)

    # Populate the indices, values, and optionally the weights arrays.
    index_cursor = 0
    for sample_index in range(local_batch_size):
      sample_length = len(current_feature[sample_index])
      for elem_index in range(sample_length):
        indices[index_cursor] = [sample_index, elem_index]
        concatenated_values[index_cursor] = current_feature[sample_index][
            elem_index
        ]
        if concatenated_weights is not None:
          concatenated_weights[index_cursor] = current_weights[sample_index][
              elem_index
          ]
        index_cursor += 1

    flatten_indices.append(indices)
    flatten_values.append(concatenated_values)
    flatten_weights.append(concatenated_weights)

  return (local_batch_size, flatten_indices, flatten_values, flatten_weights)


@jax.named_call
def tpu_sparse_dense_matmul(
    lhs_row_pointers: Mapping[str, jax.Array],
    lhs_embedding_ids: Mapping[str, jax.Array],
    lhs_sample_ids: Mapping[str, jax.Array],
    lhs_gains: Mapping[str, jax.Array],
    embedding_variables: Mapping[str, embedding.EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    mini_batching_config: Mapping[str, Any],
    global_device_count: int,
    sharding_strategy: str = "MOD",
) -> Nested[jax.Array]:
  """Computes the sparse dense matmul, or embedding lookup.

  Check the docstring of `tpu_sparse_dense_matmul` in embedding.py for
  more details.

  Args:
    lhs_row_pointers: The row pointers to process. The keys are the stacked
      table names.
    lhs_embedding_ids: The embedding ids to process. The keys are the stacked
      table names. Must have same structure as `lhs_row_pointers`.
    lhs_sample_ids: The sample ids to process. The keys are the stacked table
      names. Must have same structure as `lhs_row_pointers`.
    lhs_gains: The gains to process. The keys are the stacked table names. Must
      have same structure as `lhs_row_pointers`.
    embedding_variables: A tuple of embedding tables and slot variables. The
      first one is always the embedding table, the following ones are slot
      variables. The tree structure must be identical to the lhs_row_pointers.
    feature_specs: The input features for the current process.
    mini_batching_config: The mini-batching config. Note that the mini-batch
      size in the config should come from the result of each preprocess call.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    sharding_strategy: The sharding strategy (e.g., MOD)

  Returns:
    The activations structure with the same structure as feature_specs.

  Raises:
    ValueError: The input arrays and tuples are not of the expected structure or
      the sharding strategy is not supported.
  """
  assert lhs_row_pointers.keys() == lhs_embedding_ids.keys()
  assert lhs_row_pointers.keys() == lhs_gains.keys()
  assert lhs_row_pointers.keys() == lhs_sample_ids.keys()
  assert lhs_row_pointers.keys() == embedding_variables.keys()

  stacked_table_specs = embedding.get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  sharding_strategy = embedding_utils.sharding_strategy_to_enum(
      sharding_strategy
  )

  activations = {}
  for stacked_table_name in stacked_table_specs:
    row_pointer = lhs_row_pointers[stacked_table_name]
    embedding_id = lhs_embedding_ids[stacked_table_name]
    sample_id = lhs_sample_ids[stacked_table_name]
    gain = lhs_gains[stacked_table_name]
    embedding_variable = embedding_variables[stacked_table_name]
    stacked_table = stacked_table_specs[stacked_table_name]

    if mini_batching_config[CONFIG_MODE] in (
        MODE_VOCABULARY_DIMENSION,
        MODE_EXPERIMENTAL_FORCE_VOCABULARY_DIV,
        MODE_EXPERIMENTAL_FORCE_VOCABULARY_MOD,
    ):
      activations[stacked_table.stack_name] = (
          sparse_dense_matmul_csr_with_mini_batching.tpu_sparse_dense_matmul_csr_with_mini_batching_primitive.bind(
              row_pointer,
              embedding_id,
              sample_id,
              gain,
              mini_batching_config[CONFIG_SIZE],
              embedding_variable[0],  # [0] is the embedding table
              device_batch_size=(stacked_table.total_sample_count
                                 // global_device_count),
              max_ids_per_partition=stacked_table.max_ids_per_partition,
              max_unique_ids_per_partition=stacked_table.max_unique_ids_per_partition,
              sharding_strategy=sharding_strategy,
          )
      )
    else:
      raise ValueError(
          f"Unsupported mini-batching mode: {mini_batching_config[CONFIG_MODE]}"
      )

  return embedding_utils.unstack_embedding_activations(
      activations, feature_specs, global_device_count
  )


@jax.named_call
def tpu_sparse_dense_matmul_grad(
    activation_gradients: Nested[jax.Array],
    lhs_row_pointers: Mapping[str, jax.Array],
    lhs_embedding_ids: Mapping[str, jax.Array],
    lhs_sample_ids: Mapping[str, jax.Array],
    lhs_gains: Mapping[str, jax.Array],
    embedding_variables: Mapping[str, embedding.EmbeddingVariables],
    feature_specs: Nested[embedding_spec.FeatureSpec],
    mini_batching_config: Mapping[str, Any],
    sharding_strategy: str = "MOD",
    label: str = "",
) -> Mapping[str, embedding.EmbeddingVariables]:
  """Computes the updated embedding variables based on the activation gradients.

  Check the docstring of `tpu_sparse_dense_matmul_grad` in embedding.py for
  more details.

  Args:
    activation_gradients: The activation gradients.
    lhs_row_pointers: The row pointers to process. The keys are the stacked
      table names.
    lhs_embedding_ids: The embedding ids to process. The keys are the stacked
      table names. Must have same structure as `lhs_row_pointers`.
    lhs_sample_ids: The sample ids to process. The keys are the stacked table
      names. Must have same structure as `lhs_row_pointers`.
    lhs_gains: The gains to process. The keys are the stacked table names. Must
      have same structure as `lhs_row_pointers`.
    embedding_variables: A tuple of embedding tables and slot variables. The
      first one is always the embedding table, the following ones are slot
      variables. The tree structure must be identical to the lhs_row_pointers.
    feature_specs: The input features for the current process.
    mini_batching_config: The mini-batching config. Note that the mini-batch
      size in the config should come from the result of each preprocess call.
    sharding_strategy: The sharding strategy (e.g., MOD)
    label: The label for the optimizer computation.

  Returns:
    The updated activation embedding variables.
  """

  # Verify the input structures and lengths.
  assert lhs_row_pointers.keys() == lhs_embedding_ids.keys()
  assert lhs_row_pointers.keys() == lhs_gains.keys()
  assert lhs_row_pointers.keys() == lhs_sample_ids.keys()
  assert lhs_row_pointers.keys() == embedding_variables.keys()
  # Activations match the feature specs structure
  tree.assert_same_structure(feature_specs, activation_gradients)

  stacked_table_specs = embedding.get_stacked_table_specs(feature_specs)
  assert lhs_row_pointers.keys() == stacked_table_specs.keys()

  gradients = embedding_utils.stack_embedding_gradients(
      activation_gradients, feature_specs
  )
  assert lhs_row_pointers.keys() == gradients.keys()

  sharding_strategy = embedding_utils.sharding_strategy_to_enum(
      sharding_strategy
  )

  updated_embedding_variables = {}
  for stacked_table_name in stacked_table_specs:
    row_pointer = lhs_row_pointers[stacked_table_name]
    embedding_id = lhs_embedding_ids[stacked_table_name]
    sample_id = lhs_sample_ids[stacked_table_name]
    gain = lhs_gains[stacked_table_name]
    embedding_variable = embedding_variables[stacked_table_name]
    activation_gradient = gradients[stacked_table_name]
    stack_table_spec = stacked_table_specs[stacked_table_name]
    learning_rate = stack_table_spec.optimizer.get_learning_rate()
    hyper_params = [learning_rate]
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
        mini_batching_config[CONFIG_SIZE],
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
