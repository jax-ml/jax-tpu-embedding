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
"""FDO Manager class."""

import dataclasses
import functools
import logging
from typing import Mapping

import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


def _update_feature_spec_limits(
    f: embedding_spec.FeatureSpec,
    max_ids_per_partition: Mapping[str, np.ndarray],
    max_unique_ids_per_partition: Mapping[str, np.ndarray],
    max_required_buffer_size_per_sc: Mapping[str, np.ndarray],
    current_buffer_size: Mapping[str, np.ndarray],
    num_sc_per_device: int,
):
  """Updates the feature spec limits based on the provided mappings.

  Args:
    f: The feature spec.
    max_ids_per_partition: A mapping of table names to max_ids_per_partition.
    max_unique_ids_per_partition: A mapping of table names to
      max_unique_ids_per_partition.
    max_required_buffer_size_per_sc: A mapping of table names to
      max_required_buffer_size.
    current_buffer_size: A mapping of table names to current buffer size.
    num_sc_per_device: The number of sparse cores per device.
  """
  stacked_table_spec = f.table_spec.stacked_table_spec
  if stacked_table_spec is None:
    raise ValueError(f'stacked_table_spec is None for feature {f.name}')
  stack_name = stacked_table_spec.stack_name
  logging.info('Maybe updating limits for table: %s', stack_name)
  max_id = max_ids_per_partition.get(
      stack_name,
      stacked_table_spec.max_ids_per_partition,
  )
  max_id = int(np.max(max_id))
  max_unique_id = max_unique_ids_per_partition.get(
      stack_name,
      stacked_table_spec.max_unique_ids_per_partition,
  )
  max_unique_id = int(np.max(max_unique_id))
  max_required_buffer_size = max_required_buffer_size_per_sc.get(
      stack_name,
      stacked_table_spec.suggested_coo_buffer_size,
  )
  new_buffer_size_per_device = None
  if max_required_buffer_size is not None:
    # max_required_buffer_size is valid only if the table has FDO stats or the
    # user provides a suggested size. If both of these conditions are false, the
    # `max_required_buffer_size` will be None (we will eventually use
    # theoretical max in the CC library).
    new_buffer_size_per_device = int(
        max_required_buffer_size * num_sc_per_device
    )
  logging.info(
      'Updating limits for table %s. Prev (max_id, max_unique_id,'
      ' required_size): (%s, %s, %s) -> New: (%s, %s, %s)',
      stack_name,
      stacked_table_spec.max_ids_per_partition,
      stacked_table_spec.max_unique_ids_per_partition,
      current_buffer_size.get(stack_name),
      max_id,
      max_unique_id,
      new_buffer_size_per_device,
  )
  new_stacked_spec = dataclasses.replace(
      stacked_table_spec,
      max_ids_per_partition=max_id,
      max_unique_ids_per_partition=max_unique_id,
      suggested_coo_buffer_size=new_buffer_size_per_device,
  )
  f.table_spec.stacked_table_spec = new_stacked_spec


def maybe_perform_fdo_update(
    max_ids_per_partition: Mapping[str, np.ndarray],
    max_unique_ids_per_partition: Mapping[str, np.ndarray],
    max_required_buffer_size_per_sc: Mapping[str, np.ndarray],
    feature_specs: embedding.Nested[embedding_spec.FeatureSpec],
    preprocessed_inputs: embedding.SparseDenseMatmulInput,
    num_sc_per_device: int,
):
  """Updates feature specs based on FDO data.

  Args:
    max_ids_per_partition: A mapping of table names to max_ids_per_partition.
    max_unique_ids_per_partition: A mapping of table names to
      max_unique_ids_per_partition.
    max_required_buffer_size_per_sc: A mapping of table names to
      max_required_buffer_size.
    feature_specs: The current feature specs.
    preprocessed_inputs: The current preprocessed inputs. Only used for
      calculating buffer size.
    num_sc_per_device: The number of sparse cores per device.
  """
  current_buffer_size = jax.tree.map(
      lambda x: jnp.shape(x)[0], preprocessed_inputs.lhs_embedding_ids
  )
  maybe_update_limits = functools.partial(
      _update_feature_spec_limits,
      max_ids_per_partition=max_ids_per_partition,
      max_unique_ids_per_partition=max_unique_ids_per_partition,
      max_required_buffer_size_per_sc=max_required_buffer_size_per_sc,
      current_buffer_size=current_buffer_size,
      num_sc_per_device=num_sc_per_device,
  )
  jax.tree_util.tree_map(
      maybe_update_limits,
      feature_specs,
  )
