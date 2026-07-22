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
"""Proto interoperability utilities for SparseCore embedding tables and optimizers.

This file contains functions to convert between Python embedding specification
classes (such as feature and optimizer specifications) and their corresponding
protocol buffer definitions for serving and checkpointing.
"""

# Required for PEP 604 types (|) with Sphinx mocked imports.
from __future__ import annotations

import collections
from typing import Any, Mapping, Sequence, TypeAlias, TypeVar

import jax
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils

T: TypeAlias = TypeVar("T")
Nested: TypeAlias = T | Sequence[T] | Mapping[str, T]


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


_OPTIMIZER_TYPE_TO_CLASS: dict[
    embedding_spec_pb2.OptimizerSpecProto.OptimizerType,
    type[embedding_spec.OptimizerSpec],
] = {
    embedding_spec_pb2.OptimizerSpecProto.SGD: embedding_spec.SGDOptimizerSpec,
    embedding_spec_pb2.OptimizerSpecProto.ADAGRAD: (
        embedding_spec.AdagradOptimizerSpec
    ),
    embedding_spec_pb2.OptimizerSpecProto.ADAM: (
        embedding_spec.AdamOptimizerSpec
    ),
    embedding_spec_pb2.OptimizerSpecProto.FTRL: (
        embedding_spec.FTRLOptimizerSpec
    ),
    embedding_spec_pb2.OptimizerSpecProto.F2A: embedding_spec.F2AOptimizerSpec,
    embedding_spec_pb2.OptimizerSpecProto.ADAGRAD_MOMENTUM: (
        embedding_spec.AdagradMomentumOptimizerSpec
    ),
}

_OPTIMIZER_CLASS_TO_TYPE: dict[
    type[embedding_spec.OptimizerSpec],
    embedding_spec_pb2.OptimizerSpecProto.OptimizerType,
] = {cls: opt_type for opt_type, cls in _OPTIMIZER_TYPE_TO_CLASS.items()}


def get_optimizer_class(
    optimizer_type: embedding_spec_pb2.OptimizerSpecProto.OptimizerType,
) -> type[embedding_spec.OptimizerSpec] | None:
  """Gets the OptimizerSpec class corresponding to the given proto optimizer type.

  Args:
    optimizer_type: The OptimizerType enum value from OptimizerSpecProto.

  Returns:
    The corresponding OptimizerSpec subclass, or None if not recognized.
  """
  return _OPTIMIZER_TYPE_TO_CLASS.get(optimizer_type)


def get_optimizer_type(
    optimizer_class: type[embedding_spec.OptimizerSpec],
) -> embedding_spec_pb2.OptimizerSpecProto.OptimizerType:
  """Gets the proto optimizer type corresponding to an OptimizerSpec class.

  Args:
    optimizer_class: The class of the OptimizerSpec.

  Returns:
    The corresponding OptimizerType enum value, or CUSTOM if not recognized.
  """
  return _OPTIMIZER_CLASS_TO_TYPE.get(
      optimizer_class, embedding_spec_pb2.OptimizerSpecProto.CUSTOM
  )


def _extract_learning_rate_tag(
    learning_rate: embedding_spec.LearningRate,
) -> str | None:
  """Extracts a string tag from a callable learning rate, or None if static."""
  if not callable(learning_rate) or isinstance(learning_rate, (int, float)):
    return None

  if hasattr(learning_rate, "tag") and getattr(learning_rate, "tag"):
    return str(getattr(learning_rate, "tag"))

  name = getattr(
      learning_rate,
      "__qualname__",
      getattr(learning_rate, "__name__", ""),
  )
  if name and name != "<lambda>":
    module = getattr(learning_rate, "__module__", "")
    return f"{module}.{name}" if module else name

  return f"{type(learning_rate).__name__}_{id(learning_rate)}"


def optimizer_spec_to_proto(
    opt: embedding_spec.OptimizerSpec | None,
) -> embedding_spec_pb2.OptimizerSpecProto | None:
  """Converts an OptimizerSpec instance to OptimizerSpecProto."""
  if opt is None:
    return None

  lr_kwargs = {}
  lr_tag = _extract_learning_rate_tag(opt.learning_rate)
  if lr_tag:
    lr_kwargs["learning_rate_tag"] = lr_tag
  elif isinstance(opt.learning_rate, (int, float)):
    lr_kwargs["learning_rate"] = float(opt.learning_rate)

  opt_type: Any = get_optimizer_type(type(opt))
  float_params = (
      opt.get_float_params()
      if opt_type != embedding_spec_pb2.OptimizerSpecProto.CUSTOM
      else {}
  )
  bool_params = (
      opt.get_bool_params()
      if opt_type != embedding_spec_pb2.OptimizerSpecProto.CUSTOM
      else {}
  )

  return embedding_spec_pb2.OptimizerSpecProto(
      type=opt_type,
      float_params=float_params,
      bool_params=bool_params,
      **lr_kwargs,
  )


def proto_to_optimizer_spec(
    proto: embedding_spec_pb2.OptimizerSpecProto | None,
) -> embedding_spec.OptimizerSpec:
  """Converts an OptimizerSpecProto to an OptimizerSpec object.

  Args:
    proto: The optimizer spec proto to convert, or None.

  Returns:
    The corresponding OptimizerSpec instance. If proto is None or UNSPECIFIED,
    returns an SGDOptimizerSpec.
  """
  if proto is None:
    return embedding_spec.SGDOptimizerSpec(learning_rate=0.0)

  if proto.WhichOneof("learning_rate_spec") == "learning_rate_tag":
    lr: embedding_spec.LearningRate = embedding_spec.CallablePlaceholder(
        proto.learning_rate_tag
    )
  else:
    lr = proto.learning_rate

  cls = get_optimizer_class(proto.type)
  if cls is None:
    return embedding_spec.SGDOptimizerSpec(learning_rate=lr)

  return cls.from_float_and_bool_params(
      dict(proto.float_params), dict(proto.bool_params), learning_rate=lr
  )


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
  if global_device_count is None:
    global_device_count = jax.device_count()
  num_sc_per_dev = _get_num_sc_per_device(num_sparsecore_per_device)

  stacked_table_specs: dict[str, embedding_spec_pb2.StackedTableSpecProto] = {}
  stack_to_table_specs: dict[
      str, dict[str, embedding_spec_pb2.TableSpecProto]
  ] = collections.defaultdict(dict)
  # Traverse the feature specs and create the StackedTableSpecProto.
  for feature in jax.tree.leaves(feature_specs):
    stacked_spec = feature.table_spec.stacked_table_spec
    setting = feature.table_spec.setting_in_stack
    current_stack_name = stacked_spec.stack_name
    current_table_name = feature.table_spec.name
    if current_stack_name not in stacked_table_specs:
      stacked_table_specs[current_stack_name] = (
          embedding_spec_pb2.StackedTableSpecProto(
              stack_name=current_stack_name,
              stack_vocab_size=stacked_spec.stack_vocab_size,
              stack_embedding_dim=stacked_spec.stack_embedding_dim,
              total_sample_count=stacked_spec.total_sample_count,
              max_ids_per_partition=stacked_spec.max_ids_per_partition,
              num_sparsecores=(num_sc_per_dev * global_device_count),
              max_unique_ids_per_partition=stacked_spec.max_unique_ids_per_partition,
          )
      )
    if current_table_name not in stack_to_table_specs[current_stack_name]:
      stack_to_table_specs[current_stack_name][current_table_name] = (
          embedding_spec_pb2.TableSpecProto(
              table_name=current_table_name,
              vocab_size=feature.table_spec.vocabulary_size,
              embedding_dim=feature.table_spec.embedding_dim,
              padded_vocab_size=setting.padded_vocab_size,
              padded_embedding_dim=setting.padded_embedding_dim,
              row_offset_in_shard=setting.row_offset_in_shard,
              shard_rotation=setting.shard_rotation,
              combiner=feature.table_spec.combiner,
              optimizer=optimizer_spec_to_proto(feature.table_spec.optimizer),
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
