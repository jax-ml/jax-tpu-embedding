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
"""Specs for the SparseCore embedding layer."""

from __future__ import annotations

import abc
import collections
import dataclasses
import inspect
from typing import Callable, Sequence, TypeAlias

import jax
from jax import core
import jax.extend as jex
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_laprop
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_sgd

HyperParameterType: TypeAlias = Callable[[], jax.Array] | float

# Standard initializers are defined in jax.nn.initializers. See
# http://jax.readthedocs.io/en/latest/jax.nn.initializers.html
CallableTableInitializer: TypeAlias = jax.nn.initializers.Initializer

_OptimizerDefinition: TypeAlias = collections.namedtuple(
    "_OptimizerDefinition",
    ["primitive", "slot_variable_type", "default_initializers"],
)


SGDSlotVariables = collections.namedtuple("SGDSlotVariables", [])
AdagradSlotVariables = collections.namedtuple(
    "AdagradSlotVariables", ["accumulator"]
)

LaPropSlotVariables = collections.namedtuple(
    "LaPropSlotVariables", ["mu", "nu"]
)


# TODO(b/365975374): Create helper functions for generating OptimizerSpecs.
@dataclasses.dataclass(frozen=True, order=True)
class OptimizerSpec(metaclass=abc.ABCMeta):
  """Base class for the optimizer specs.

  This base class defines the interface for all the optimizer specs.

  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
  """

  def __init__(
      self,
      learning_rate: float | Callable[..., float | jax.Array],
  ):
    self.learning_rate = learning_rate

  def get_learning_rate(self, step: jax.Array | int | None = None) -> jax.Array:
    """Returns the learning rate for the optimizer."""
    learning_rate = self.learning_rate
    if callable(learning_rate):
      # Callable learning rate functions are expected to take a singular step
      # count argument, or no arguments.
      args = inspect.getfullargspec(learning_rate).args
      # If not a function, then it's an object instance with `self` as the first
      # argument.
      num_args = (
          len(args) if inspect.isfunction(learning_rate) else len(args) - 1
      )
      if num_args == 0:
        return jnp.array(learning_rate(), dtype=jnp.float32)
      elif num_args == 1:
        if step is not None:
          return jnp.array(learning_rate(step), dtype=jnp.float32)
        else:
          raise ValueError(
              "Specified learning rate callable {learning_rate} requires "
              "a `step` argument to be specified."
          )
      else:
        raise ValueError(
            "Learning rate callbacks should either take no parameters, or "
            "a single step count argument."
        )
    else:
      return jnp.array(learning_rate, dtype=jnp.float32)

  def get_hyperparameters(
      self, step: jax.Array | int | None = None
  ) -> tuple[jax.Array, ...]:
    """Returns the hyperparameters for the optimizer."""
    return (self.get_learning_rate(step),)

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    """Slot variables initializers for the optimizer.

    Derived classes should implement this method to return the initializers for
    the applicable slot variables, if any.

    Returns:
      A tuple of initializers for the slot variables.
    """
    return ()

  def slot_variables_count(self) -> int:
    """Returns the number of slot variables for the optimizer."""
    return len(self.slot_variables_initializers())

  @abc.abstractmethod
  def get_optimizer_primitive(self) -> jex.core.Primitive:
    """Derived classes should implement this method to return the xla primitive for the optimizer."""
    raise NotImplementedError

  @abc.abstractmethod
  def short_name(self) -> str:
    """Implement this method to return a short name for the optimizer.

    This short name will be used as part of the identifier for the variables
    being trained.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __hash__(self) -> int:
    raise NotImplementedError

  def __eq__(self, other: OptimizerSpec) -> bool:
    return self.__hash__() == other.__hash__()


class SGDOptimizerSpec(OptimizerSpec):
  """Spec for the Stochastic Gradient Descent (SGD) optimizer.

  An iterative optimization method that updates the weights of the embedding
  variables by taking a step in the direction of the gradient. The step size is
  controlled by the learning rate.
  SGD is a usually a default choice in training setup.

  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
  """

  def __init__(
      self,
      learning_rate=0.001,
  ):
    super().__init__(
        learning_rate=learning_rate,
    )

  def __hash__(self) -> int:
    return hash((self.learning_rate,))

  def short_name(self) -> str:
    return "sgd"

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    """SGD does not have any slot variables, hence this returns an empty tuple."""
    return SGDSlotVariables()

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    """Returns the optimizer primitive for the SGD optimizer."""
    return (
        sparse_dense_matmul_grad_with_sgd.tpu_sparse_dense_matmul_grad_with_sgd_primitive
    )


class AdagradOptimizerSpec(OptimizerSpec):
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

  def __init__(
      self,
      learning_rate=0.001,
      initial_accumulator_value: HyperParameterType = 0.1,
  ):
    super().__init__(
        learning_rate=learning_rate,
    )
    self.initial_accumulator_value = initial_accumulator_value

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    return AdagradSlotVariables(
        accumulator=jax.nn.initializers.constant(self.initial_accumulator_value)
    )

  def __hash__(self) -> int:
    return hash((
        self.learning_rate,
        self.initial_accumulator_value,
    ))

  def short_name(self) -> str:
    return "adagrad"

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive
    )


class LaPropOptimizerSpec(OptimizerSpec):
  """Spec for the LaProp optimizer.

  Laprop decouples momentum and adaptivity in the Adam-style methods, leading to
  improved speed and stability compare to Adam.
  https://arxiv.org/abs/2002.04839

  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the squared gradient to improve numerical stability.
    rms_clip_threshold: Clipping threshold for RMS.
    initial_slot_value: Initial value for the slot variables.
  """

  def __init__(
      self,
      learning_rate=0.001,
      b1: float = 0.9,
      b2: float = 0.95,
      eps: float = 1e-30,
      rms_clip_threshold: float = 1.0,
      initial_slot_value: float = 0.0,
  ):
    super().__init__(
        learning_rate=learning_rate,
    )
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.rms_clip_threshold = rms_clip_threshold
    self.initial_slot_value = initial_slot_value

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    return LaPropSlotVariables(
        mu=jax.nn.initializers.constant(self.initial_slot_value),
        nu=jax.nn.initializers.constant(self.initial_slot_value),
    )

  def get_decay_rate(self, step: jax.Array | int | None = None) -> jax.Array:
    """Returns the decay rate for the optimizer."""

    if step is None:
      return jnp.array(self.b2, dtype=jnp.float32)

    decay_rate = (
        self.b2
        * (1.0 - jnp.power(self.b2, step))
        / ((1.0 - jnp.power(self.b2, step+1.0)))
    )

    return jnp.array(decay_rate, dtype=jnp.float32)

  def get_hyperparameters(
      self, step: jax.Array | int | None = None
  ) -> tuple[jax.Array, ...]:
    """Returns the LaProp hyperparameters: (learning_rate, b1, decay_rate, eps)."""
    return (
        self.get_learning_rate(step),
        jnp.array(self.b1, dtype=jnp.float32),
        self.get_decay_rate(step),
        jnp.array(self.eps, dtype=jnp.float32),
    )

  def __hash__(self) -> int:
    return hash((
        self.learning_rate,
        self.b1,
        self.b2,
        self.eps,
        self.rms_clip_threshold,
        self.initial_slot_value,
    ))

  def short_name(self) -> str:
    return "laprop"

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_laprop.tpu_sparse_dense_matmul_grad_with_laprop_primitive
    )


@dataclasses.dataclass(eq=True, frozen=True, order=True)
class FeatureIdTransformation:
  """Transformation to apply to the input feature ids."""

  # Applicable in case of feature stacking, which means more than one feature
  # points to the same table. The `row_offset` specifies the number of samples
  # this feature is offset. For example, if feature A, B and C refer to the same
  # table and have input batch size 16, the row_offset for feature A is 0,
  # feature B is 16 and feature C is 32.
  row_offset: int = 0
  # Applicable in case of table stacking. When tables (embedding tables)
  # corresponding to multiple features are stacked then `col_offset` defines the
  # offset for the embedding of this feature in the stacked table.
  # For example Feature A has a table with embedding vocab 128 and Feature A
  # and feature B are stacked then the `col_offset` for feature B is 128
  col_offset: int = 0
  # Applicable in case of table stacking. When tables (embedding tables)
  # corresponding to multiple features are stacked then `col_shift` defines how
  # embedding table shards are shifted (aka rotated) on the device.
  col_shift: int = 0


@dataclasses.dataclass(eq=True, frozen=True, order=True)
class TableSettingInStack:
  """Placement of the table shard relative to the shard of the stack."""

  stack_name: str
  padded_vocab_size: int
  padded_embedding_dim: int
  row_offset_in_shard: int = 0  # Position of first row of this table in stack
  shard_rotation: int = 0


@dataclasses.dataclass(eq=True, unsafe_hash=True, order=True)
class TableSpec:
  """Spec for one embedding table."""

  name: str
  vocabulary_size: int
  embedding_dim: int
  initializer: CallableTableInitializer
  optimizer: OptimizerSpec
  combiner: str
  max_ids_per_partition: int = 256
  max_unique_ids_per_partition: int = 256
  # This points to the stacked table spec which this table belongs to.
  # If this is None, this table is the top-most table.
  stacked_table_spec: StackedTableSpec | None = dataclasses.field(
      default=None, compare=False
  )
  # How this table is placed in the stack. By default, it is the top-most table
  # and hence the setting is inferred to be in line with the table spec.
  _setting_in_stack: TableSettingInStack | None = dataclasses.field(
      default=None, compare=False
  )

  @property
  def setting_in_stack(self) -> TableSettingInStack:
    assert self._setting_in_stack is not None
    return self._setting_in_stack

  @setting_in_stack.setter
  def setting_in_stack(self, setting_in_stack: TableSettingInStack) -> None:
    self._setting_in_stack = setting_in_stack

  def __post_init__(
      self,
  ):
    # Populate the settings to default(no table stacking) if it is None.
    if self._setting_in_stack is None:
      self._setting_in_stack = TableSettingInStack(
          stack_name=self.name,
          padded_vocab_size=self.vocabulary_size,
          padded_embedding_dim=self.embedding_dim,
          row_offset_in_shard=0,
          shard_rotation=0,
      )


@dataclasses.dataclass(eq=True, unsafe_hash=True, order=True)
class FeatureSpec:
  """Spec for one embedding feature."""

  name: str
  table_spec: TableSpec
  # The shape of the input jax array.
  input_shape: Sequence[int]
  # The expected shape of the output activation jax array.
  output_shape: Sequence[int]
  # The transformation to apply to the input feature.
  _id_transformation: FeatureIdTransformation | None = dataclasses.field(
      default=None, compare=False
  )

  @property
  def id_transformation(self) -> FeatureIdTransformation:
    assert self._id_transformation is not None
    return self._id_transformation

  @id_transformation.setter
  def id_transformation(
      self, id_transformation: FeatureIdTransformation
  ) -> None:
    self._id_transformation = id_transformation


@dataclasses.dataclass(eq=True, frozen=True, order=True)
class StackedTableSpec:
  """Spec for a stacked table that is a combination of multiple tables."""

  stack_name: str
  stack_vocab_size: int
  stack_embedding_dim: int
  optimizer: OptimizerSpec
  combiner: str
  total_sample_count: int
  max_ids_per_partition: int = 256
  max_unique_ids_per_partition: int = 256
