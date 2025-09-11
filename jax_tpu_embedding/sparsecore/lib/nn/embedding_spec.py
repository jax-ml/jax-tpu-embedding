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

from flax import struct
import jax
import jax.extend as jex
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad_momentum
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adam
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_ftrl
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
AdamSlotVariables = collections.namedtuple(
    "AdamSlotVariables", ["momentum", "velocity"]
)

AdagradMomentumSlotVariables = collections.namedtuple(
    "AdagradMomentumSlotVariables", ["accumulator", "momentum"]
)

FTRLSlotVariables = collections.namedtuple(
    "FtrlSlotVariables", ["accumulator", "linear"]
)
LaPropSlotVariables = collections.namedtuple(
    "LaPropSlotVariables", ["mu", "nu"]
)


# TODO(b/365975374): Create helper functions for generating OptimizerSpecs.
@dataclasses.dataclass(frozen=True)
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


class AdamOptimizerSpec(OptimizerSpec):
  """Spec for the Adam optimizer.

  Adam optimization is a stochastic gradient descent method that is based on
  adaptive estimation of first-order and second-order moments.

  According to
  [Kingma et al., 2014](http://arxiv.org/abs/1412.6980), the method is
  "*computationally efficient, has little memory requirement, invariant to
  diagonal rescaling of gradients, and is well suited for problems that are
  large in terms of data/parameters*".

  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
    beta_1: A float value or a constant float tensor, or a callable that takes
      no arguments and returns the actual value to use. The exponential decay
      rate for the 1st moment estimates. Defaults to `0.9`.
    beta_2: A float value or a constant float tensor, or a callable that takes
      no arguments and returns the actual value to use. The exponential decay
      rate for the 2nd moment estimates. Defaults to `0.999`.
    epsilon: A small constant for numerical stability.  Defaults to `1e-8`.
  """

  def __init__(
      self,
      learning_rate: (
          float | jax.Array | Callable[..., float | jax.Array]
      ) = 0.001,
      beta_1: float | jax.Array = 0.9,
      beta_2: float | jax.Array = 0.999,
      epsilon: float | jax.Array = 1e-8,
  ):
    super().__init__(
        learning_rate=learning_rate,
    )
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    return AdamSlotVariables(
        momentum=jax.nn.initializers.constant(0.0),
        velocity=jax.nn.initializers.constant(0.0),
    )

  def get_hyperparameters(
      self, step: jax.Array | int | None = None
  ) -> tuple[jax.Array, ...]:
    """Compute the bias-corrected Adam hyperparameters.

    Here we use the bias-corrected parameters from section 2.1 of the
    original paper:
      alpha_t = alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t)
      epsilon_hat = epsilon * sqrt(1 + beta_2^t)

    Args:
      step: The step count for the optimizer.

    Returns:
      A tuple of the scaled hyperparameters (alpha_t, beta_1, beta_2,
      epsilon_hat).
    """
    if step is None:
      step = 0

    beta_1 = jnp.asarray(self.beta_1, jnp.float32)
    beta_2 = jnp.asarray(self.beta_2, jnp.float32)
    beta_1_t = jnp.power(beta_1, step + 1)
    beta_2_t = jnp.power(beta_2, step + 1)
    c_2 = jnp.sqrt(1.0 - beta_2_t)
    alpha_t = self.get_learning_rate(step) * c_2 / (1.0 - beta_1_t)
    epsilon_hat = jnp.asarray(self.epsilon, jnp.float32) * c_2
    return (
        alpha_t,
        beta_1,
        beta_2,
        epsilon_hat,
    )

  def __hash__(self) -> int:
    return hash((
        self.learning_rate,
        self.beta_1,
        self.beta_2,
        self.epsilon,
    ))

  def short_name(self) -> str:
    return "adam"

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_adam.tpu_sparse_dense_matmul_grad_with_adam_primitive
    )


class AdagradMomentumOptimizerSpec(OptimizerSpec):
  """Spec for the Adagrad with Momentum optimizer.

  An Adagrad with Momentum optimizer is an adaptive optimizer that combines
  the benefits of both Adagrad and Momentum. It adjusts the learning rate
  for each embedding variable based on its past gradients, while also
  incorporating momentum to accelerate convergence.
  Attributes:
    learning_rate: The learning rate for the training variables or embeddings.
    momentum: The momentum parameter.
    initial_accumulator_value: The initial value for the accumulator slot
      variable.
    initial_momentum_value: The initial value for the momentum slot variable.
    beta2: The exponential decay rate for the 2nd moment estimates.
    epsilon: A small constant for numerical stability.
    exponent: The exponent for the gradient squared accumulator.
    use_nesterov: Whether to use Nesterov momentum.
  """

  def __init__(
      self,
      learning_rate=0.001,
      momentum: float = 0.9,
      beta2: float = 1.0,
      epsilon: float = 1e-10,
      exponent: float = 2.0,
      use_nesterov: bool = False,
      initial_accumulator_value: float = 0.1,
      initial_momentum_value: float = 0.0,
  ):
    super().__init__(
        learning_rate=learning_rate,
    )
    self.momentum = momentum
    self.beta2 = beta2
    self.epsilon = epsilon
    self.exponent = exponent
    self.use_nesterov = use_nesterov
    self.initial_accumulator_value = initial_accumulator_value
    self.initial_momentum_value = initial_momentum_value

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    return AdagradMomentumSlotVariables(
        accumulator=jax.nn.initializers.constant(
            self.initial_accumulator_value
        ),
        momentum=jax.nn.initializers.constant(self.initial_momentum_value),
    )

  def get_hyperparameters(self, step=None) -> tuple[jax.Array, ...]:
    return (
        self.get_learning_rate(step),  # λ
        jnp.array(self.momentum, dtype=jnp.float32),  # k (β₁)
        jnp.array(self.beta2, dtype=jnp.float32),  # β₂
        jnp.array(self.epsilon, dtype=jnp.float32),  # ε
        jnp.array(self.exponent, dtype=jnp.float32),  # k-exp
        jnp.array(self.use_nesterov, dtype=jnp.bool_),
    )

  def __hash__(self):
    return hash((
        self.learning_rate,
        self.momentum,
        self.beta2,
        self.epsilon,
        self.exponent,
        self.use_nesterov,
        self.initial_accumulator_value,
        self.initial_momentum_value,
    ))

  def short_name(self) -> str:
    return "adagrad_momentum"

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_adagrad_momentum.tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive
    )


class FTRLOptimizerSpec(OptimizerSpec):
  """Spec for the FTRL optimizer.

  Follow The Regularized Leader (FTRL) is an optimization algorithm developed
  at Google for click-through rate prediction.

  Attributes:
    learning_rate: The learning rate.
    learning_rate_power: A float value, typically -0.5.
    l1_regularization_strength: A float value, must be greater than or equal to
      0.
    l2_regularization_strength: A float value, must be greater than or equal to
      0.
    beta: A float value.
    initial_accumulator_value: Initial value for the accumulator slot.
    initial_linear_value: Initial value for the linear slot.
    multiply_linear_by_learning_rate: A bool value, if True, multiply the linear
      slot by the learning rate.
  """

  def __init__(
      self,
      learning_rate: (
          float | jax.Array | Callable[..., float | jax.Array]
      ) = 0.01,
      learning_rate_power: float = -0.5,
      l1_regularization_strength: float = 0.0,
      l2_regularization_strength: float = 0.0,
      beta: float = 0.0,
      initial_accumulator_value: float = 0.1,
      initial_linear_value: float = 0.0,
      multiply_linear_by_learning_rate: bool = False,
  ):
    super().__init__(learning_rate=learning_rate)
    self.learning_rate_power = learning_rate_power
    self.l1_regularization_strength = l1_regularization_strength
    self.l2_regularization_strength = l2_regularization_strength
    self.beta = beta
    self.initial_accumulator_value = initial_accumulator_value
    self.initial_linear_value = initial_linear_value
    self.multiply_linear_by_learning_rate = multiply_linear_by_learning_rate

  def slot_variables_initializers(self) -> tuple[CallableTableInitializer, ...]:
    return FTRLSlotVariables(
        accumulator=jax.nn.initializers.constant(
            self.initial_accumulator_value
        ),
        linear=jax.nn.initializers.constant(self.initial_linear_value),
    )

  def get_hyperparameters(
      self, step: jax.Array | int | None = None
  ) -> tuple[jax.Array, ...]:
    """Returns the FTRL hyperparameters."""
    return (
        self.get_learning_rate(step),
        jnp.array(self.learning_rate_power, dtype=jnp.float32),
        jnp.array(self.l1_regularization_strength, dtype=jnp.float32),
        jnp.array(self.l2_regularization_strength, dtype=jnp.float32),
        jnp.array(self.beta, dtype=jnp.float32),
    )

  def __hash__(self) -> int:
    return hash((
        self.learning_rate,
        self.learning_rate_power,
        self.l1_regularization_strength,
        self.l2_regularization_strength,
        self.beta,
        self.initial_accumulator_value,
        self.initial_linear_value,
        self.multiply_linear_by_learning_rate,
    ))

  def short_name(self) -> str:
    return "ftrl"

  def get_optimizer_primitive(self) -> jex.core.Primitive:
    return (
        sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive
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
        / ((1.0 - jnp.power(self.b2, step + 1.0)))
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


@dataclasses.dataclass(eq=True, frozen=True)
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


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class TableSettingInStack:
  """Placement of the table shard relative to the shard of the stack."""

  stack_name: str
  padded_vocab_size: int
  padded_embedding_dim: int
  row_offset_in_shard: int = 0  # Position of first row of this table in stack
  shard_rotation: int = 0


@dataclasses.dataclass(eq=True, unsafe_hash=True, kw_only=True)
class TableSpec:
  """Specifies one embedding table.

  Attributes:
    name: Name of the table.
    vocabulary_size: The total number of unique embedding IDs. This is the
      number of rows in the embedding table.
    embedding_dim: The number of columns in the embedding table.
    initializer: An initializer for the embedding table. See
      [jax.nn.initializers](https://docs.jax.dev/en/latest/jax.nn.initializers.html)
      for more details.
    optimizer: An optimizer for the embedding table.
    combiner: The aggregation function to compute activations for each sample.
      For example, sum or mean.
    max_ids_per_partition: The maximum number of embedding IDs that can be
      packed into a single partition.
    max_unique_ids_per_partition: The maximum number of unique embedding IDs
      that can be packed into a single partition.
    suggested_coo_buffer_size_per_device: The minimum size of the input buffer
      that the preprocessing should try to create.
    quantization_config: Quantization config (min, max, num_buckets) which
      represent the float range and number of discrete integer buckets to use
      for quantization.
  """

  name: str
  vocabulary_size: int
  embedding_dim: int
  initializer: CallableTableInitializer
  optimizer: OptimizerSpec
  combiner: str
  max_ids_per_partition: int = 256
  max_unique_ids_per_partition: int = 256
  # The minimum size of the input buffer that the preprocessing should try to
  # create.
  suggested_coo_buffer_size_per_device: int | None = None
  # Quantization config (min, max, num_buckets) which represent the float
  # range and number of discrete integer buckets to use for quantization.
  quantization_config: QuantizationConfig | None = None

  # This points to the stacked table spec which this table belongs to.
  # If this is None, this table is the top-most table.
  _stacked_table_spec: StackedTableSpec | None = dataclasses.field(
      default=None, compare=False
  )
  # How this table is placed in the stack. By default, it is the top-most table
  # and hence the setting is inferred to be in line with the table spec.
  _setting_in_stack: TableSettingInStack | None = dataclasses.field(
      default=None, compare=False
  )

  @property
  def setting_in_stack(self) -> TableSettingInStack:
    """Returns the setting of this table in the stack."""
    assert self._setting_in_stack is not None
    return self._setting_in_stack

  @property
  def stacked_table_spec(self) -> StackedTableSpec:
    """Returns the stacked table spec which this table belongs to."""
    assert (
        self._stacked_table_spec is not None
    ), f"Table {self.name} is not stacked."
    return self._stacked_table_spec

  @setting_in_stack.setter
  def setting_in_stack(self, setting_in_stack: TableSettingInStack) -> None:
    self._setting_in_stack = setting_in_stack

  @stacked_table_spec.setter
  def stacked_table_spec(self, stacked_table_spec: StackedTableSpec) -> None:
    self._stacked_table_spec = stacked_table_spec

  def is_stacked(self) -> bool:
    return self._stacked_table_spec is not None

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


@dataclasses.dataclass(eq=True, unsafe_hash=True, kw_only=True)
class FeatureSpec:
  """Specification for one embedding feature.

  Attributes:
    name: Name of the feature.
    table_spec: The table spec for the feature.
    input_shape: The shape of the input jax array, this is [batch_size,
      feature_valency]. The second element can be omitted for ragged input.
    output_shape: The expected shape of the output activation jax array, this is
      [batch_size, embedding_dim]
  """

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
    """Returns the transformation to apply to the input feature ids."""
    assert self._id_transformation is not None, self.name
    return self._id_transformation

  @id_transformation.setter
  def id_transformation(
      self, id_transformation: FeatureIdTransformation
  ) -> None:
    self._id_transformation = id_transformation


@struct.dataclass(eq=True, frozen=True, kw_only=True)
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
  # The minimum size of the input buffer that the preprocessing should try to
  # create.
  suggested_coo_buffer_size_per_device: int | None = None
  # Quantization config (min, max, num_buckets) which represent the float
  # range and number of discrete integer buckets to use for quantization.
  quantization_config: QuantizationConfig | None = None


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QuantizationConfig:
  """Per-table quantization parameters (None means disabled)."""

  min_value: float
  max_value: float
  num_buckets: int

  def __post_init__(self):
    if self.num_buckets < 2:
      raise ValueError("num_buckets must be ≥ 2.")
    if self.min_value >= self.max_value:
      raise ValueError("min_value must be < max_value.")

  def as_tuple(self) -> tuple[float, float, int]:
    return (self.min_value, self.max_value, self.num_buckets)
