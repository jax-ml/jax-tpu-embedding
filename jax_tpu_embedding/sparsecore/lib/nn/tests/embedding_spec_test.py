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
"""Tests for embedding spec."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_adagrad_momentum
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_ftrl
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_laprop
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from optax import schedules


class OptimizerSpecTest(absltest.TestCase):

  def test_compare_optimizers_basic(self):
    self.assertEqual(
        embedding_spec.SGDOptimizerSpec(learning_rate=0.1),
        embedding_spec.SGDOptimizerSpec(learning_rate=0.1),
    )
    self.assertNotEqual(
        embedding_spec.SGDOptimizerSpec(learning_rate=0.1),
        embedding_spec.AdagradOptimizerSpec(learning_rate=0.1),
    )
    self.assertNotEqual(
        embedding_spec.AdagradOptimizerSpec(),
        embedding_spec.AdagradOptimizerSpec(learning_rate=0.1),
    )

  def test_compare_adagrad(self):
    self.assertEqual(
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.1,
            initial_accumulator_value=0.1,
        ),
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.1,
            initial_accumulator_value=0.1,
        ),
    )
    self.assertNotEqual(
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.1,
            initial_accumulator_value=0.1,
        ),
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.1,
            initial_accumulator_value=0.2,
        ),
    )
    self.assertNotEqual(
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.1,
            initial_accumulator_value=0.1,
        ),
        embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.2,
            initial_accumulator_value=0.1,
        ),
    )
    op = embedding_spec.AdagradOptimizerSpec(
        learning_rate=0.1,
        initial_accumulator_value=0.1,
    )
    self.assertEqual(op.learning_rate, 0.1)
    self.assertEqual(op.initial_accumulator_value, 0.1)

  def test_compare_adagrad_momentum(self):
    self.assertEqual(
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.9,
            initial_accumulator_value=0.1,
            initial_momentum_value=0.0,
        ),
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.9,
            initial_accumulator_value=0.1,
            initial_momentum_value=0.0,
        ),
    )
    self.assertNotEqual(
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.8,
            initial_accumulator_value=0.1,
            initial_momentum_value=0.0,
        ),
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.9,
            initial_accumulator_value=0.1,
            initial_momentum_value=0.0,
        ),
    )
    self.assertNotEqual(
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.9,
            initial_accumulator_value=0.2,
            initial_momentum_value=0.0,
        ),
        embedding_spec.AdagradMomentumOptimizerSpec(
            learning_rate=0.1,
            momentum=0.9,
            initial_accumulator_value=0.1,
            initial_momentum_value=0.0,
        ),
    )
    op = embedding_spec.AdagradMomentumOptimizerSpec(
        learning_rate=0.2,
        momentum=0.7,
        initial_accumulator_value=0.3,
        initial_momentum_value=0.1,
    )
    self.assertEqual(op.learning_rate, 0.2)
    self.assertEqual(op.momentum, 0.7)
    self.assertEqual(op.initial_accumulator_value, 0.3)
    self.assertEqual(op.initial_momentum_value, 0.1)

  def test_compare_laprop(self):
    self.assertEqual(
        embedding_spec.LaPropOptimizerSpec(
            learning_rate=0.1,
            b1=0.9,
            b2=0.95,
            eps=1e-30,
            rms_clip_threshold=1.0,
            initial_slot_value=0.0,
        ),
        embedding_spec.LaPropOptimizerSpec(
            learning_rate=0.1,
            b1=0.9,
            b2=0.95,
            eps=1e-30,
            rms_clip_threshold=1.0,
            initial_slot_value=0.0,
        ),
    )
    self.assertNotEqual(
        embedding_spec.LaPropOptimizerSpec(
            learning_rate=0.1,
            b1=0.8,
            b2=0.95,
            eps=1e-30,
            rms_clip_threshold=1.0,
            initial_slot_value=0.0,
        ),
        embedding_spec.LaPropOptimizerSpec(
            learning_rate=0.1,
            b1=0.9,
            b2=0.95,
            eps=1e-30,
            rms_clip_threshold=1.0,
            initial_slot_value=0.0,
        ),
    )

  def test_compare_adam(self):
    self.assertEqual(
        embedding_spec.AdamOptimizerSpec(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        embedding_spec.AdamOptimizerSpec(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
    )
    self.assertNotEqual(
        embedding_spec.AdamOptimizerSpec(
            learning_rate=0.1,
            beta_1=0.8,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        embedding_spec.AdamOptimizerSpec(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
    )

  def test_compare_ftrl(self):
    self.assertEqual(
        embedding_spec.FTRLOptimizerSpec(
            learning_rate=0.1,
            learning_rate_power=-0.5,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1,
            beta=0.1,
            initial_accumulator_value=0.1,
            initial_linear_value=0.0,
        ),
        embedding_spec.FTRLOptimizerSpec(
            learning_rate=0.1,
            learning_rate_power=-0.5,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1,
            beta=0.1,
            initial_accumulator_value=0.1,
            initial_linear_value=0.0,
        ),
    )
    self.assertNotEqual(
        embedding_spec.FTRLOptimizerSpec(learning_rate=0.1),
        embedding_spec.FTRLOptimizerSpec(learning_rate=0.2),
    )
    self.assertNotEqual(
        embedding_spec.FTRLOptimizerSpec(l1_regularization_strength=0.1),
        embedding_spec.FTRLOptimizerSpec(l1_regularization_strength=0.2),
    )
    op = embedding_spec.FTRLOptimizerSpec(
        learning_rate=0.05,
        l1_regularization_strength=0.01,
    )
    self.assertEqual(op.learning_rate, 0.05)
    self.assertEqual(op.l1_regularization_strength, 0.01)

  def test_adagrad_optimizer_primitive_and_initializers(self):
    op = embedding_spec.AdagradOptimizerSpec(
        learning_rate=0.1, initial_accumulator_value=0.05
    )
    expected_primitive = (
        sparse_dense_matmul_grad_with_adagrad.tpu_sparse_dense_matmul_grad_with_adagrad_primitive
    )
    self.assertEqual(op.get_optimizer_primitive(), expected_primitive)

    slot_inits = op.slot_variables_initializers()
    self.assertIsInstance(slot_inits, embedding_spec.AdagradSlotVariables)
    self.assertTrue(callable(slot_inits.accumulator))

    dummy_key = jax.random.PRNGKey(0)
    dummy_shape = (2, 3)
    self.assertTrue(
        jnp.allclose(
            slot_inits.accumulator(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_accumulator_value),
        )
    )

  def test_adagrad_momentum_optimizer_primitive_and_initializers(self):
    op = embedding_spec.AdagradMomentumOptimizerSpec(
        learning_rate=0.1,
        momentum=0.9,
        initial_accumulator_value=0.05,
        initial_momentum_value=0.02,
    )
    expected_primitive = (
        sparse_dense_matmul_grad_with_adagrad_momentum.tpu_sparse_dense_matmul_grad_with_adagrad_momentum_primitive
    )
    self.assertEqual(op.get_optimizer_primitive(), expected_primitive)

    slot_inits = op.slot_variables_initializers()
    self.assertIsInstance(
        slot_inits, embedding_spec.AdagradMomentumSlotVariables
    )
    self.assertTrue(callable(slot_inits.accumulator))
    self.assertTrue(callable(slot_inits.momentum))

    dummy_key = jax.random.PRNGKey(0)
    dummy_shape = (2, 3)
    self.assertTrue(
        jnp.allclose(
            slot_inits.accumulator(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_accumulator_value),
        )
    )
    self.assertTrue(
        jnp.allclose(
            slot_inits.momentum(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_momentum_value),
        )
    )

  def test_laprop_optimizer_primitive_and_initializers(self):
    op = embedding_spec.LaPropOptimizerSpec(
        learning_rate=0.1, initial_slot_value=0.02
    )
    expected_primitive = (
        sparse_dense_matmul_grad_with_laprop.tpu_sparse_dense_matmul_grad_with_laprop_primitive
    )
    self.assertEqual(op.get_optimizer_primitive(), expected_primitive)

    slot_inits = op.slot_variables_initializers()
    self.assertIsInstance(slot_inits, embedding_spec.LaPropSlotVariables)
    self.assertTrue(callable(slot_inits.mu))
    self.assertTrue(callable(slot_inits.nu))

    dummy_key = jax.random.PRNGKey(1)
    dummy_shape = (3, 2)
    self.assertTrue(
        jnp.allclose(
            slot_inits.mu(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_slot_value),
        )
    )
    self.assertTrue(
        jnp.allclose(
            slot_inits.nu(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_slot_value),
        )
    )

  def test_ftrl_optimizer_primitive_and_initializers(self):
    op = embedding_spec.FTRLOptimizerSpec(
        learning_rate=0.05,
        initial_accumulator_value=0.1,
        initial_linear_value=0.01,
    )
    expected_primitive = (
        sparse_dense_matmul_grad_with_ftrl.tpu_sparse_dense_matmul_grad_with_ftrl_primitive
    )
    self.assertEqual(op.get_optimizer_primitive(), expected_primitive)

    slot_inits = op.slot_variables_initializers()
    self.assertIsInstance(slot_inits, embedding_spec.FTRLSlotVariables)
    self.assertTrue(callable(slot_inits.accumulator))
    self.assertTrue(callable(slot_inits.linear))

    dummy_key = jax.random.PRNGKey(2)
    dummy_shape = (1, 5)
    self.assertTrue(
        jnp.allclose(
            slot_inits.accumulator(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_accumulator_value),
        )
    )
    self.assertTrue(
        jnp.allclose(
            slot_inits.linear(dummy_key, dummy_shape),
            jnp.full(dummy_shape, op.initial_linear_value),
        )
    )

  def test_learning_rate_callable(self):

    def lr_callable():
      return 0.1

    optimizer_specs = [
        embedding_spec.SGDOptimizerSpec(learning_rate=lr_callable),
        embedding_spec.AdagradOptimizerSpec(learning_rate=lr_callable),
        embedding_spec.LaPropOptimizerSpec(learning_rate=lr_callable),
        embedding_spec.FTRLOptimizerSpec(learning_rate=lr_callable),
        embedding_spec.AdagradMomentumOptimizerSpec(learning_rate=lr_callable),
    ]
    for op_spec in optimizer_specs:
      self.assertEqual(op_spec.get_learning_rate(), 0.1)

  def test_short_name(self):
    optimizers = {
        "sgd": embedding_spec.SGDOptimizerSpec(),
        "adagrad": embedding_spec.AdagradOptimizerSpec(),
        "laprop": embedding_spec.LaPropOptimizerSpec(),
        "ftrl": embedding_spec.FTRLOptimizerSpec(),
        "adagrad_momentum": embedding_spec.AdagradMomentumOptimizerSpec(),
    }
    for expected_name, optimizer_spec in optimizers.items():
      self.assertEqual(optimizer_spec.short_name(), expected_name)

  def test_learning_rate_schedule(self):
    lr_schedule = schedules.linear_schedule(
        init_value=1.0, end_value=0.1, transition_steps=100
    )
    optimizer_specs = [
        embedding_spec.SGDOptimizerSpec(learning_rate=lr_schedule),
        embedding_spec.AdagradOptimizerSpec(learning_rate=lr_schedule),
        embedding_spec.LaPropOptimizerSpec(learning_rate=lr_schedule),
        embedding_spec.FTRLOptimizerSpec(learning_rate=lr_schedule),
        embedding_spec.AdagradMomentumOptimizerSpec(learning_rate=lr_schedule),
    ]

    for op_spec in optimizer_specs:
      self.assertEqual(op_spec.get_learning_rate(0), 1.0)
      self.assertEqual(op_spec.get_learning_rate(50), 0.55)
      self.assertEqual(op_spec.get_learning_rate(100), 0.1)

  def test_hyperparameters(self):
    op = embedding_spec.AdagradOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        )
    )
    self.assertEqual(op.get_hyperparameters(0), (1.0,))

    op = embedding_spec.AdagradMomentumOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        ),
        momentum=0.9,
    )
    expected_hyperparameters_adagrad_momentum = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.9, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(1e-10, dtype=jnp.float32),
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array(False, dtype=jnp.bool_),
    )
    self.assertEqual(
        op.get_hyperparameters(0), expected_hyperparameters_adagrad_momentum
    )

    self.assertEqual(
        op.get_hyperparameters(0), expected_hyperparameters_adagrad_momentum
    )

    op = embedding_spec.AdamOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        ),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )
    c2 = jnp.sqrt(1 - jnp.float32(op.beta_2))
    alpha_t = op.learning_rate(0) * c2 / (1 - jnp.float32(op.beta_1))
    epsilon_hat = op.epsilon * c2
    expected_hyperparameters = (
        jnp.array(alpha_t, dtype=jnp.float32),  # alpha_t
        jnp.array(0.9, dtype=jnp.float32),  # beta_1
        jnp.array(0.999, dtype=jnp.float32),  # beta_2
        jnp.array(epsilon_hat, dtype=jnp.float32),  # epsilon_hat
    )
    self.assertEqual(op.get_hyperparameters(0), expected_hyperparameters)

    op = embedding_spec.FTRLOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        ),
        learning_rate_power=-0.5,
        l1_regularization_strength=0.01,
        l2_regularization_strength=0.02,
        beta=0.001,
    )
    expected_hyperparameters_ftrl = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(-0.5, dtype=jnp.float32),
        jnp.array(0.01, dtype=jnp.float32),
        jnp.array(0.02, dtype=jnp.float32),
        jnp.array(0.001, dtype=jnp.float32),
    )
    self.assertEqual(op.get_hyperparameters(0), expected_hyperparameters_ftrl)

  def test_table_spec_quantization_config_equality(self):
    """Tables should compare equal only when the quantization config matches."""
    q_cfg = embedding_spec.QuantizationConfig(
        min_value=0.0, max_value=10.0, num_buckets=128
    )
    initializer = jax.nn.initializers.normal()
    ts1 = embedding_spec.TableSpec(
        vocabulary_size=8,
        embedding_dim=4,
        name="t",
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        initializer=initializer,
        quantization_config=q_cfg,
    )
    ts2 = embedding_spec.TableSpec(
        vocabulary_size=8,
        embedding_dim=4,
        name="t",
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        initializer=initializer,
        quantization_config=q_cfg,
    )
    ts3 = embedding_spec.TableSpec(  # No quantization
        vocabulary_size=8,
        embedding_dim=4,
        name="t",
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        initializer=initializer,
        quantization_config=None,
    )
    self.assertEqual(ts1, ts2)
    self.assertNotEqual(ts1, ts3)


if __name__ == "__main__":
  absltest.main()
