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
import jax.numpy as jnp
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

  def test_learning_rate_callable(self):
    def lr():
      return 0.1

    op = embedding_spec.AdagradOptimizerSpec(learning_rate=lr)
    self.assertEqual(op.get_learning_rate(), 0.1)

  def test_learning_rate_schedule(self):
    op = embedding_spec.AdagradOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        )
    )

    self.assertEqual(op.get_learning_rate(0), 1.0)
    self.assertEqual(op.get_learning_rate(50), 0.55)
    self.assertEqual(op.get_learning_rate(100), 0.1)

  def test_hyperparameters(self):
    op = embedding_spec.AdagradOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        )
    )
    self.assertEqual(op.get_hyperparameters(0), (1.0,))

    op = embedding_spec.LaPropOptimizerSpec(
        learning_rate=schedules.linear_schedule(
            init_value=1.0, end_value=0.1, transition_steps=100
        ),
        b1=0.9,
        b2=0.95,
        eps=1e-30,
        rms_clip_threshold=1.0,
        initial_slot_value=0.0,
    )
    expected_hyperparameters = (
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.9, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
        jnp.array(1e-30, dtype=jnp.float32),
    )
    self.assertEqual(op.get_hyperparameters(0), expected_hyperparameters)

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


if __name__ == "__main__":
  absltest.main()
