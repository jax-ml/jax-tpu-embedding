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
"""Tests for StableHLO lowering utilities for custom SparseCore optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import custom_optimizer_lowering


class CustomOptimizerLoweringTest(parameterized.TestCase):

  def test_apply_clipping(self):
    val = jnp.array([-2.0, 0.0, 2.0, 5.0])
    # No clipping
    self.assertTrue(
        jnp.allclose(custom_optimizer_lowering.apply_clipping(val), val)
    )
    # Min clipping only
    self.assertTrue(
        jnp.allclose(
            custom_optimizer_lowering.apply_clipping(val, min_value=0.0),
            jnp.array([0.0, 0.0, 2.0, 5.0]),
        )
    )
    # Max clipping only
    self.assertTrue(
        jnp.allclose(
            custom_optimizer_lowering.apply_clipping(val, max_value=3.0),
            jnp.array([-2.0, 0.0, 2.0, 3.0]),
        )
    )
    # Both min and max clipping
    self.assertTrue(
        jnp.allclose(
            custom_optimizer_lowering.apply_clipping(
                val, min_value=0.0, max_value=3.0
            ),
            jnp.array([0.0, 0.0, 2.0, 3.0]),
        )
    )

  def test_lower_to_stablehlo_basic(self):
    def simple_sgd(grad, param, lr):
      return param - lr * grad

    stablehlo_text = custom_optimizer_lowering.lower_to_stablehlo(
        simple_sgd,
        embedding_dim=4,
        num_slot_variables=0,
        num_hyperparameters=1,
    )
    self.assertIsInstance(stablehlo_text, str)
    self.assertIn("module", stablehlo_text)
    self.assertIn("stablehlo", stablehlo_text.lower())

  def test_lower_to_stablehlo_with_slots_and_clipping(self):
    def custom_adagrad(grad, param, accum, lr):
      new_accum = accum + grad * grad
      new_param = param - lr * grad / (jnp.sqrt(new_accum) + 1e-7)
      return new_param, new_accum

    stablehlo_text = custom_optimizer_lowering.lower_to_stablehlo(
        custom_adagrad,
        embedding_dim=8,
        num_slot_variables=1,
        num_hyperparameters=1,
        min_value=-1.0,
        max_value=1.0,
    )
    self.assertIsInstance(stablehlo_text, str)
    self.assertIn("module", stablehlo_text)

  def test_wrap_stablehlo_with_limits(self):
    def simple_sgd(grad, param, lr):
      return param - lr * grad

    raw_stablehlo = custom_optimizer_lowering.lower_to_stablehlo(
        simple_sgd,
        embedding_dim=4,
        num_slot_variables=0,
        num_hyperparameters=1,
    )

    # Without limits, returns original string
    no_limits = custom_optimizer_lowering.wrap_stablehlo_with_limits(
        raw_stablehlo,
        embedding_dim=4,
        num_slot_variables=0,
        num_hyperparameters=1,
    )
    self.assertEqual(no_limits, raw_stablehlo)

    # With limits, re-traces and lowers with clipping
    with_limits = custom_optimizer_lowering.wrap_stablehlo_with_limits(
        raw_stablehlo,
        embedding_dim=4,
        num_slot_variables=0,
        num_hyperparameters=1,
        min_value=0.0,
        max_value=10.0,
    )
    self.assertIsInstance(with_limits, str)
    self.assertIn("module", with_limits)


if __name__ == "__main__":
  absltest.main()
