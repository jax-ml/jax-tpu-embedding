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
import operator
from typing import Any

from absl.testing import absltest
from flax import struct
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed_optimizer
import optax


class EmbedOptimizerTest(absltest.TestCase):

  def test_create_and_apply_optimizer_for_sc_model(self):
    # Define a sample model params with the specified path
    class ModelParams(struct.PyTreeNode):
      params: Any  # Unnecessary indirection to introduce a non-dict key

    model_params = ModelParams(
        params={
            "layers_0": {
                "sc_embedding_variables": {
                    "value": {"table": {"table": jnp.array([1.0, 2.0])}}
                }
            },
            "layers_2": {
                "Dense_0": {
                    "bias": jnp.array([1.0, 2.0]),
                    "kernel": jnp.array([1.0, 2.0]),
                },
                "Dense_1": {
                    "bias": jnp.array([1.0, 2.0]),
                    "kernel": jnp.array([1.0, 2.0]),
                },
            },
        }
    )
    # Create the optimizer
    optimizer: optax.GradientTransformation = (
        embed_optimizer.create_optimizer_for_sc_model(
            model_params,
            tc_optimizer=optax.sgd(learning_rate=0.1),
        )
    )

    state = optimizer.init(model_params)  # pytype:disable=wrong-arg-types
    rand_key = jax.random.key(42)
    updates = jax.tree.map(
        lambda x: jax.random.uniform(rand_key, x.shape), model_params
    )
    transformed_updates, _ = optimizer.update(updates, state, model_params)  # pytype:disable=wrong-arg-types
    new_model_params: ModelParams = embed_optimizer.apply_updates_for_sc_model(
        model_params, transformed_updates
    )
    expected_updated_params = ModelParams(
        params={
            "layers_0": updates.params[
                "layers_0"
            ],  # SC apply updates just return updates
            "layers_2": jax.tree.map(
                operator.add,
                model_params.params["layers_2"],
                transformed_updates.params["layers_2"],
            ),
        }
    )
    jax.tree.map(
        lambda x, y: self.assertTrue(
            jnp.allclose(x, y), f"Arrays {x} and {y} are not equal."
        ),
        new_model_params,
        expected_updated_params,
    )


if __name__ == "__main__":
  absltest.main()
