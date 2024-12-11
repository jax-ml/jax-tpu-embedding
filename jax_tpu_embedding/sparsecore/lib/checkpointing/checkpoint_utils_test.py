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
"""Checkpoint utils tests."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.checkpointing import checkpoint_utils
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


def assert_embedding_variable_equal(testclass, x, y, array):
  testclass.assertIsInstance(x, embedding.EmbeddingVariables)
  testclass.assertIsInstance(y, embedding.EmbeddingVariables)
  testclass.assertTrue(jnp.array_equal(array, y.table))
  testclass.assertEqual(x.slot, y.slot)


class CheckpointUtilsTest(absltest.TestCase):

  def test_convert_orbax_restored_dict_to_embedding_variables_sgd(self):
    table_array = jnp.array([1, 2, 3])

    init_ev = {
        "r": embedding.EmbeddingVariables(
            table=table_array,
            slot=embedding_spec.SGDSlotVariables(),
        )
    }
    restored = {
        "r": {
            "table": table_array,
            "slot": None,
        }
    }
    restored_ev = (
        checkpoint_utils.convert_orbax_restored_dict_to_embedding_variables(
            init_ev, restored
        )
    )
    jax.tree.map(
        lambda x, y: assert_embedding_variable_equal(self, x, y, table_array),
        init_ev,
        restored_ev,
        is_leaf=lambda x: isinstance(x, embedding.EmbeddingVariables),
    )

  def test_convert_orbax_restored_dict_to_embedding_variables_adagrad(self):
    accumulator_array = jnp.array([0.1, 0.2, 0.3])
    table_array = jnp.array([1, 2, 3])

    init_ev = {
        "r": embedding.EmbeddingVariables(
            table=table_array,
            slot=embedding_spec.AdagradSlotVariables(
                accumulator=accumulator_array,
            ),
        )
    }
    restored = {
        "r": {
            "table": table_array,
            "slot": {"accumulator": accumulator_array},
        }
    }
    restored_ev = (
        checkpoint_utils.convert_orbax_restored_dict_to_embedding_variables(
            init_ev, restored
        )
    )
    jax.tree.map(
        lambda x, y: assert_embedding_variable_equal(self, x, y, table_array),
        init_ev,
        restored_ev,
        is_leaf=lambda x: isinstance(x, embedding.EmbeddingVariables),
    )


if __name__ == "__main__":
  absltest.main()
