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
"""Unit tests for SparseCore embedding pipelining utility functions."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_pipelining_utils
import numpy as np


class EmbeddingPipeliningUtilsTest(parameterized.TestCase):

  def test_safe_copy_jax_array(self):
    arr = jnp.array([1.0, 2.0, 3.0])
    copied = embedding_pipelining_utils._safe_copy(arr)
    self.assertIsInstance(copied, jax.Array)
    np.testing.assert_array_equal(copied, arr)

  def test_safe_copy_numpy_array(self):
    arr = np.array([4, 5, 6])
    copied = embedding_pipelining_utils._safe_copy(arr)
    self.assertIsInstance(copied, jax.Array)
    np.testing.assert_array_equal(copied, arr)

  def test_safe_copy_shape_dtype_struct(self):
    struct = jax.ShapeDtypeStruct(shape=(8, 16), dtype=jnp.float32)
    copied = embedding_pipelining_utils._safe_copy(struct)
    self.assertIs(copied, struct)

  def test_safe_copy_abstract_value(self):
    shaped_array = jax.core.ShapedArray(shape=(4,), dtype=jnp.int32)
    copied = embedding_pipelining_utils._safe_copy(shaped_array)
    self.assertIs(copied, shaped_array)

  @parameterized.named_parameters(
      ("none", None),
      ("int", 42),
      ("float", 3.14),
      ("str", "test"),
      ("tuple", (1, 2)),
  )
  def test_safe_copy_non_arrays(self, value):
    copied = embedding_pipelining_utils._safe_copy(value)
    self.assertEqual(copied, value)

  def test_get_initial_state_eval_shape(self):
    pipeline_input = embedding_pipelining_utils.CurrentStepInput(
        sparse_inputs=jnp.zeros((4, 8), dtype=jnp.float32),
        dense_inputs=jnp.zeros((4, 16), dtype=jnp.float32),
    )
    tc_train_state = {
        "params": jnp.zeros((16, 8), dtype=jnp.float32)
    }
    embedding_variables = {
        "table": jnp.zeros((100, 8), dtype=jnp.float32)
    }

    def sc_fwd_fn(sparse_inputs, embedding_variables):
      del embedding_variables
      return sparse_inputs, None

    def tc_fn(
        embedding_activations,
        dense_inputs,
        train_state,
        sc_fwd_aux=None,
    ):
      del dense_inputs, sc_fwd_aux
      emb_grad = embedding_activations
      out = jnp.zeros((4,), dtype=jnp.float32)
      return emb_grad, out, train_state, None

    state_shape = jax.eval_shape(
        functools.partial(
            embedding_pipelining_utils.get_initial_state,
            sc_fwd_function=sc_fwd_fn,
            tc_function=tc_fn,
        ),
        pipeline_input,
        tc_train_state,
        embedding_variables,
    )
    self.assertEqual(state_shape.pipeline_step.shape, ())
    self.assertEqual(
        state_shape.step_before_last_step_inputs.sparse_inputs.shape, (4, 8)
    )


if __name__ == "__main__":
  absltest.main()
