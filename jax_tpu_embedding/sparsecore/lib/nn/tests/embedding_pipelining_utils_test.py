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
"""Tests for embedding_pipelining_utils."""

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


if __name__ == "__main__":
  absltest.main()
