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
"""Tests for sparse_dense_matmul_gradient_stack.

Note: These tests only verify abstract evaluation and lowering. Functional
correctness is not verified on CPU as the primitive relies on a TPU-specific
custom call.
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_gradient_stack


class SparseDenseMatmulGradientStackTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size_1 = 128
    self.dim_1 = 32
    self.batch_size_2 = 256
    self.dim_2 = 32

    self.stacked_batch_size = self.batch_size_1 + self.batch_size_2
    self.stacked_feature_dim = 32

    self.grad1 = jnp.zeros((self.batch_size_1, self.dim_1), dtype=jnp.float32)
    self.grad2 = jnp.zeros((self.batch_size_2, self.dim_2), dtype=jnp.float32)

    self.stack_primitive = jax.named_call(
        sparse_dense_matmul_gradient_stack.tpu_sparse_dense_matmul_gradient_stack_primitive.bind,
        name="tpu_sparse_dense_matmul_gradient_stack",
    )

  def test_abstract_eval(self):
    def fun(g1, g2):
      return self.stack_primitive(
          g1,
          g2,
          stacked_batch_size=self.stacked_batch_size,
          stacked_feature_dim=self.stacked_feature_dim,
      )

    out_aval = jax.eval_shape(fun, self.grad1, self.grad2)

    self.assertEqual(
        out_aval.shape, (self.stacked_batch_size, self.stacked_feature_dim)
    )
    self.assertEqual(out_aval.dtype, jnp.float32)

  def test_functional_correctness(self):
    # Mock implementation that assumes sequential stacking
    def impl(
        *unstacked_gradients,
        stacked_batch_size,
        stacked_feature_dim,
    ):
      del stacked_batch_size
      del stacked_feature_dim
      return jnp.concatenate(unstacked_gradients, axis=0)

    sparse_dense_matmul_gradient_stack.tpu_sparse_dense_matmul_gradient_stack_primitive.def_impl(
        impl
    )

    g1 = jnp.ones((self.batch_size_1, self.dim_1), dtype=jnp.float32)
    g2 = jnp.ones((self.batch_size_2, self.dim_2), dtype=jnp.float32) * 2

    output = self.stack_primitive(
        g1, g2,
        stacked_batch_size=self.stacked_batch_size,
        stacked_feature_dim=self.stacked_feature_dim,
    )

    expected = jnp.concatenate([g1, g2], axis=0)
    self.assertTrue(jnp.array_equal(output, expected))


if __name__ == "__main__":
  absltest.main()
