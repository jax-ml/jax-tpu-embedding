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
"""Tests for sparse_dense_matmul_activation_unstack.

Note: These tests only verify abstract evaluation and lowering. Functional
correctness is not verified on CPU as the primitive relies on a TPU-specific
custom call.
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_activation_unstack


class SparseDenseMatmulActivationUnstackTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size_1 = 128
    self.dim_1 = 32
    self.batch_size_2 = 256
    self.dim_2 = 64
    # Ensure input shape is consistent with outputs for functional testing.
    # Input rows = num_sc * (batch_size_1 + batch_size_2). Assuming num_sc=1.
    self.input_rows = self.batch_size_1 + self.batch_size_2
    self.input_dim = max(self.dim_1, self.dim_2)
    self.input_shape = (self.input_rows, self.input_dim)
    self.per_feature_batch_sizes = (self.batch_size_1, self.batch_size_2)
    self.per_feature_dims = (self.dim_1, self.dim_2)

    self.unstack_primitive = jax.named_call(
        sparse_dense_matmul_activation_unstack.tpu_sparse_dense_matmul_activation_unstack_primitive.bind,
        name="tpu_sparse_dense_matmul_activation_unstack",
    )

  def test_abstract_eval(self):
    def fun(x):
      return self.unstack_primitive(
          x,
          per_feature_batch_sizes=self.per_feature_batch_sizes,
          per_feature_dims=self.per_feature_dims,
      )

    out_avals = jax.eval_shape(fun, jnp.zeros(self.input_shape))

    self.assertLen(out_avals, 2)
    self.assertEqual(out_avals[0].shape, (self.batch_size_1, self.dim_1))
    self.assertEqual(out_avals[0].dtype, jnp.float32)
    self.assertEqual(out_avals[1].shape, (self.batch_size_2, self.dim_2))
    self.assertEqual(out_avals[1].dtype, jnp.float32)

  def test_functional_correctness(self):
    # Mock implementation that assumes sequential unstacking.
    def impl(
        stacked_activations,
        *,
        per_feature_batch_sizes,
        per_feature_dims,
    ):
      activations = []
      start = 0
      for batch_size, dim in zip(per_feature_batch_sizes, per_feature_dims):
        end = start + batch_size
        activations.append(stacked_activations[start:end, :dim])
        start = end
      return tuple(activations)

    sparse_dense_matmul_activation_unstack.tpu_sparse_dense_matmul_activation_unstack_primitive.def_impl(
        impl
    )

    input_data = jnp.arange(
        self.input_rows * self.input_dim, dtype=jnp.float32
    ).reshape(self.input_rows, self.input_dim)

    outputs = self.unstack_primitive(
        input_data,
        per_feature_batch_sizes=self.per_feature_batch_sizes,
        per_feature_dims=self.per_feature_dims,
    )

    self.assertLen(outputs, 2)

    # Verify outputs match sequential slicing
    expected_1 = input_data[0 : self.batch_size_1, : self.dim_1]
    expected_2 = input_data[
        self.batch_size_1 : self.batch_size_1 + self.batch_size_2, : self.dim_2
    ]

    self.assertTrue(jnp.array_equal(outputs[0], expected_1))
    self.assertTrue(jnp.array_equal(outputs[1], expected_2))


if __name__ == "__main__":
  absltest.main()
