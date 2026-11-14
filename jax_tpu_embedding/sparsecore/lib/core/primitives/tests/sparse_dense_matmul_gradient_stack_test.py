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

from typing import override

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import (
    sparse_dense_matmul_gradient_stack as gradient_stack,
)
import numpy as np


class SparseDenseMatmulGradientStackTest(parameterized.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.batch_size_1 = 128
    self.dim_1 = 32
    self.batch_size_2 = 256
    self.dim_2 = 32
    self.stacked_batch_size = self.batch_size_1 + self.batch_size_2
    self.stacked_feature_dim = 32

    self.primitive = (
        gradient_stack.tpu_sparse_dense_matmul_gradient_stack_primitive
    )
    self.stack_primitive = jax.named_call(
        self.primitive.bind,
        name="tpu_sparse_dense_matmul_gradient_stack",
    )

  @parameterized.named_parameters(
      ("float32", jnp.float32),
      ("bfloat16", jnp.bfloat16),
  )
  def test_abstract_eval(self, in_dtype):
    g1 = jnp.zeros((self.batch_size_1, self.dim_1), dtype=in_dtype)
    g2 = jnp.zeros((self.batch_size_2, self.dim_2), dtype=in_dtype)
    out_aval = jax.eval_shape(
        lambda x, y: self.stack_primitive(
            x,
            y,
            stacked_batch_size=self.stacked_batch_size,
            stacked_feature_dim=self.stacked_feature_dim,
        ),
        g1,
        g2,
    )

    self.assertEqual(
        out_aval.shape, (self.stacked_batch_size, self.stacked_feature_dim)
    )
    self.assertEqual(out_aval.dtype, jnp.float32)

  @parameterized.named_parameters(
      ("float32", jnp.float32, "f32"),
      ("bfloat16", jnp.bfloat16, "bf16"),
  )
  def test_mlir_lowering(self, in_dtype, ir_elem_type):
    g1 = jnp.zeros((self.batch_size_1, self.dim_1), dtype=in_dtype)
    g2 = jnp.zeros((self.batch_size_2, self.dim_2), dtype=in_dtype)
    lowered = jax.jit(
        lambda x, y: self.stack_primitive(
            x,
            y,
            stacked_batch_size=self.stacked_batch_size,
            stacked_feature_dim=self.stacked_feature_dim,
        )
    ).lower(g1, g2)
    mlir_ir = str(lowered.compiler_ir())

    self.assertIn("SparseGradientsStackInterleaved", mlir_ir)
    self.assertIn(f"tensor<128x32x{ir_elem_type}>", mlir_ir)
    self.assertIn(f"tensor<256x32x{ir_elem_type}>", mlir_ir)
    self.assertIn("tensor<384x32xf32>", mlir_ir)

  @parameterized.named_parameters(
      ("float32", jnp.float32),
      ("bfloat16", jnp.bfloat16),
  )
  def test_functional_correctness(self, in_dtype):
    def impl(
        *unstacked_gradients,
        stacked_batch_size,
        stacked_feature_dim,
    ):
      del stacked_batch_size, stacked_feature_dim
      return jnp.concatenate(unstacked_gradients, axis=0).astype(jnp.float32)

    self.addCleanup(self.primitive.def_impl, self.primitive.impl)
    self.primitive.def_impl(impl)

    g1 = jnp.ones((self.batch_size_1, self.dim_1), dtype=in_dtype)
    g2 = jnp.ones((self.batch_size_2, self.dim_2), dtype=in_dtype) * 2
    output = self.stack_primitive(
        g1,
        g2,
        stacked_batch_size=self.stacked_batch_size,
        stacked_feature_dim=self.stacked_feature_dim,
    )

    expected = jnp.concatenate([g1, g2], axis=0).astype(jnp.float32)
    self.assertEqual(output.dtype, jnp.float32)
    np.testing.assert_array_equal(output, expected)

  @parameterized.named_parameters(
      (
          "empty_inputs",
          (),
          384,
          32,
          "unstacked_gradients must be non-empty",
      ),
      (
          "rank_1_input",
          (((128,), jnp.bfloat16), ((256, 32), jnp.bfloat16)),
          384,
          32,
          "Each unstacked gradient must be rank 2",
      ),
      (
          "rank_3_input",
          (((1, 128, 32), jnp.bfloat16), ((256, 32), jnp.bfloat16)),
          384,
          32,
          "Each unstacked gradient must be rank 2",
      ),
      (
          "unsupported_dtype_f16",
          (((128, 32), jnp.float16), ((256, 32), jnp.float16)),
          384,
          32,
          "Unsupported unstacked gradient dtype",
      ),
      (
          "unsupported_dtype_i32",
          (((128, 32), jnp.int32), ((256, 32), jnp.int32)),
          384,
          32,
          "Unsupported unstacked gradient dtype",
      ),
      (
          "mixed_f32_bf16",
          (((128, 32), jnp.float32), ((256, 32), jnp.bfloat16)),
          384,
          32,
          "All unstacked_gradients must have the same dtype",
      ),
      (
          "mixed_bf16_f32",
          (((128, 32), jnp.bfloat16), ((256, 32), jnp.float32)),
          384,
          32,
          "All unstacked_gradients must have the same dtype",
      ),
      (
          "non_positive_stacked_batch",
          (((128, 32), jnp.float32),),
          0,
          32,
          "must be positive",
      ),
      (
          "non_positive_stacked_dim",
          (((128, 32), jnp.float32),),
          128,
          0,
          "must be positive",
      ),
      (
          "row_sum_mismatch",
          (((128, 32), jnp.bfloat16), ((256, 32), jnp.bfloat16)),
          300,
          32,
          "Sum of unstacked gradient rows",
      ),
      (
          "feature_dim_exceeds_stacked_dim",
          (((128, 32), jnp.bfloat16), ((256, 32), jnp.bfloat16)),
          384,
          16,
          "Max unstacked gradient columns",
      ),
  )
  def test_abstract_eval_validation_errors(
      self, input_specs, stacked_batch, stacked_dim, expected_error
  ):
    inputs = [jnp.zeros(shape, dtype=dt) for shape, dt in input_specs]
    with self.assertRaisesRegex(ValueError, expected_error):
      jax.eval_shape(
          lambda *xs: self.stack_primitive(
              *xs,
              stacked_batch_size=stacked_batch,
              stacked_feature_dim=stacked_dim,
          ),
          *inputs,
      )


if __name__ == "__main__":
  absltest.main()
