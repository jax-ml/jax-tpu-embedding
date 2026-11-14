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

from typing import override

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core.primitives import (
    sparse_dense_matmul_activation_unstack as activation_unstack,
)
import numpy as np


class SparseDenseMatmulActivationUnstackTest(parameterized.TestCase):

  @override
  def setUp(self):
    super().setUp()
    self.batch_size_1 = 128
    self.dim_1 = 32
    self.batch_size_2 = 256
    self.dim_2 = 64
    self.input_rows = self.batch_size_1 + self.batch_size_2
    self.input_dim = max(self.dim_1, self.dim_2)
    self.input_shape = (self.input_rows, self.input_dim)
    self.per_feature_batch_sizes = (self.batch_size_1, self.batch_size_2)
    self.per_feature_dims = (self.dim_1, self.dim_2)

    self.primitive = (
        activation_unstack.tpu_sparse_dense_matmul_activation_unstack_primitive
    )
    self.unstack_primitive = jax.named_call(
        self.primitive.bind,
        name="tpu_sparse_dense_matmul_activation_unstack",
    )

  @parameterized.named_parameters(
      ("default", None, jnp.float32),
      ("float32", jnp.float32, jnp.float32),
      ("bfloat16", jnp.bfloat16, jnp.bfloat16),
      ("bfloat16_str", "bfloat16", jnp.bfloat16),
  )
  def test_abstract_eval(self, out_dtype, expected_dtype):
    kwargs = {} if out_dtype is None else {"out_dtype": out_dtype}
    out_avals = jax.eval_shape(
        lambda x: self.unstack_primitive(
            x,
            per_feature_batch_sizes=self.per_feature_batch_sizes,
            per_feature_dims=self.per_feature_dims,
            **kwargs,
        ),
        jnp.zeros(self.input_shape, dtype=jnp.float32),
    )

    self.assertLen(out_avals, 2)
    self.assertEqual(out_avals[0].shape, (self.batch_size_1, self.dim_1))
    self.assertEqual(out_avals[0].dtype, expected_dtype)
    self.assertEqual(out_avals[1].shape, (self.batch_size_2, self.dim_2))
    self.assertEqual(out_avals[1].dtype, expected_dtype)

  @parameterized.named_parameters(
      ("default", None, "f32"),
      ("float32", jnp.float32, "f32"),
      ("bfloat16", jnp.bfloat16, "bf16"),
  )
  def test_mlir_lowering(self, out_dtype, ir_elem_type):
    kwargs = {} if out_dtype is None else {"out_dtype": out_dtype}
    lowered = jax.jit(
        lambda x: self.unstack_primitive(
            x,
            per_feature_batch_sizes=self.per_feature_batch_sizes,
            per_feature_dims=self.per_feature_dims,
            **kwargs,
        )
    ).lower(jnp.zeros(self.input_shape, dtype=jnp.float32))
    mlir_ir = str(lowered.compiler_ir())

    self.assertIn("SparseActivationsUnstackInterleaved", mlir_ir)
    self.assertIn("tensor<384x64xf32>", mlir_ir)
    self.assertIn(f"tensor<128x32x{ir_elem_type}>", mlir_ir)
    self.assertIn(f"tensor<256x64x{ir_elem_type}>", mlir_ir)

  @parameterized.named_parameters(
      ("float32", jnp.float32),
      ("bfloat16", jnp.bfloat16),
  )
  def test_functional_correctness(self, out_dtype):
    def impl(
        stacked_activations,
        *,
        per_feature_batch_sizes,
        per_feature_dims,
        out_dtype=jnp.float32,
    ):
      dtype = jnp.dtype(out_dtype)
      activations = []
      start = 0
      for batch_size, dim in zip(per_feature_batch_sizes, per_feature_dims):
        end = start + batch_size
        activations.append(stacked_activations[start:end, :dim].astype(dtype))
        start = end
      return tuple(activations)

    self.addCleanup(self.primitive.def_impl, self.primitive.impl)
    self.primitive.def_impl(impl)

    input_data = (
        jnp.arange(self.input_rows * self.input_dim, dtype=jnp.float32).reshape(
            self.input_rows, self.input_dim
        )
        * 0.125
    )
    outputs = self.unstack_primitive(
        input_data,
        per_feature_batch_sizes=self.per_feature_batch_sizes,
        per_feature_dims=self.per_feature_dims,
        out_dtype=out_dtype,
    )

    self.assertLen(outputs, 2)
    self.assertEqual(outputs[0].dtype, out_dtype)
    self.assertEqual(outputs[1].dtype, out_dtype)
    np.testing.assert_array_equal(
        outputs[0],
        input_data[: self.batch_size_1, : self.dim_1].astype(out_dtype),
    )
    np.testing.assert_array_equal(
        outputs[1],
        input_data[self.batch_size_1 :, : self.dim_2].astype(out_dtype),
    )

  @parameterized.named_parameters(
      (
          "unsupported_out_dtype_f16",
          (384, 64),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.float16,
          "Unsupported out_dtype",
      ),
      (
          "unsupported_out_dtype_i32",
          (384, 64),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.int32,
          "Unsupported out_dtype",
      ),
      (
          "bf16_input_with_bf16_out",
          (384, 64),
          jnp.bfloat16,
          (128, 256),
          (32, 64),
          jnp.bfloat16,
          "stacked_activations must have dtype float32",
      ),
      (
          "bf16_input_with_f32_out",
          (384, 64),
          jnp.bfloat16,
          (128, 256),
          (32, 64),
          jnp.float32,
          "stacked_activations must have dtype float32",
      ),
      (
          "rank_1_input",
          (384,),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.float32,
          "stacked_activations must be rank 2",
      ),
      (
          "rank_3_input",
          (1, 384, 64),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.float32,
          "stacked_activations must be rank 2",
      ),
      (
          "empty_metadata",
          (384, 64),
          jnp.float32,
          (),
          (),
          jnp.float32,
          "must be non-empty and have matching lengths",
      ),
      (
          "mismatched_metadata_lengths",
          (384, 64),
          jnp.float32,
          (128,),
          (32, 64),
          jnp.float32,
          "must be non-empty and have matching lengths",
      ),
      (
          "non_positive_batch_size",
          (256, 64),
          jnp.float32,
          (0, 256),
          (32, 64),
          jnp.float32,
          "must be positive",
      ),
      (
          "non_positive_feature_dim",
          (384, 64),
          jnp.float32,
          (128, 256),
          (0, 64),
          jnp.float32,
          "must be positive",
      ),
      (
          "row_sum_mismatch",
          (300, 64),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.float32,
          "Sum of per_feature_batch_sizes",
      ),
      (
          "feature_dim_exceeds_stacked_cols",
          (384, 48),
          jnp.float32,
          (128, 256),
          (32, 64),
          jnp.float32,
          "Max of per_feature_dims",
      ),
  )
  def test_abstract_eval_validation_errors(
      self,
      in_shape,
      in_dtype,
      batch_sizes,
      feature_dims,
      out_dtype,
      expected_error,
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      jax.eval_shape(
          lambda x: self.unstack_primitive(
              x,
              per_feature_batch_sizes=batch_sizes,
              per_feature_dims=feature_dims,
              out_dtype=out_dtype,
          ),
          jnp.zeros(in_shape, dtype=in_dtype),
      )


if __name__ == "__main__":
  absltest.main()
