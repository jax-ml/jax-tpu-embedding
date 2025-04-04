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
"""LaProp optimizer for sparse dense matmul backward pass.

This implements the Jax primitive for the LaProp optimizer for the sparse dense
matmul backward pass, as a custom call to the
SparseDenseMatmulGradOpWithOptimizerUpdate op. This op takes the preprocessed
input tensors, embedding table, LaProp hyperparameters (b1, b2, eps), LaProp
states (mu, nu), and the grad as inputs and returns the updated embedding table
and mu, nu values.
"""

import functools
from typing import Tuple

from jax._src import dispatch
from jax._src.lib.mlir import ir
import jax.extend as jex
from jax_tpu_embedding.sparsecore.lib.core.primitives import utils
import numpy as np

tpu_sparse_dense_matmul_grad_with_laprop_primitive = jex.core.Primitive(
    "sparse_dense_matmul_grad_with_laprop_primitive",
)

tpu_sparse_dense_matmul_grad_with_laprop_primitive.multiple_results = True


tpu_sparse_dense_matmul_grad_with_laprop_primitive.def_impl(
    functools.partial(
        dispatch.apply_primitive,
        tpu_sparse_dense_matmul_grad_with_laprop_primitive,
    )
)


def _annotate_sparse_compute_type(op: ir.OpView):
  op.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(
      {"_xla_compute_type": ir.StringAttr.get("sparse")}
  )
  return op


def _tpu_sparse_dense_matmul_grad_with_laprop_abstract_eval(
    lhs_row_pointers: np.ndarray,
    lhs_local_embedding_ids: np.ndarray,
    lhs_local_sample_ids: np.ndarray,
    lhs_gains: np.ndarray,
    embedding_table: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    activations_grad: np.ndarray,
    b1: float,
    b2: float,
    eps: float,
    *_,
    max_ids_per_partition: int,
    max_unique_ids_per_partition: int,
    computation_name: str = "laprop_optimizer_update",
    sharding_strategy: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Abstract eval for sparse_dense_matmul_laprop."""

  utils.validate_abstract_eval_params(
      lhs_row_pointers,
      lhs_local_embedding_ids,
      lhs_local_sample_ids,
      lhs_gains,
      embedding_table,
      activations_grad,
      max_ids_per_partition,
      max_unique_ids_per_partition,
      computation_name,
      sharding_strategy,
  )

  utils.ensure_dtype(b1, np.float32, "b1")
  utils.ensure_dtype(b2, np.float32, "b2")
  utils.ensure_dtype(eps, np.float32, "eps")
  utils.ensure_dtype(mu, np.float32, "mu")
  utils.ensure_dtype(nu, np.float32, "nu")

  if embedding_table.shape != mu.shape:
    raise ValueError(
        "embedding_table and mu must have equal shapes, got"
        f" {embedding_table.shape} and {mu.shape}"
    )
  elif embedding_table.shape != nu.shape:
    raise ValueError(
        "embedding_table and nu must have equal shapes, got"
        f" {embedding_table.shape} and {nu.shape}"
    )

  return embedding_table, mu, nu


tpu_sparse_dense_matmul_grad_with_laprop_primitive.def_abstract_eval(
    _tpu_sparse_dense_matmul_grad_with_laprop_abstract_eval
)
