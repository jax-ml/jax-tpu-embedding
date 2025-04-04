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
from absl.testing import absltest
from absl.testing import parameterized
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_laprop
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithAdagradTest(parameterized.TestCase):
  row_pointers = np.array([0, 1, 2, 4], dtype=np.int32)
  sample_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  embedding_ids = np.array([0, 1, 2, 3], dtype=np.int32)
  gains = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
  embedding_table = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  mu = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
  nu = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
  activations_grad = np.array(
      [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
  )
  b1 = 0.9
  b2 = 0.95
  eps = 1e-8
  max_ids_per_partition = 16
  max_unique_ids_per_partition = 16

  @parameterized.named_parameters(
      dict(
          testcase_name="row_pointers_dtype is not np.int32",
          row_pointers=row_pointers.astype(np.float32),
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="sample_ids_dtype is not np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids.astype(np.float32),
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_ids_dtype is not np.int32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids.astype(np.float32),
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="gains_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains.astype(np.int32),
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table.astype(np.int32),
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="mu is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu.astype(np.int32),
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="nu is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu.astype(np.int32),
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dtype is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad.astype(np.int32),
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="b1 is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="b2 is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=1,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="eps is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=1,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="row_pointers_shape rank is not 1",
          row_pointers=np.array([[0, 1, 2, 3]]),
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_shape doesn't match accumulator shape",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0, 4.0]]),
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="embedding_table_dim is not 2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([1.0, 2.0, 3.0, 4.0]),
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="activations_grad_dim is not 2",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=np.array([1.0, 2.0, 3.0]),
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "embedding_table_activations_width doesn't match grad width"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=np.array([[1.0, 2.0, 3.0, 4.0]]),
          mu=mu,
          nu=nu,
          activations_grad=np.array([[1.0, 2.0, 3.0]]),
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="max_ids_per_partition is less than or equal to 0",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=0,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name=(
              "max_unique_ids_per_partition is less than or equal to 0"
          ),
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          b1=b1,
          b2=b2,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=0,
      ),
  )
  def test_raising_value_error_while_evaluating_abstracts(
      self,
      row_pointers,
      sample_ids,
      embedding_ids,
      gains,
      embedding_table,
      mu,
      nu,
      activations_grad,
      b1,
      b2,
      eps,
      max_ids_per_partition,
      max_unique_ids_per_partition,
  ):
    with self.assertRaises(ValueError):
      sparse_dense_matmul_grad_with_laprop.tpu_sparse_dense_matmul_grad_with_laprop_primitive.bind(
          row_pointers,
          sample_ids,
          embedding_ids,
          gains,
          embedding_table,
          mu,
          nu,
          activations_grad,
          b1,
          b2,
          eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )


if __name__ == "__main__":
  absltest.main()
