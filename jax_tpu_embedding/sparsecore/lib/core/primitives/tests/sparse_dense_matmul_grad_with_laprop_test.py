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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import einops
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.core.primitives import sparse_dense_matmul_grad_with_laprop
import numpy as np

# Constants for the test.
_BATCH_SIZE = 16
_VOCAB_SIZE = 32
_EMB_SIZE = 8
_NUM_SC_PER_DEVICE = 4


class SparseDenseMatmulGradWithLapropTest(parameterized.TestCase):
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
  learning_rate = 0.001
  b1 = 0.9
  decay_rate = 0.95
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=1,
          decay_rate=decay_rate,
          eps=eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      ),
      dict(
          testcase_name="decay_rate is not np.float32",
          row_pointers=row_pointers,
          sample_ids=sample_ids,
          embedding_ids=embedding_ids,
          gains=gains,
          embedding_table=embedding_table,
          mu=mu,
          nu=nu,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=1,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          mu=mu.astype(np.int32),
          nu=nu,
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          nu=nu.astype(np.int32),
          activations_grad=activations_grad,
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
          learning_rate=learning_rate,
          b1=b1,
          decay_rate=decay_rate,
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
      learning_rate,
      b1,
      decay_rate,
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
          1,  # num_minibatches_per_physical_sparse_core
          embedding_table,
          mu,
          nu,
          activations_grad,
          learning_rate,
          b1,
          decay_rate,
          eps,
          max_ids_per_partition=max_ids_per_partition,
          max_unique_ids_per_partition=max_unique_ids_per_partition,
      )

  def test_laprop_optimizer_update(self):
    # Process the input.
    input_tensor = np.array(
        [
            [5],
            [3],
            [9],
            [1],
            [6],
            [12],
            [0],
            [4],
            [15],
            [13],
            [11],
            [7],
            [8],
            [14],
            [2],
            [10],
        ],
        dtype=np.int32,
    )
    input_weights = np.array(
        [[1.0] for _ in range(16)],
        dtype=np.float32,
    )
    global_devices = np.array([mock.create_autospec(jax.Device)])
    mesh = jax.sharding.Mesh(global_devices, "x")
    (
        lhs_row_pointers,
        lhs_local_embedding_ids,
        lhs_local_sample_ids,
        lhs_gains,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        input_tensor,
        input_weights,
        mesh,
        max_ids_per_partition=16,
        num_sc_per_device=_NUM_SC_PER_DEVICE,
    )
    emb_table = (
        np.array([[i for _ in range(_EMB_SIZE)] for i in range(_VOCAB_SIZE)])
        .reshape(_VOCAB_SIZE, _EMB_SIZE)
        .astype(np.float32)
    )
    emb_table_sharded = einops.rearrange(
        emb_table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )
    mu_init = jnp.full(
        emb_table_sharded[0].shape,
        0.00,
        np.float32,
    )

    nu_init = jnp.full(
        emb_table_sharded[0].shape,
        0.00,
        np.float32,
    )

    learning_rate = 0.1
    b1 = 0.9
    decay_rate = 0.95
    eps = 1e-8

    z_grad = jnp.full(
        (
            _BATCH_SIZE,
            _EMB_SIZE,
        ),
        0.01,
        np.float32,
    )

    # Expected values are computed manually on colab.
    expected_embedding_table = np.array([
        _EMB_SIZE * [-0.19492617],
        _EMB_SIZE * [0.80507386],
        _EMB_SIZE * [1.8050739],
        _EMB_SIZE * [2.8050737],
        _EMB_SIZE * [3.8050737],
        _EMB_SIZE * [4.8050737],
        _EMB_SIZE * [5.8050737],
        _EMB_SIZE * [6.8050737],
        _EMB_SIZE * [7.8050737],
        _EMB_SIZE * [8.805074],
        _EMB_SIZE * [9.805074],
        _EMB_SIZE * [10.805074],
        _EMB_SIZE * [11.805074],
        _EMB_SIZE * [12.805074],
        _EMB_SIZE * [13.805074],
        _EMB_SIZE * [14.805074],
        _EMB_SIZE * [16],
        _EMB_SIZE * [17],
        _EMB_SIZE * [18],
        _EMB_SIZE * [19],
        _EMB_SIZE * [20],
        _EMB_SIZE * [21],
        _EMB_SIZE * [22],
        _EMB_SIZE * [23],
        _EMB_SIZE * [24],
        _EMB_SIZE * [25],
        _EMB_SIZE * [26],
        _EMB_SIZE * [27],
        _EMB_SIZE * [28],
        _EMB_SIZE * [29],
        _EMB_SIZE * [30],
        _EMB_SIZE * [31],
    ])

    expected_embedding_table = einops.rearrange(
        expected_embedding_table,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )[0]

    # Expected values are computed manually on colab.
    expected_mu = np.array([
        (_VOCAB_SIZE // 2) * _EMB_SIZE * [1.9492617],
        (_VOCAB_SIZE // 2) * _EMB_SIZE * [0.0],
    ]).reshape(_VOCAB_SIZE, _EMB_SIZE)

    expected_mu = einops.rearrange(
        expected_mu,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )[0]

    # Expected values are computed manually on colab.
    expected_nu = np.array([
        (_VOCAB_SIZE // 2) * _EMB_SIZE * [5.0004996e-06],
        (_VOCAB_SIZE // 2) * _EMB_SIZE * [0.0],
    ]).reshape(_VOCAB_SIZE, _EMB_SIZE)

    expected_nu = einops.rearrange(
        expected_nu,
        "(v c s) f -> c (s v) f",
        c=1,
        s=4,
    )[0]

    (updated_table, updated_mu, updated_nu) = (
        sparse_dense_matmul_grad_with_laprop.tpu_sparse_dense_matmul_grad_with_laprop_primitive.bind(
            lhs_row_pointers,
            lhs_local_embedding_ids,
            lhs_local_sample_ids,
            lhs_gains,
            1,  # num_minibatches_per_physical_sparse_core
            emb_table_sharded[0],
            mu_init,
            nu_init,
            z_grad,
            learning_rate,
            b1,
            decay_rate,
            eps,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
            computation_name="optimizer_test_computation",
            sharding_strategy=1,
        )
    )

    np.testing.assert_almost_equal(
        expected_embedding_table, updated_table, decimal=6
    )
    np.testing.assert_almost_equal(expected_mu, updated_mu, decimal=6)
    np.testing.assert_almost_equal(expected_nu, updated_nu, decimal=6)


if __name__ == "__main__":
  absltest.main()
