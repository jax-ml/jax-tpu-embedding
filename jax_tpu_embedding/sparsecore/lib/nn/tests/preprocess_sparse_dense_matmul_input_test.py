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
import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.core import input_preprocessing
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np


class PreprocessSparseDenseMatmulInputTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=8,
        initializer=lambda: jnp.zeros((32, 8), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="table_a",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_a",
            padded_vocab_size=48,
            padded_embedding_dim=8,
            row_offset_in_shard=0,
            shard_rotation=0,
        ),
        stacked_table_spec=embedding_spec.StackedTableSpec(
            stack_name="table_a",
            stack_vocab_size=48,
            stack_embedding_dim=8,
            optimizer=embedding_spec.SGDOptimizerSpec(
                learning_rate=0.001,
            ),
            combiner="sum",
            total_sample_count=8,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
        ),
    )
    self.table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=16,
        embedding_dim=8,
        initializer=lambda: jnp.zeros((16, 8), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(
            learning_rate=0.001,
        ),
        combiner="sum",
        name="table_b",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_b",
            padded_vocab_size=48,
            padded_embedding_dim=8,
            row_offset_in_shard=0,
            shard_rotation=0,
        ),
        stacked_table_spec=embedding_spec.StackedTableSpec(
            stack_name="table_b",
            stack_vocab_size=48,
            stack_embedding_dim=8,
            optimizer=embedding_spec.SGDOptimizerSpec(
                learning_rate=0.001,
            ),
            combiner="sum",
            total_sample_count=8,
            max_ids_per_partition=16,
            max_unique_ids_per_partition=16,
        ),
    )
    self.feature_a_input = np.array(
        [
            np.array([5, 18], dtype=np.int32),
            np.array([0, 2, 31], dtype=np.int32),
            np.array([18, 0, 20, 6], dtype=np.int32),
            np.array([1, 28, 5, 8], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([12, 7, 3, 11], dtype=np.int32),
            np.array([18, 0, 7, 3], dtype=np.int32),
            np.array([6, 4, 19, 2], dtype=np.int32),
        ],
        dtype=object,
    )
    self.feature_b_input = np.array(
        [
            [2, 4, 6, 8],
            [10, 12, 14, 14],
            [1, 3, 5, 7],
            [9, 11, 13, 15],
            [3, 4, 5, 6],
            [7, 8, 9, 10],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=np.int32,
    )
    self.feature_spec_a = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=[len(self.feature_a_input), 4],
        output_shape=[
            len(self.feature_a_input),
            self.table_spec_a.embedding_dim,
        ],
        name="feature_spec_a",
    )
    self.feature_spec_b = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_b,
        input_shape=[len(self.feature_b_input), 4],
        output_shape=[
            len(self.feature_b_input),
            self.table_spec_b.embedding_dim,
        ],
        name="feature_spec_b",
    )
    self.input_weights_a = np.array(
        [
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ],
        dtype=object,
    )
    self.input_weights_b = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    self.global_devices = np.array([
        mock.create_autospec(jax.Device),
        mock.create_autospec(jax.Device),
    ])

    self.mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    self.mesh.size = 2
    self.local_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    self.local_mesh.size = 2
    self.mesh.devices = self.global_devices
    self.mesh.local_mesh = self.local_mesh

  def test_preprocess_static_buffer_size_multiplier(self):
    multiplier = 32
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features={
            "feature_b": self.feature_b_input,
        },
        features_weights={
            "feature_b": self.input_weights_b,
        },
        feature_specs={
            "feature_b": self.feature_spec_b,
        },
        local_device_count=1,
        global_device_count=1,
        static_buffer_size_multiplier=multiplier,
        num_sc_per_device=4,
        sharding_strategy="MOD",
    )
    self.assertLen(preprocessed_input.lhs_row_pointers, 1)
    self.assertLen(preprocessed_input.lhs_embedding_ids, 1)
    self.assertLen(preprocessed_input.lhs_sample_ids, 1)
    self.assertLen(preprocessed_input.lhs_gains, 1)
    self.assertLen(
        preprocessed_input.lhs_embedding_ids[
            self.feature_spec_b.table_spec.name
        ],
        len(self.feature_b_input) * multiplier,
    )
    self.assertLen(
        preprocessed_input.lhs_sample_ids[self.feature_spec_b.table_spec.name],
        len(self.feature_b_input) * multiplier,
    )
    self.assertLen(
        preprocessed_input.lhs_gains[self.feature_spec_b.table_spec.name],
        len(self.feature_b_input) * multiplier,
    )

  def test_preprocess_for_single_feature_single_device(self):
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features={
            "feature_b": self.feature_b_input,
        },
        features_weights={
            "feature_b": self.input_weights_b,
        },
        feature_specs={
            "feature_b": self.feature_spec_b,
        },
        local_device_count=1,
        global_device_count=1,
        num_sc_per_device=4,
        sharding_strategy="MOD",
    )
    self.assertLen(preprocessed_input.lhs_row_pointers, 1)
    self.assertLen(preprocessed_input.lhs_embedding_ids, 1)
    self.assertLen(preprocessed_input.lhs_sample_ids, 1)
    self.assertLen(preprocessed_input.lhs_gains, 1)
    temp_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    temp_mesh.size = 1
    tmp_local_mesh = mock.create_autospec(jax.sharding.Mesh, instance=True)
    tmp_local_mesh.size = 1
    temp_mesh.devices = self.global_devices[:1]
    temp_mesh.local_mesh = tmp_local_mesh
    (
        first_half_b_row_pointers,
        first_half_b_embedding_ids,
        first_half_b_sample_ids,
        first_half_b_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_b_input,
        self.input_weights_b,
        temp_mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_row_pointers["table_b"],
        np.concatenate((first_half_b_row_pointers, [])),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_embedding_ids["table_b"],
        np.concatenate((first_half_b_embedding_ids, [])),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_sample_ids["table_b"],
        np.concatenate((first_half_b_sample_ids, [])),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_gains["table_b"],
        np.concatenate((first_half_b_weights, [])),
    )

  def test_preprocess_sparse_dense_matmul_input_for_two_features(self):
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features={
            "feature_b": self.feature_b_input,
            "feature_a": self.feature_a_input,
        },
        features_weights={
            "feature_a": self.input_weights_a,
            "feature_b": self.input_weights_b,
        },
        feature_specs={
            "feature_b": self.feature_spec_b,
            "feature_a": self.feature_spec_a,
        },
        local_device_count=2,
        global_device_count=2,
        num_sc_per_device=4,
        sharding_strategy="MOD",
    )
    self.assertLen(preprocessed_input.lhs_row_pointers, 2)
    self.assertLen(preprocessed_input.lhs_embedding_ids, 2)
    self.assertLen(preprocessed_input.lhs_sample_ids, 2)
    self.assertLen(preprocessed_input.lhs_gains, 2)

    (
        first_half_a_row_pointers,
        first_half_a_embedding_ids,
        first_half_a_sample_ids,
        first_half_a_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_a_input[:4],
        self.input_weights_a[:4],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )
    (
        second_half_a_row_pointers,
        second_half_a_embedding_ids,
        second_half_a_sample_ids,
        second_half_a_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_a_input[4:],
        self.input_weights_a[4:],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )

    (
        first_half_b_row_pointers,
        first_half_b_embedding_ids,
        first_half_b_sample_ids,
        first_half_b_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_b_input[:4],
        self.input_weights_b[:4],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )
    (
        second_half_b_row_pointers,
        second_half_b_embedding_ids,
        second_half_b_sample_ids,
        second_half_b_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_b_input[4:],
        self.input_weights_b[4:],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_row_pointers["table_a"],
        np.concatenate((first_half_a_row_pointers, second_half_a_row_pointers)),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_row_pointers["table_b"],
        np.concatenate((first_half_b_row_pointers, second_half_b_row_pointers)),
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_sample_ids["table_a"],
        np.concatenate((first_half_a_sample_ids, second_half_a_sample_ids)),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_sample_ids["table_b"],
        np.concatenate((first_half_b_sample_ids, second_half_b_sample_ids)),
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_embedding_ids["table_a"],
        np.concatenate(
            (first_half_a_embedding_ids, second_half_a_embedding_ids)
        ),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_embedding_ids["table_b"],
        np.concatenate(
            (first_half_b_embedding_ids, second_half_b_embedding_ids)
        ),
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_gains["table_a"],
        np.concatenate((first_half_a_weights, second_half_a_weights)),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_gains["table_b"],
        np.concatenate((first_half_b_weights, second_half_b_weights)),
    )

  def test_preprocess_sparse_dense_matmul_input_for_two_features_with_leading_dim(
      self,
  ):
    preprocessed_input, _ = embedding.preprocess_sparse_dense_matmul_input(
        features={
            "feature_b": self.feature_b_input,
            "feature_a": self.feature_a_input,
        },
        features_weights={
            "feature_a": self.input_weights_a,
            "feature_b": self.input_weights_b,
        },
        feature_specs={
            "feature_b": self.feature_spec_b,
            "feature_a": self.feature_spec_a,
        },
        local_device_count=2,
        global_device_count=2,
        num_sc_per_device=4,
        sharding_strategy="MOD",
        has_leading_dimension=True,
    )

    self.assertLen(preprocessed_input.lhs_row_pointers, 2)
    self.assertLen(preprocessed_input.lhs_embedding_ids, 2)
    self.assertLen(preprocessed_input.lhs_sample_ids, 2)
    self.assertLen(preprocessed_input.lhs_gains, 2)

    (
        first_half_a_row_pointers,
        first_half_a_embedding_ids,
        first_half_a_sample_ids,
        first_half_a_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_a_input[:4],
        self.input_weights_a[:4],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )
    (
        second_half_a_row_pointers,
        second_half_a_embedding_ids,
        second_half_a_sample_ids,
        second_half_a_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_a_input[4:],
        self.input_weights_a[4:],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )

    (
        first_half_b_row_pointers,
        first_half_b_embedding_ids,
        first_half_b_sample_ids,
        first_half_b_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_b_input[:4],
        self.input_weights_b[:4],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )
    (
        second_half_b_row_pointers,
        second_half_b_embedding_ids,
        second_half_b_sample_ids,
        second_half_b_weights,
    ) = input_preprocessing.preprocess_sparse_dense_matmul_input(
        self.feature_b_input[4:],
        self.input_weights_b[4:],
        self.mesh,
        max_ids_per_partition=16,
        num_sc_per_device=4,
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_row_pointers["table_a"],
        np.array([first_half_a_row_pointers, second_half_a_row_pointers])
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_row_pointers["table_b"],
        np.array([first_half_b_row_pointers, second_half_b_row_pointers])
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_sample_ids["table_a"],
        np.array([first_half_a_sample_ids, second_half_a_sample_ids])
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_sample_ids["table_b"],
        np.array([first_half_b_sample_ids, second_half_b_sample_ids])
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_embedding_ids["table_a"],
        np.array(
            [first_half_a_embedding_ids, second_half_a_embedding_ids]
        ),
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_embedding_ids["table_b"],
        np.array(
            [first_half_b_embedding_ids, second_half_b_embedding_ids]
        ),
    )

    np.testing.assert_equal(
        preprocessed_input.lhs_gains["table_a"],
        np.array([first_half_a_weights, second_half_a_weights])
    )
    np.testing.assert_equal(
        preprocessed_input.lhs_gains["table_b"],
        np.array([first_half_b_weights, second_half_b_weights])
    )


if __name__ == "__main__":
  absltest.main()
