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
"""Tests for SparseCore embedding API."""

import logging

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax import sharding
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


# Helpful for detailed debug prints.
np.set_printoptions(threshold=np.inf, suppress=True)

_PARTITION_ERR_STR = (
    "PartitionSpec of the global sharding either needs to be in the format"
)
_DEVICE_ERR_STR = (
    "global_sharding needs to be created with default device order from"
)


# TODO(b/369729914): Refactor according to internal link:unit-testing-practices#
class EmbeddingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])

  def assert_all_shards_shape(self, shards, expected_shape):
    for shard in shards:
      self.assertEqual(shard.data.shape, expected_shape)

  def assert_all_shards_data(self, shards, expected_data):
    for shard in shards:
      np.testing.assert_array_equal(shard.data, expected_data)

  def assert_len_shards(self, shards, expected_len):
    self.assertLen(shards, expected_len)

  def test_get_valid_table_specs(self):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec,
            input_shape=[16, 1],
            output_shape=[16, 16],
            name="feature_a",
        )
    ]
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=self.num_sc_per_device,
    )
    total_sc_in_test = jax.device_count() * self.num_sc_per_device
    padded_vocab_size = test_utils.round_up_to_multiple(
        table_spec.vocabulary_size, 8 * total_sc_in_test
    )
    expected_stacked_table_spec = embedding_spec.StackedTableSpec(
        stack_name=table_spec.name,
        stack_vocab_size=padded_vocab_size,
        stack_embedding_dim=16,
        combiner=table_spec.combiner,
        optimizer=table_spec.optimizer,
        max_ids_per_partition=table_spec.max_ids_per_partition,
        max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
        total_sample_count=16,
    )
    self.assertEqual(
        embedding.get_table_specs(feature_specs),
        {
            table_spec.name: embedding_spec.TableSpec(
                name=table_spec.name,
                embedding_dim=table_spec.embedding_dim,
                vocabulary_size=table_spec.vocabulary_size,
                optimizer=table_spec.optimizer,
                combiner=table_spec.combiner,
                initializer=table_spec.initializer,
                max_ids_per_partition=table_spec.max_ids_per_partition,
                max_unique_ids_per_partition=table_spec.max_unique_ids_per_partition,
                _setting_in_stack=embedding_spec.TableSettingInStack(
                    stack_name=table_spec.name,
                    padded_embedding_dim=table_spec.embedding_dim,
                    padded_vocab_size=padded_vocab_size,
                    row_offset_in_shard=0,
                    shard_rotation=0,
                ),
                stacked_table_spec=expected_stacked_table_spec,
            )
        },
    )
    self.assertEqual(
        embedding.get_stacked_table_specs(feature_specs),
        {table_spec.name: expected_stacked_table_spec},
    )

  def test_get_table_specs_from_duplicated_specs(self):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
    )
    invalid_table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )
    invalid_feature_spec = embedding_spec.FeatureSpec(
        table_spec=invalid_table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
    )
    feature_specs = [feature_spec, invalid_feature_spec]
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=utils.num_sparsecores_per_device(jax.devices()[0]),
    )
    with self.assertRaises(ValueError):
      embedding.get_stacked_table_specs(feature_specs)

    with self.assertRaises(ValueError):
      embedding.get_table_specs(feature_specs)

  def test_manual_stacking_settings_error(self):
    table_spec1 = embedding_spec.TableSpec(
        vocabulary_size=256,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_a",
            padded_vocab_size=32,  # no padding
            padded_embedding_dim=16,  # no padding
            row_offset_in_shard=0,  # first table in stack
            shard_rotation=1,  # rotation of 1
        ),
    )
    table_spec2 = embedding_spec.TableSpec(
        vocabulary_size=42,
        embedding_dim=8,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_b",
            padded_vocab_size=42,  # no padding
            padded_embedding_dim=10,  # padding
            row_offset_in_shard=8,  # second table in stack
            shard_rotation=2,  # rotation of 2
        ),
    )
    feature_spec1 = embedding_spec.FeatureSpec(
        table_spec=table_spec1,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=0,
            col_offset=0,
            col_shift=1,
        ),
    )
    feature_spec2 = embedding_spec.FeatureSpec(
        table_spec=table_spec2,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_b",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=16,
            col_offset=8,
            col_shift=2,
        ),
    )
    with self.assertRaisesRegex(
        ValueError,
        "embedding.prepare_feature_specs_for_training was not called",
    ):
      embedding.get_stacked_table_specs((feature_spec1, feature_spec2))

  def test_prepare_features_for_training_with_feature_stacking(self):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=12,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table",
    )
    feature_spec_a = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
    )
    feature_spec_b = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_b",
    )
    embedding.prepare_feature_specs_for_training(
        (feature_spec_a, feature_spec_b),
        global_device_count=jax.device_count(),
        num_sc_per_device=self.num_sc_per_device,
    )
    total_sc_in_test = jax.device_count() * self.num_sc_per_device
    padded_vocab_size = test_utils.round_up_to_multiple(
        table_spec.vocabulary_size, 8 * total_sc_in_test
    )
    expected_stacked_table_spec = embedding_spec.StackedTableSpec(
        stack_name="table",
        stack_vocab_size=padded_vocab_size,
        stack_embedding_dim=16,  # Dim round up
        combiner="sum",
        optimizer=embedding_spec.SGDOptimizerSpec(),
        max_ids_per_partition=256,
        max_unique_ids_per_partition=256,
        total_sample_count=32,  # Batch stacked, feature stacking
    )
    self.assertEqual(
        feature_spec_a.table_spec.setting_in_stack.stack_name, "table"
    )
    self.assertEqual(
        feature_spec_b.table_spec.setting_in_stack.stack_name, "table"
    )
    self.assertEqual(
        feature_spec_a.table_spec.stacked_table_spec,
        expected_stacked_table_spec,
    )
    self.assertEqual(
        feature_spec_b.table_spec.stacked_table_spec,
        expected_stacked_table_spec,
    )

  def test_manual_stacking_settings(self):
    stack_table_spec = embedding_spec.StackedTableSpec(
        stack_name="table_a_table_b",
        stack_vocab_size=74,
        stack_embedding_dim=16,
        combiner="sum",
        optimizer=embedding_spec.SGDOptimizerSpec(),
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        total_sample_count=32,
    )
    table_spec1 = embedding_spec.TableSpec(
        vocabulary_size=32,
        embedding_dim=16,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_a_table_b",
            padded_vocab_size=32,  # no padding
            padded_embedding_dim=16,  # no padding
            row_offset_in_shard=0,  # first table in stack
            shard_rotation=1,  # rotation of 1
        ),
        stacked_table_spec=stack_table_spec,
    )
    table_spec2 = embedding_spec.TableSpec(
        vocabulary_size=42,
        embedding_dim=8,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
        _setting_in_stack=embedding_spec.TableSettingInStack(
            stack_name="table_a_table_b",
            padded_vocab_size=42,  # no padding
            padded_embedding_dim=16,  # padding
            row_offset_in_shard=8,  # second table in stack
            shard_rotation=2,  # rotation of 2
        ),
        stacked_table_spec=stack_table_spec,
    )
    feature_spec1 = embedding_spec.FeatureSpec(
        table_spec=table_spec1,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_a",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=0,
            col_offset=0,
            col_shift=1,
        ),
    )
    feature_spec2 = embedding_spec.FeatureSpec(
        table_spec=table_spec2,
        input_shape=[16, 1],
        output_shape=[16, 16],
        name="feature_b",
        _id_transformation=embedding_spec.FeatureIdTransformation(
            row_offset=16,
            col_offset=8,
            col_shift=2,
        ),
    )
    feature_specs = [feature_spec1, feature_spec2]
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=utils.num_sparsecores_per_device(jax.devices()[0]),
    )
    self.assertEqual(
        embedding.get_stacked_table_specs(feature_specs),
        {
            "table_a_table_b": embedding_spec.StackedTableSpec(
                stack_name="table_a_table_b",
                stack_vocab_size=74,
                stack_embedding_dim=16,
                combiner="sum",
                optimizer=embedding_spec.SGDOptimizerSpec(),
                max_ids_per_partition=16,
                max_unique_ids_per_partition=16,
                total_sample_count=32,
            )
        },
    )

  @parameterized.parameters(
      (embedding_spec.SGDOptimizerSpec()),
      (embedding_spec.AdagradOptimizerSpec(initial_accumulator_value=0.0)),
  )
  def test_init_embedding_variables_default(self, optimizer_spec):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, "x")
    # The embedding tables vocab, which is the second dimension is sharded on
    # devices.
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("x", None)
    )

    emb_var_count = optimizer_spec.slot_variables_count() + 1
    vocab_size_a = 32
    vocab_size_b = 64
    embedding_dim = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_b",
    )

    embedding_variables = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        [table_a_spec, table_b_spec],
        global_sharding=global_sharding,
        num_sparsecore_per_device=utils.num_sparsecores_per_device(devices[0]),
    )

    self.assertLen(
        jax.tree.leaves(embedding_variables["table_a"]), emb_var_count
    )
    for variable in jax.tree.leaves(embedding_variables["table_a"]):
      self.assertEqual(variable.shape, (vocab_size_a, embedding_dim))

      self.assertEqual(
          len(variable.addressable_shards),
          len(devices),
      )

      for shard in variable.addressable_shards:
        self.assertEqual(
            shard.data.shape,
            (vocab_size_a // len(devices), embedding_dim),
        )

    self.assertLen(
        jax.tree.leaves(embedding_variables["table_b"]), emb_var_count
    )
    for variable in jax.tree.leaves(embedding_variables["table_b"]):
      self.assertEqual(variable.shape, (vocab_size_b, embedding_dim))

      self.assertEqual(
          len(variable.addressable_shards),
          len(devices),
      )

      for shard in variable.addressable_shards:
        self.assertEqual(
            shard.data.shape,
            (vocab_size_b // len(devices), embedding_dim),
        )

  @parameterized.parameters(
      (embedding_spec.SGDOptimizerSpec()),
      (embedding_spec.AdagradOptimizerSpec(initial_accumulator_value=0.0)),
  )
  def test_init_embedding_variables_for_pmap(self, optimizer_spec):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, "x")
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("x", None, None)
    )

    emb_var_count = optimizer_spec.slot_variables_count() + 1
    vocab_size_a = 32
    vocab_size_b = 128
    embedding_dim = 14
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(0.0),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(1.0),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_b",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_b",
    )
    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )
    # Assert on the preconditions.
    self.assertLen(feature_specs, 2)
    total_sc_in_test = self.num_sc_per_device * jax.device_count()
    padded_vocab_size_a = test_utils.round_up_to_multiple(
        vocab_size_a, 8 * total_sc_in_test
    )
    self.assertEqual(
        feature_a_spec.table_spec.setting_in_stack.padded_vocab_size,
        padded_vocab_size_a,
    )
    padded_vocab_size_b = test_utils.round_up_to_multiple(
        vocab_size_b, 8 * total_sc_in_test
    )
    self.assertEqual(
        feature_b_spec.table_spec.setting_in_stack.padded_vocab_size,
        padded_vocab_size_b,
    )
    updated_table_specs = [f.table_spec for f in feature_specs]

    # Create embedding variables with stacked features.
    embedding_variables = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        updated_table_specs,
        global_sharding=global_sharding,
        num_sparsecore_per_device=self.num_sc_per_device,
    )

    expected_sc_shard_a = jax.numpy.zeros([8, 16]).reshape([1, -1, 16])
    expected_sc_shard_b = jax.numpy.ones([8, 16]).reshape([1, -1, 16])
    expected_stack_sc_shard = jax.numpy.concatenate(
        [expected_sc_shard_a, expected_sc_shard_b], axis=1
    )
    expected_device_shard = jax.numpy.concatenate(
        [expected_stack_sc_shard] * self.num_sc_per_device, axis=1
    )

    self.assertLen(
        jax.tree.leaves(embedding_variables["table_a_table_b"]), emb_var_count
    )
    for variable in jax.tree.leaves(embedding_variables):
      self.assertEqual(
          variable.shape,
          (
              jax.device_count(),
              (padded_vocab_size_a + padded_vocab_size_b) // len(devices),
              16,
          ),
      )
      self.assertEqual(
          len(variable.addressable_shards),
          len(devices),
      )

    self.assert_all_shards_shape(
        embedding_variables["table_a_table_b"].table.addressable_shards,
        (
            1,
            (padded_vocab_size_a + padded_vocab_size_b) // len(devices),
            16,
        ),
    )
    self.assert_all_shards_data(
        embedding_variables["table_a_table_b"].table.addressable_shards,
        expected_device_shard,
    )

  @parameterized.parameters(
      (embedding_spec.SGDOptimizerSpec()),
      (embedding_spec.AdagradOptimizerSpec(initial_accumulator_value=0.0)),
  )
  def test_init_embedding_variables_stacking_for_jit(self, optimizer_spec):
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, "x")
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("x", None)
    )

    emb_var_count = optimizer_spec.slot_variables_count() + 1
    vocab_size_a = 32
    vocab_size_b = 128
    embedding_dim = 16
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(0.0),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(1.0),
        optimizer=optimizer_spec,
        combiner="sum",
        name="table_b",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_b",
    )
    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )
    # Assert on the preconditions.
    self.assertLen(feature_specs, 2)
    total_sc_in_test = self.num_sc_per_device * jax.device_count()
    padded_vocab_size_a = test_utils.round_up_to_multiple(
        vocab_size_a, 8 * total_sc_in_test
    )
    self.assertEqual(
        feature_a_spec.table_spec.setting_in_stack.padded_vocab_size,
        padded_vocab_size_a,
    )
    padded_vocab_size_b = test_utils.round_up_to_multiple(
        vocab_size_b, 8 * total_sc_in_test
    )
    self.assertEqual(
        feature_b_spec.table_spec.setting_in_stack.padded_vocab_size,
        padded_vocab_size_b,
    )
    updated_table_specs = [f.table_spec for f in feature_specs]

    # Create embedding variables with stacked features.
    embedding_variables = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        updated_table_specs,
        global_sharding=global_sharding,
        num_sparsecore_per_device=self.num_sc_per_device,
    )
    expected_sc_shard_a = jax.numpy.zeros([8, 16])
    expected_sc_shard_b = jax.numpy.ones([8, 16])
    expected_stack_sc_shard = jax.numpy.concatenate(
        [expected_sc_shard_a, expected_sc_shard_b], axis=0
    )
    expected_device_shard = jax.numpy.concatenate(
        [expected_stack_sc_shard] * self.num_sc_per_device, axis=0
    )

    self.assertLen(
        jax.tree.leaves(embedding_variables["table_a_table_b"]), emb_var_count
    )
    for variable in jax.tree.leaves(embedding_variables):
      self.assertEqual(
          variable.shape,
          (padded_vocab_size_a + padded_vocab_size_b, embedding_dim),
      )

      self.assertEqual(
          len(variable.addressable_shards),
          len(devices),
      )

      self.assert_all_shards_shape(
          variable.addressable_shards,
          (
              (padded_vocab_size_a + padded_vocab_size_b) // len(devices),
              16,
          ),
      )
    self.assert_all_shards_data(
        embedding_variables["table_a_table_b"].table.addressable_shards,
        expected_device_shard,
    )

  @parameterized.parameters(
      ((None,),),
      ((None, "x"),),
      ((None, "x", None),),
      ((None, None, "x"),),
      (("x", None, None, None),),
      ((None, None, None, "x"),),
      ((None, "x", None, None),),
      ((None, None, "x", None),),
  )
  def test_malformated_partition_for_init_embedding_variables(self, pspec):
    mesh = jax.sharding.Mesh(jax.devices(), "x")
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec(*pspec)
    )

    vocab_size_a = 32
    embedding_dim = 16
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_a_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )

    updated_table_specs = [f.table_spec for f in feature_specs]

    with self.assertRaisesRegex(
        ValueError,
        _PARTITION_ERR_STR,
    ):
      # Create embedding variables with stacked features.
      _ = embedding.init_embedding_variables(
          jax.random.PRNGKey(0),
          updated_table_specs,
          global_sharding=global_sharding,
          num_sparsecore_per_device=self.num_sc_per_device,
      )

  def test_non_default_device_mesh_for_init_embedding_variables(self):
    # create non-standard device list
    devices = jax.devices()
    popped_device = devices.pop(jax.device_count() // 2)
    devices.append(popped_device)

    mesh = jax.sharding.Mesh(devices, "x")
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("x", None)
    )

    vocab_size_a = 32
    embedding_dim = 16
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_a_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )

    updated_table_specs = [f.table_spec for f in feature_specs]

    with self.assertRaisesRegex(
        ValueError,
        _DEVICE_ERR_STR,
    ):
      # Create embedding variables with stacked features.
      _ = embedding.init_embedding_variables(
          jax.random.PRNGKey(0),
          updated_table_specs,
          global_sharding=global_sharding,
          bypass_mesh_check=False,
          num_sparsecore_per_device=self.num_sc_per_device,
      )

    # Call the method again with bypass_mesh_check and it should succeed.
    _ = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        updated_table_specs,
        global_sharding=global_sharding,
        bypass_mesh_check=True,
        num_sparsecore_per_device=self.num_sc_per_device,
    )

  def test_nd_mesh_for_init_embedding_variables(self):
    devices = np.array(jax.devices()).reshape(2, -1)
    mesh = jax.sharding.Mesh(devices, ("x", "y"))
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec(("x", "y"), None)
    )

    vocab_size_a = 32
    vocab_size_b = 77
    embedding_dim = 14
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_b",
    )

    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )
    updated_table_specs = [f.table_spec for f in feature_specs]

    # Create embedding variables with stacked features.
    embedding_variables = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        updated_table_specs,
        global_sharding=global_sharding,
        num_sparsecore_per_device=self.num_sc_per_device,
    )

    table_spec = updated_table_specs[0]
    assert table_spec.stacked_table_spec is not None
    self.assertEqual(
        embedding_variables["table_a_table_b"][0].shape,
        (
            table_spec.stacked_table_spec.stack_vocab_size,
            table_spec.stacked_table_spec.stack_embedding_dim,
        ),
    )
    self.assertGreaterEqual(
        table_spec.stacked_table_spec.stack_vocab_size,
        vocab_size_a + vocab_size_b,
    )
    self.assertGreaterEqual(
        table_spec.stacked_table_spec.stack_embedding_dim, embedding_dim
    )
    self.assertEqual(table_spec.stacked_table_spec.stack_embedding_dim % 8, 0)

  def test_muti_dimensional_mesh_for_init_embedding_variables(self):
    # create non-standard device list
    devices = np.array(jax.devices()).reshape(1, 1, -1, 1)

    mesh = jax.sharding.Mesh(devices, ("a", "b", "c", "d"))
    global_sharding = sharding.NamedSharding(
        mesh, sharding.PartitionSpec("c", None)
    )

    vocab_size_a = 32
    vocab_size_b = 77
    embedding_dim = 14
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.truncated_normal(),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_b",
    )

    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )
    updated_table_specs = [f.table_spec for f in feature_specs]

    # Create embedding variables with stacked features.
    embedding_variables = embedding.init_embedding_variables(
        jax.random.PRNGKey(0),
        updated_table_specs,
        global_sharding=global_sharding,
        num_sparsecore_per_device=self.num_sc_per_device,
    )

    table_spec = updated_table_specs[0]
    assert table_spec.stacked_table_spec is not None
    self.assertEqual(
        embedding_variables["table_a_table_b"][0].shape,
        (
            table_spec.stacked_table_spec.stack_vocab_size,
            table_spec.stacked_table_spec.stack_embedding_dim,
        ),
    )
    self.assertGreaterEqual(
        table_spec.stacked_table_spec.stack_vocab_size,
        vocab_size_a + vocab_size_b,
    )
    self.assertGreaterEqual(
        table_spec.stacked_table_spec.stack_embedding_dim, embedding_dim
    )
    self.assertEqual(table_spec.stacked_table_spec.stack_embedding_dim % 8, 0)

  def test_create_proto_from_feature_specs(self):
    vocab_size_a = 32
    vocab_size_b = 128
    embedding_dim = 14
    batch_size = 16

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(0.0),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_a",
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(1.0),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner="sum",
        name="table_b",
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_a",
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name="feature_b",
    )
    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    embedding.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=jax.device_count(),
    )
    expected_proto = embedding_spec_pb2.EmbeddingSpecProto()
    num_sparsecores = self.num_sc_per_device * jax.device_count()
    padded_vocab_size_a = test_utils.round_up_to_multiple(
        feature_a_spec.table_spec.vocabulary_size, 8 * num_sparsecores
    )
    padded_vocab_size_b = test_utils.round_up_to_multiple(
        feature_b_spec.table_spec.vocabulary_size, 8 * num_sparsecores
    )
    stack_vocab_size = padded_vocab_size_a + padded_vocab_size_b
    text_format.Parse(
        f"""stacked_table_specs {{
        stack_name: "table_a_table_b"
        stack_vocab_size: {stack_vocab_size}
        stack_embedding_dim: 16
        total_sample_count: 32
        max_ids_per_partition: 256
        max_unique_ids_per_partition: 256
        num_sparsecores: {num_sparsecores}
        table_specs {{
          table_name: "table_a"
          vocab_size: 32
          embedding_dim: 14
          padded_vocab_size: {padded_vocab_size_a}
          padded_embedding_dim: 16
          row_offset_in_shard: 0
          shard_rotation: 0
          feature_specs {{
            feature_name: "feature_a"
            input_shape: 16
            input_shape: 1
            output_shape: 16
            output_shape: 14
            row_offset: 0
            col_offset: 0
            col_shift: 0
          }}
        }}
        table_specs {{
          table_name: "table_b"
          vocab_size: 128
          embedding_dim: 14
          padded_vocab_size: {padded_vocab_size_b}
          padded_embedding_dim: 16
          row_offset_in_shard: 8
          shard_rotation: {self.num_sc_per_device}
          feature_specs {{
            feature_name: "feature_b"
            input_shape: 16
            input_shape: 1
            output_shape: 16
            output_shape: 14
            row_offset: 16
            col_offset: {padded_vocab_size_a}
            col_shift: {self.num_sc_per_device}
          }}
        }}
      }}""",
        expected_proto,
    )
    actual = embedding.create_proto_from_feature_specs(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sparsecore_per_device=self.num_sc_per_device,
    )
    logging.info("actual =%s\n", actual)
    self.assertEqual(
        text_format.MessageToString(expected_proto),
        text_format.MessageToString(actual),
    )


if __name__ == "__main__":
  absltest.main()
