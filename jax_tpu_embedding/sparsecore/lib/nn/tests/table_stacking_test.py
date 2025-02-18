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
"""Tests for embeddingtable stacking."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils


class TableStackingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_sc_per_device = utils.num_sparsecores_per_device(jax.devices()[0])

  def test_no_stacking(self):
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.assertEqual(table_spec.setting_in_stack.stack_name, 'table_a')
    self.assertEqual(table_spec.setting_in_stack.padded_vocab_size, 64)
    self.assertEqual(table_spec.setting_in_stack.padded_embedding_dim, 12)
    self.assertEqual(table_spec.setting_in_stack.row_offset_in_shard, 0)
    self.assertEqual(table_spec.setting_in_stack.shard_rotation, 0)

  @parameterized.parameters(
      dict(device_count=1),
      dict(device_count=2),
      dict(device_count=4),
      dict(device_count=-1),  # All.
  )
  def test_auto_stack_two_tables(self, device_count: int):
    if device_count < 0:
      device_count = jax.device_count()

    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_b',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec_a,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_a.embedding_dim,
            ),
            name='feature_spec_a',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_spec_b',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_spec_c',
        ),
    ]
    table_stacking.auto_stack_tables(
        feature_specs,
        global_device_count=device_count,
        num_sc_per_device=self.num_sc_per_device,
    )
    self.assertLen(feature_specs, 3)
    feature_a, feature_b, feature_c = feature_specs
    total_sc_in_test = device_count * self.num_sc_per_device
    expected_padded_vocab_a = test_utils.round_up_to_multiple(
        feature_a.table_spec.vocabulary_size, 8 * total_sc_in_test
    )
    expected_padded_vocab_b = test_utils.round_up_to_multiple(
        feature_b.table_spec.vocabulary_size, 8 * total_sc_in_test
    )

    self.assertEqual(
        feature_a.table_spec.setting_in_stack.stack_name, 'table_a_table_b'
    )
    # Round up padded vocab size to next multiple of 8 * total_sc_in_test
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab_a,
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_embedding_dim, 16
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.row_offset_in_shard, 0
    )
    self.assertEqual(feature_a.id_transformation.row_offset, 0)
    self.assertEqual(feature_a.id_transformation.col_offset, 0)
    self.assertEqual(feature_a.id_transformation.col_shift, 0)

    self.assertEqual(
        feature_b.table_spec.setting_in_stack.stack_name, 'table_a_table_b'
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab_b,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.row_offset_in_shard,
        expected_padded_vocab_a // total_sc_in_test,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab_b,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_embedding_dim, 16
    )
    self.assertEqual(feature_b.id_transformation.row_offset, 16)
    self.assertEqual(
        feature_b.id_transformation.col_offset,
        feature_b.table_spec.setting_in_stack.row_offset_in_shard
        * self.num_sc_per_device
        * device_count,
    )
    self.assertEqual(feature_c.id_transformation.row_offset, 16 + 16)
    self.assertEqual(
        feature_c.id_transformation.col_offset,
        feature_c.table_spec.setting_in_stack.row_offset_in_shard
        * self.num_sc_per_device
        * device_count,
    )

  @parameterized.parameters(
      dict(device_count=1),
      dict(device_count=2),
      dict(device_count=4),
      dict(device_count=-1),  # All.
  )
  def test_auto_stack_two_tables_adagrad(self, device_count: int):
    if device_count < 0:
      device_count = jax.device_count()

    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.5, initial_accumulator_value=1.0
        ),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.AdagradOptimizerSpec(
            learning_rate=0.5, initial_accumulator_value=1.0
        ),
        combiner='sum',
        name='table_b',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec_a,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_a.embedding_dim,
            ),
            name='feature_spec_a',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_spec_b',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_spec_c',
        ),
    ]
    table_stacking.auto_stack_tables(
        feature_specs,
        global_device_count=device_count,
        num_sc_per_device=self.num_sc_per_device,
    )
    self.assertLen(feature_specs, 3)
    feature_a, feature_b, feature_c = feature_specs
    total_sc_in_test = device_count * self.num_sc_per_device
    expected_padded_vocab_a = test_utils.round_up_to_multiple(
        feature_a.table_spec.vocabulary_size, 8 * total_sc_in_test
    )
    expected_padded_vocab_b = test_utils.round_up_to_multiple(
        feature_b.table_spec.vocabulary_size, 8 * total_sc_in_test
    )

    self.assertEqual(
        feature_a.table_spec.setting_in_stack.stack_name,
        'table_a_table_b',
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_vocab_size,
        test_utils.round_up_to_multiple(
            feature_a.table_spec.vocabulary_size, 8 * total_sc_in_test
        ),
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.row_offset_in_shard,
        0,
    )
    self.assertEqual(feature_a.id_transformation.row_offset, 0)
    self.assertEqual(feature_a.id_transformation.col_offset, 0)
    self.assertEqual(feature_a.id_transformation.col_shift, 0)

    self.assertEqual(
        feature_b.table_spec.setting_in_stack.stack_name,
        'table_a_table_b',
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.row_offset_in_shard,
        expected_padded_vocab_a // total_sc_in_test,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab_b,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(feature_b.id_transformation.row_offset, 16)
    self.assertEqual(
        feature_b.id_transformation.col_offset,
        feature_b.table_spec.setting_in_stack.row_offset_in_shard
        * self.num_sc_per_device
        * device_count,
    )
    self.assertEqual(feature_c.id_transformation.row_offset, 16 + 16)
    self.assertEqual(
        feature_c.id_transformation.col_offset,
        feature_c.table_spec.setting_in_stack.row_offset_in_shard
        * self.num_sc_per_device
        * device_count,
    )

  @parameterized.parameters(
      dict(device_count=1),
      dict(device_count=2),
      dict(device_count=4),
      dict(device_count=-1),  # All.
  )
  def test_manual_stacking(self, device_count: int):
    if device_count < 0:
      device_count = jax.device_count()

    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.5),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_b',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_c = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=16,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_c',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec_a,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_a.embedding_dim,
            ),
            name='feature_spec_a',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_spec_b',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_c,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_c.embedding_dim,
            ),
            name='feature_spec_c',
        ),
    ]

    table_stacking.stack_tables(
        feature_specs,
        ['table_a'],
        global_device_count=device_count,
        num_sc_per_device=self.num_sc_per_device,
    )

    def limits(name: str, batch_size: int) -> int:
      del name
      return int(batch_size * 0.8)

    table_stacking.stack_tables(
        feature_specs,
        ['table_c', 'table_b'],
        global_device_count=device_count,
        stack_to_max_ids_per_partition=limits,
        stack_to_max_unique_ids_per_partition=limits,
        num_sc_per_device=self.num_sc_per_device,
    )
    self.assertLen(feature_specs, 3)
    updated_feature_a, updated_feature_b, updated_feature_c = feature_specs
    self.assertEqual(
        updated_feature_a.table_spec.setting_in_stack.stack_name,
        'table_a',
    )
    self.assertEqual(
        updated_feature_a.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.stack_name,
        'table_c_table_b',
    )
    self.assertEqual(
        updated_feature_c.table_spec.setting_in_stack.stack_name,
        'table_c_table_b',
    )
    total_sc_in_test = device_count * self.num_sc_per_device
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.padded_vocab_size,
        test_utils.round_up_to_multiple(
            updated_feature_b.table_spec.vocabulary_size,
            8 * total_sc_in_test,
        ),
    )
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.shard_rotation,
        self.num_sc_per_device % total_sc_in_test,
    )
    assert updated_feature_b.table_spec.stacked_table_spec is not None
    self.assertEqual(
        updated_feature_b.table_spec.stacked_table_spec.max_ids_per_partition,
        25,
    )
    self.assertEqual(
        updated_feature_b.table_spec.stacked_table_spec.max_unique_ids_per_partition,
        25,
    )
    updated_feature_b.table_spec.stacked_table_spec.optimizer.learning_rate = (
        0.5
    )

  @parameterized.parameters(
      dict(device_count=1),
      dict(device_count=2),
      dict(device_count=4),
      dict(device_count=-1),  # All.
  )
  def test_manual_stacking_reuse_table_name(self, device_count: int):
    if device_count < 0:
      device_count = jax.device_count()

    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.5),
        combiner='sum',
        name='A',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.5),
        combiner='sum',
        name='B',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_c = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.5),
        combiner='sum',
        name='C',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec_a,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_a.embedding_dim,
            ),
            name='feature_a',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_b',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_c,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_c.embedding_dim,
            ),
            name='feature_c',
        ),
    ]

    table_stacking.stack_tables(
        feature_specs,
        ('A', 'B'),
        global_device_count=device_count,
        num_sc_per_device=self.num_sc_per_device,
        stack_name='custom_stack',
    )

    with self.assertRaisesRegex(ValueError, 'custom_stack.*already used.*'):
      table_stacking.stack_tables(
          feature_specs,
          ('C',),
          global_device_count=device_count,
          num_sc_per_device=self.num_sc_per_device,
          stack_name='custom_stack',
      )

  def test_manual_stacking_not_same_optimizer(self):
    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.5),
        combiner='sum',
        name='A',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='B',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    feature_specs = [
        embedding_spec.FeatureSpec(
            table_spec=table_spec_a,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_a.embedding_dim,
            ),
            name='feature_a',
        ),
        embedding_spec.FeatureSpec(
            table_spec=table_spec_b,
            input_shape=(16, 1),
            output_shape=(
                16,
                table_spec_b.embedding_dim,
            ),
            name='feature_b',
        ),
    ]

    with self.assertRaisesRegex(ValueError, '.*different optimizers.*'):
      table_stacking.stack_tables(
          feature_specs,
          ('A', 'B'),
          global_device_count=jax.device_count(),
          num_sc_per_device=utils.num_sparsecores_per_device(jax.devices()[0]),
      )

  def test_manual_stacking_overlapping_stacks(self):
    table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=64,
        embedding_dim=12,
        initializer=lambda: jnp.zeros((64, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='A',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=10,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='B',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    table_spec_c = embedding_spec.TableSpec(
        vocabulary_size=120,
        embedding_dim=16,
        initializer=lambda: jnp.zeros((128, 16), dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='C',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    # Use new feature specs for each of the following tests.
    def get_feature_specs() -> list[embedding_spec.FeatureSpec]:
      return [
          embedding_spec.FeatureSpec(
              table_spec=table_spec_a,
              input_shape=(16, 1),
              output_shape=(
                  16,
                  table_spec_a.embedding_dim,
              ),
              name='feature_a',
          ),
          embedding_spec.FeatureSpec(
              table_spec=table_spec_b,
              input_shape=(16, 1),
              output_shape=(
                  16,
                  table_spec_b.embedding_dim,
              ),
              name='feature_b',
          ),
          embedding_spec.FeatureSpec(
              table_spec=table_spec_c,
              input_shape=(16, 1),
              output_shape=(
                  16,
                  table_spec_c.embedding_dim,
              ),
              name='feature_c',
          ),
      ]

    with self.assertRaisesRegex(ValueError, 'Table A is repeated in group'):
      table_stacking.stack_tables(
          get_feature_specs(),
          ('A', 'B', 'A'),
          global_device_count=jax.device_count(),
          num_sc_per_device=self.num_sc_per_device,
      )

    table_stacking.stack_tables(
        get_feature_specs(),
        ('A', 'B'),
        global_device_count=jax.device_count(),
        num_sc_per_device=self.num_sc_per_device,
    )

    with self.assertRaisesRegex(ValueError, 'Table B is already stacked.'):
      table_stacking.stack_tables(
          get_feature_specs(),
          ('C', 'B'),
          global_device_count=jax.device_count(),
          num_sc_per_device=self.num_sc_per_device,
      )

  @parameterized.product(
      donate=[True, False],
      device_count=[1, 2, 4, -1],
  )
  def test_unshard_and_unstack_stacked_table(
      self, donate: bool, device_count: int
  ):
    if device_count < 0:
      device_count = jax.device_count()

    vocab_size_a = 32
    vocab_size_b = 128
    embedding_dim = 14
    batch_size = 16

    vocab_size_c = 256
    embedding_dim_c = 32
    devices = jax.devices()[:device_count]

    table_a_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_a,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(0.0),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_a',
    )

    table_b_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_b,
        embedding_dim=embedding_dim,
        initializer=jax.nn.initializers.constant(1.0),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_b',
    )

    table_c_spec = embedding_spec.TableSpec(
        vocabulary_size=vocab_size_c,
        embedding_dim=embedding_dim_c,
        initializer=jax.nn.initializers.constant(1.0),
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name='table_c',
    )

    feature_a_spec = embedding_spec.FeatureSpec(
        table_spec=table_a_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name='feature_a',
    )

    feature_b_spec = embedding_spec.FeatureSpec(
        table_spec=table_b_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name='feature_b',
    )

    feature_c_spec = embedding_spec.FeatureSpec(
        table_spec=table_c_spec,
        input_shape=[batch_size, 1],
        output_shape=[batch_size, embedding_dim],
        name='feature_c',
    )

    # Prepare feature specs with stacking
    feature_specs = [feature_a_spec, feature_b_spec]
    table_stacking.auto_stack_tables(
        feature_specs,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=device_count,
    )

    feature_specs_c = [feature_c_spec]
    table_stacking.auto_stack_tables(
        feature_specs_c,
        num_sc_per_device=self.num_sc_per_device,
        global_device_count=device_count,
    )

    updated_table_a = feature_specs[0].table_spec
    updated_table_b = feature_specs[1].table_spec
    updated_table_c = feature_specs_c[0].table_spec

    table_a = test_utils.row_id_initializer(
        (
            updated_table_a.setting_in_stack.padded_vocab_size,
            table_a_spec.setting_in_stack.padded_embedding_dim,
        ),
        offset=0,
    )
    table_b = test_utils.row_id_initializer(
        (
            updated_table_b.setting_in_stack.padded_vocab_size,
            table_b_spec.setting_in_stack.padded_embedding_dim,
        ),
        offset=1000,
    )
    table_c = test_utils.row_id_initializer(
        (
            updated_table_c.setting_in_stack.padded_vocab_size,
            table_c_spec.setting_in_stack.padded_embedding_dim,
        ),
        offset=0,
    )

    emb_tables = [table_a, table_b]
    embedding_var = test_utils.create_per_device_sharded_stacked_tables(
        emb_tables,
        num_devices=device_count,
        num_sparsecore_per_device=self.num_sc_per_device,
        rotation=self.num_sc_per_device,
    )
    embedding_var = embedding_var.reshape(
        -1, feature_specs[0].table_spec.setting_in_stack.padded_embedding_dim
    )
    logging.vlog(1, 'embedding_var: %s', embedding_var)

    emb_tables_c = [table_c]
    embedding_var_c = test_utils.create_per_device_sharded_stacked_tables(
        emb_tables_c,
        num_devices=device_count,
        num_sparsecore_per_device=self.num_sc_per_device,
        rotation=self.num_sc_per_device,
    )
    embedding_var_c = embedding_var_c.reshape(
        -1, feature_specs_c[0].table_spec.setting_in_stack.padded_embedding_dim
    )
    logging.vlog(1, 'embedding_var_c: %s', embedding_var_c)

    # distribute to all devices
    mesh = jax.sharding.Mesh(devices, 'data')
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('data')
    )
    embedding_var = jax.device_put(embedding_var, sharding, donate=True)
    embedding_var_c = jax.device_put(embedding_var_c, sharding, donate=True)

    table_spec_proto = embedding.create_proto_from_feature_specs(
        feature_specs,
        global_device_count=device_count,
        num_sparsecore_per_device=self.num_sc_per_device,
    )

    # prepare array in the expected unstacked forms
    table_a_expected = jax.device_put(table_a, sharding)[
        :vocab_size_a, :embedding_dim
    ]
    table_b_expected = jax.device_put(table_b, sharding)[
        :vocab_size_b, :embedding_dim
    ]
    table_c_expected = jax.device_put(table_c, sharding)[
        :vocab_size_c, :embedding_dim_c
    ]

    with self.subTest('test_single_stack_table'):
      stacked_table_spec = table_spec_proto.stacked_table_specs[0]
      ret = table_stacking._unstack_and_unshard_stacked_table(
          embedding_var, stacked_table_spec, donate=donate
      )

      logging.vlog(1, "ret['table_a']: %s", ret['table_a'])
      self.assertTrue(
          jnp.array_equal(
              ret['table_a'],
              table_a_expected,
          )
      )

      logging.vlog(1, "ret['table_b']: %s", ret['table_b'])
      self.assertTrue(
          jnp.array_equal(
              ret['table_b'],
              table_b_expected,
          )
      )

    with self.subTest('test_multiple_stack_tables'):
      table_spec_proto = embedding.create_proto_from_feature_specs(
          feature_specs + feature_specs_c,
          global_device_count=device_count,
          num_sparsecore_per_device=self.num_sc_per_device,
      )
      ret = table_stacking.unstack_and_unshard_stacked_tables(
          {
              table_spec_proto.stacked_table_specs[0].stack_name: embedding_var,
              table_spec_proto.stacked_table_specs[
                  1
              ].stack_name: embedding_var_c,
          },
          table_spec_proto,
          donate=donate,
      )

      logging.vlog(1, "ret['table_a']: %s", ret['table_a'])
      self.assertTrue(
          jnp.array_equal(
              ret['table_a'],
              table_a_expected,
          )
      )

      logging.vlog(1, "ret['table_b']: %s", ret['table_b'])
      self.assertTrue(
          jnp.array_equal(
              ret['table_b'],
              table_b_expected,
          )
      )

      logging.vlog(1, "ret['table_c']: %s", ret['table_c'])
      self.assertTrue(
          jnp.array_equal(
              ret['table_c'],
              table_c_expected,
          )
      )


if __name__ == '__main__':
  absltest.main()
