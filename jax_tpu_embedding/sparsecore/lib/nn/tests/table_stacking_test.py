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

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking


class TableStackingTest(absltest.TestCase):

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

  def test_auto_stack_two_tables(self):
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
        global_device_count=jax.device_count(),
    )
    self.assertLen(jax.devices(), 4)
    self.assertLen(feature_specs, 3)
    feature_a, feature_b, feature_c = feature_specs
    expected_padded_vocab = 128  # Both tables round up to 128
    total_sc_in_test = jax.device_count() * 4  # num_sc_per_device
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.stack_name, 'table_a_table_b'
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab,
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
        feature_b.table_spec.setting_in_stack.row_offset_in_shard,
        expected_padded_vocab // total_sc_in_test,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_embedding_dim, 16
    )
    self.assertEqual(feature_b.id_transformation.row_offset, 16)
    self.assertEqual(
        feature_b.id_transformation.col_offset, expected_padded_vocab
    )
    self.assertEqual(feature_c.id_transformation.row_offset, 16 + 16)
    self.assertEqual(
        feature_c.id_transformation.col_offset, expected_padded_vocab
    )

  def test_auto_stack_two_tables_adagrad(self):
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
        global_device_count=jax.device_count(),
    )
    self.assertLen(jax.devices(), 4)
    self.assertLen(feature_specs, 3)
    feature_a, feature_b, feature_c = feature_specs
    expected_padded_vocab = 128  # Both tables round up to 128
    total_sc_in_test = jax.device_count() * 4  # num_sc_per_device
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.stack_name,
        'table_a_table_b',
    )
    self.assertEqual(
        feature_a.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab,
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
        expected_padded_vocab // total_sc_in_test,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_vocab_size,
        expected_padded_vocab,
    )
    self.assertEqual(
        feature_b.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(feature_b.id_transformation.row_offset, 16)
    self.assertEqual(
        feature_b.id_transformation.col_offset, expected_padded_vocab
    )
    self.assertEqual(feature_c.id_transformation.row_offset, 16 + 16)
    self.assertEqual(
        feature_c.id_transformation.col_offset, expected_padded_vocab
    )

  def test_manual_stacking(self):
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
        global_device_count=jax.device_count(),
    )

    def limits(name: str, batch_size: int) -> int:
      del name
      return int(batch_size * 0.8)

    table_stacking.stack_tables(
        feature_specs,
        ['table_c', 'table_b'],
        global_device_count=jax.device_count(),
        stack_to_max_ids_per_partition=limits,
        stack_to_max_unique_ids_per_partition=limits,
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
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.padded_vocab_size,
        128,
    )
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.padded_embedding_dim,
        16,
    )
    self.assertEqual(
        updated_feature_b.table_spec.setting_in_stack.shard_rotation,
        4,
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
      )

    table_stacking.stack_tables(
        get_feature_specs(),
        ('A', 'B'),
        global_device_count=jax.device_count(),
    )

    with self.assertRaisesRegex(ValueError, 'Table B is already stacked.'):
      table_stacking.stack_tables(
          get_feature_specs(),
          ('C', 'B'),
          global_device_count=jax.device_count(),
      )


if __name__ == '__main__':
  absltest.main()
