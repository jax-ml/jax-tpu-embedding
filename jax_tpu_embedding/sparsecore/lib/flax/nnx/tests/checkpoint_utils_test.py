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
"""Tests for SparseCore NNX checkpoint_utils."""

import os
from absl import flags
from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax.nnx import checkpoint_utils
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn import table_stacking
from jax_tpu_embedding.sparsecore.lib.proto import embedding_spec_pb2
import orbax.checkpoint as ocp

FLAGS = flags.FLAGS


class CheckpointUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def _create_feature_specs(
      self, batch_size: int = 2048, dim: int = 640
  ) -> dict[str, embedding_spec.FeatureSpec]:
    opt = embedding_spec.SGDOptimizerSpec(learning_rate=0.01)
    specs = {}
    for name in ['table_a', 'table_b', 'table_c', 'table_d']:
      tspec = embedding_spec.TableSpec(
          vocabulary_size=2048,
          embedding_dim=dim,
          initializer=jax.nn.initializers.ones,
          optimizer=opt,
          combiner='sum',
          name=name,
      )
      fspec = embedding_spec.FeatureSpec(
          table_spec=tspec,
          input_shape=(batch_size, 1),
          output_shape=(batch_size, dim),
          name=f'feature_{name}',
      )
      specs[f'feature_{name}'] = fspec
    return specs

  def test_cross_topology_conversion_automatic_regrouping(self):
    src_devices = 8
    num_sc = 2
    src_specs = self._create_feature_specs(batch_size=2048, dim=640)
    table_stacking.auto_stack_tables(
        src_specs, global_device_count=src_devices, num_sc_per_device=num_sc
    )

    src_stacks = {
        v.table_spec.stacked_table_spec.stack_name for v in src_specs.values()
    }
    self.assertLen(src_stacks, 1)

    src_dir = os.path.join(self.temp_dir, 'source_cp')
    src_proto = embedding.create_proto_from_feature_specs(
        src_specs, src_devices, num_sc
    )

    table_specs = checkpoint_utils._get_table_specs(src_specs)
    num_shards = src_devices * num_sc
    dummy_tables = {
        name: jnp.ones((spec.vocabulary_size, spec.embedding_dim))
        for name, spec in table_specs.items()
    }
    stacked_tables = table_stacking.stack_and_shard_tables(
        table_specs, dummy_tables, num_shards=num_shards
    )
    mock_model_state_table = {}
    for s_name, arr in stacked_tables.items():
      assert isinstance(arr, jax.Array)
      mock_model_state_table[s_name] = {
          'table': arr.reshape((-1, arr.shape[-1])),
          'slot': (arr.reshape((-1, arr.shape[-1])),),
      }
    mock_model_state = {
        'embedding': {'embedding_table': mock_model_state_table}
    }
    mock_opt_state = {'step': 1}

    cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=src_dir,
        cp_options=ocp.CheckpointManagerOptions(create=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    cp_mgr.save(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(mock_model_state),
            optimizer=ocp.args.StandardSave(mock_opt_state),
            embedding_spec=ocp.args.ProtoSave(src_proto),
        ),
    )
    cp_mgr.wait_until_finished()

    target_devices = 4
    target_specs = self._create_feature_specs(batch_size=2048, dim=640)
    table_stacking.auto_stack_tables(
        target_specs,
        global_device_count=target_devices,
        num_sc_per_device=num_sc,
    )
    target_stacks = {
        v.table_spec.stacked_table_spec.stack_name
        for v in target_specs.values()
    }
    self.assertLen(target_stacks, 2)

    # Automatic conversion WITHOUT target_feature_specs
    out_dir = os.path.join(self.temp_dir, 'converted_cp_auto')
    checkpoint_utils.convert_cross_topology_checkpoint(
        input_checkpoint_path=src_dir,
        output_checkpoint_path=out_dir,
        num_global_devices=target_devices,
        num_sc_per_device=num_sc,
    )

    out_cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=out_dir,
        cp_options=ocp.CheckpointManagerOptions(read_only=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    restored = out_cp_mgr.restore(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(),
            optimizer=ocp.args.StandardRestore(),
            embedding_spec=ocp.args.ProtoRestore(
                embedding_spec_pb2.EmbeddingSpecProto
            ),
        ),
    )
    converted_proto = restored['embedding_spec']
    converted_stack_names = {
        s.stack_name for s in converted_proto.stacked_table_specs
    }
    self.assertEqual(converted_stack_names, target_stacks)

  def test_cross_topology_conversion_automatic_regrouping_with_target_batch_size(
      self,
  ):
    src_devices = 8
    num_sc = 2
    src_specs = self._create_feature_specs(batch_size=2048, dim=640)
    table_stacking.auto_stack_tables(
        src_specs, global_device_count=src_devices, num_sc_per_device=num_sc
    )

    src_dir = os.path.join(self.temp_dir, 'source_cp_bs')
    src_proto = embedding.create_proto_from_feature_specs(
        src_specs, src_devices, num_sc
    )

    table_specs = checkpoint_utils._get_table_specs(src_specs)
    num_shards = src_devices * num_sc
    dummy_tables = {
        name: jnp.ones((spec.vocabulary_size, spec.embedding_dim))
        for name, spec in table_specs.items()
    }
    stacked_tables = table_stacking.stack_and_shard_tables(
        table_specs, dummy_tables, num_shards=num_shards
    )
    mock_model_state_table = {}
    for s_name, arr in stacked_tables.items():
      assert isinstance(arr, jax.Array)
      mock_model_state_table[s_name] = {
          'table': arr.reshape((-1, arr.shape[-1])),
          'slot': (arr.reshape((-1, arr.shape[-1])),),
      }
    mock_model_state = {
        'embedding': {'embedding_table': mock_model_state_table}
    }
    mock_opt_state = {'step': 1}

    cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=src_dir,
        cp_options=ocp.CheckpointManagerOptions(create=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    cp_mgr.save(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(mock_model_state),
            optimizer=ocp.args.StandardSave(mock_opt_state),
            embedding_spec=ocp.args.ProtoSave(src_proto),
        ),
    )
    cp_mgr.wait_until_finished()

    target_devices = 4
    # With target_batch_size=1024, per-SC activation memory =
    # 640 * (1024 // 8) * 4 = 320 KiB/table. 4 tables = 1.25 MiB < 2 MiB ->
    # 1 stack.
    out_dir_bs1024 = os.path.join(self.temp_dir, 'converted_cp_bs1024')
    checkpoint_utils.convert_cross_topology_checkpoint(
        input_checkpoint_path=src_dir,
        output_checkpoint_path=out_dir_bs1024,
        num_global_devices=target_devices,
        num_sc_per_device=num_sc,
        target_batch_size=1024,
    )

    out_cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=out_dir_bs1024,
        cp_options=ocp.CheckpointManagerOptions(read_only=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    restored = out_cp_mgr.restore(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(),
            optimizer=ocp.args.StandardRestore(),
            embedding_spec=ocp.args.ProtoRestore(
                embedding_spec_pb2.EmbeddingSpecProto
            ),
        ),
    )
    converted_proto = restored['embedding_spec']
    self.assertLen(converted_proto.stacked_table_specs, 1)

  def test_recompute_target_specs_upscale_legacy_proto(self):
    source_proto = embedding_spec_pb2.EmbeddingSpecProto()
    stacked_spec = source_proto.stacked_table_specs.add()
    stacked_spec.stack_name = 'stack_0'
    stacked_spec.stack_embedding_dim = 16
    stacked_spec.num_sparsecores = 2
    stacked_spec.total_sample_count = 100

    t1 = stacked_spec.table_specs.add()
    t1.table_name = 'table_a'
    t1.vocab_size = 384
    t1.embedding_dim = 16

    t2 = stacked_spec.table_specs.add()
    t2.table_name = 'table_b'
    t2.vocab_size = 390
    t2.embedding_dim = 16

    # 2 shards: alignment 16 -> table_a=384, table_b=400. stack_vocab_size = 784
    stacked_spec.stack_vocab_size = 784

    # Upscale to 4 shards: alignment 32 -> table_a=384, table_b=416.
    # stack_vocab_size = 800
    _, logical_table_specs = checkpoint_utils._recompute_target_specs(
        source_proto=source_proto,
        num_global_devices=4,
        num_sc_per_device=1,
    )

    for tspec in logical_table_specs.values():
      self.assertEqual(tspec.stacked_table_spec.stack_vocab_size, 800)

    dummy_tables = {
        'table_a': jnp.ones((384, 16)),
        'table_b': jnp.ones((390, 16)),
    }
    stacked = table_stacking.stack_and_shard_tables(
        logical_table_specs, dummy_tables, num_shards=4
    )
    self.assertIn('stack_0', stacked)
    stack_arr = stacked['stack_0']
    assert isinstance(stack_arr, jax.Array)
    self.assertEqual(stack_arr.shape, (4, 200, 16))

  def test_cross_topology_conversion_with_dict_slots(self):
    src_devices = 8
    num_sc = 2
    src_specs = self._create_feature_specs(batch_size=2048, dim=640)
    table_stacking.auto_stack_tables(
        src_specs, global_device_count=src_devices, num_sc_per_device=num_sc
    )

    src_dir = os.path.join(self.temp_dir, 'source_cp_dict_slots')
    src_proto = embedding.create_proto_from_feature_specs(
        src_specs, src_devices, num_sc
    )

    table_specs = checkpoint_utils._get_table_specs(src_specs)
    num_shards = src_devices * num_sc
    dummy_tables = {
        name: jnp.ones((spec.vocabulary_size, spec.embedding_dim))
        for name, spec in table_specs.items()
    }
    stacked_tables = table_stacking.stack_and_shard_tables(
        table_specs, dummy_tables, num_shards=num_shards
    )
    mock_model_state_table = {}
    for s_name, arr in stacked_tables.items():
      assert isinstance(arr, jax.Array)
      mock_model_state_table[s_name] = {
          'table': arr.reshape((-1, arr.shape[-1])),
          'slot': {
              'm': arr.reshape((-1, arr.shape[-1])),
              'v': arr.reshape((-1, arr.shape[-1])) * 2.0,
          },
      }
    mock_model_state = {
        'embedding': {'embedding_table': mock_model_state_table}
    }
    mock_opt_state = {'step': 1}

    cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=src_dir,
        cp_options=ocp.CheckpointManagerOptions(create=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    cp_mgr.save(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(mock_model_state),
            optimizer=ocp.args.StandardSave(mock_opt_state),
            embedding_spec=ocp.args.ProtoSave(src_proto),
        ),
    )
    cp_mgr.wait_until_finished()

    target_devices = 4
    out_dir = os.path.join(self.temp_dir, 'converted_cp_dict_slots')
    checkpoint_utils.convert_cross_topology_checkpoint(
        input_checkpoint_path=src_dir,
        output_checkpoint_path=out_dir,
        num_global_devices=target_devices,
        num_sc_per_device=num_sc,
    )

    out_cp_mgr = checkpoint_utils.create_checkpoint_manager(
        cp_path=out_dir,
        cp_options=ocp.CheckpointManagerOptions(read_only=True),
        model_key='model',
        optimizer_key='optimizer',
        embedding_spec_key='embedding_spec',
    )
    restored = out_cp_mgr.restore(
        1,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(),
            optimizer=ocp.args.StandardRestore(),
            embedding_spec=ocp.args.ProtoRestore(
                embedding_spec_pb2.EmbeddingSpecProto
            ),
        ),
    )
    restored_table_dict = restored['model']['embedding']['embedding_table']
    for s_name, ev_dict in restored_table_dict.items():
      self.assertIsInstance(ev_dict['slot'], dict)
      self.assertIn('m', ev_dict['slot'])
      self.assertIn('v', ev_dict['slot'])


if __name__ == '__main__':
  absltest.main()
