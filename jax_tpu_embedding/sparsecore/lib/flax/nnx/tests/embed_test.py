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
import einops
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.lib.flax.nnx import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import portpicker


_VOC_A = 31
_VOC_B = 75
_VOC_C = 33
_DIM_A = 7
_DIM_B = 15
_DIM_C = 6
_BATCH_SIZE = 16
_PAD_VALUE = -1


def count_num(arr, num):
  """Helper method that counts the number of occurrences of `num` in `arr`."""
  count = 0
  for array in arr:
    for n in array:
      if n == num:
        count += 1
  return count


class EmbeddingLayerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.table_spec_a = embedding_spec.TableSpec(
        vocabulary_size=_VOC_A,
        embedding_dim=_DIM_A,
        initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
        combiner='sum',
        name='table_a',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.table_spec_b = embedding_spec.TableSpec(
        vocabulary_size=_VOC_B,
        embedding_dim=_DIM_B,
        initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
        combiner='sum',
        name='table_b',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.table_spec_c = embedding_spec.TableSpec(
        vocabulary_size=_VOC_C,
        embedding_dim=_DIM_C,
        initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
        optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
        combiner='sum',
        name='table_c',
        max_ids_per_partition=16,
        max_unique_ids_per_partition=16,
    )
    self.feature_spec_a = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_a,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_a.embedding_dim,
        ),
        name='feature_spec_a',
    )
    self.feature_spec_b = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_b,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_b.embedding_dim,
        ),
        name='feature_spec_b',
    )
    self.feature_spec_c = embedding_spec.FeatureSpec(
        table_spec=self.table_spec_c,
        input_shape=(_BATCH_SIZE, 1),
        output_shape=(
            _BATCH_SIZE,
            self.table_spec_c.embedding_dim,
        ),
        name='feature_spec_c',
    )
    self.input_tensor = np.array(
        [
            np.array([5, 4, 2], dtype=np.int32),
            np.array([3], dtype=np.int32),
            np.array([9], dtype=np.int32),
            np.array([1, 9, 16], dtype=np.int32),
            np.array([6, 12, 1, 10], dtype=np.int32),
            np.array([12, 19], dtype=np.int32),
            np.array([0, 15, 2, 25, 25], dtype=np.int32),
            np.array([4, 28, 25], dtype=np.int32),
            np.array([15, 0], dtype=np.int32),
            np.array([13], dtype=np.int32),
            np.array([11], dtype=np.int32),
            np.array([7, 1], dtype=np.int32),
            np.array([8, 9], dtype=np.int32),
            np.array([14, 14, 14], dtype=np.int32),
            np.array([2, 28], dtype=np.int32),
            np.array([10, 16], dtype=np.int32),
        ],
        dtype=object,
    )
    self.input_tensor_table_b = np.array(
        [
            np.array([50, 40, 20], dtype=np.int32),
            np.array([33], dtype=np.int32),
            np.array([59], dtype=np.int32),
            np.array([1, 9, 16], dtype=np.int32),
            np.array([6, 12, 1, 10], dtype=np.int32),
            np.array([12, 19], dtype=np.int32),
            np.array([0, 15, 2, 25, 25], dtype=np.int32),
            np.array([4, 28, 25], dtype=np.int32),
            np.array([50, 51, 0], dtype=np.int32),
            np.array([10, 50, 60], dtype=np.int32),
            np.array([11, 12, 13], dtype=np.int32),
            np.array([61, 60], dtype=np.int32),
            np.array([58, 2], dtype=np.int32),
            np.array([41, 41, 50], dtype=np.int32),
            np.array([2, 58], dtype=np.int32),
            np.array([10, 16], dtype=np.int32),
        ],
        dtype=object,
    )
    self.port = portpicker.pick_unused_port()

  @parameterized.named_parameters(
      dict(testcase_name='_with_minibatching', enable_minibatching=True),
      dict(testcase_name='_without_minibatching', enable_minibatching=False),
  )
  def test_forward_and_backward_with_one_table(self, enable_minibatching: bool):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
    sharding_axis = 'x'
    mesh = Mesh(devices, sharding_axis)
    data_sharding = NamedSharding(mesh, PartitionSpec(sharding_axis))

    feature_specs = (self.feature_spec_a, self.feature_spec_b)
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
    )

    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
        sharding_axis=sharding_axis,
        mesh=mesh,
        rngs=nnx.Rngs(params=1337),
        enable_minibatching=enable_minibatching,
    )
    step = 42
    all_reduce_interface = None
    if enable_minibatching:
      all_reduce_interface = embedding.get_all_reduce_interface(
          peer_addresses=[], minibatching_port=self.port
      )
    embedding_lookup_input = embedding.preprocess_sparse_dense_matmul_input(
        (self.input_tensor, self.input_tensor_table_b),
        None,  # uniform weights
        feature_specs,
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
        batch_number=step,
        enable_minibatching=enable_minibatching,
        all_reduce_interface=all_reduce_interface,
    )[0]
    embedding_lookup_input = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(data_sharding, x),
        embedding_lookup_input,
    )

    padded_vocab_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_vocab_size
    )
    padded_vocab_b = (
        self.feature_spec_b.table_spec.setting_in_stack.padded_vocab_size
    )
    padded_dim_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_embedding_dim
    )
    padded_dim_b = (
        self.feature_spec_b.table_spec.setting_in_stack.padded_embedding_dim
    )

    # Here, we construct the embedding tables manually to control the row values
    # so we can verify the activations.
    device_count = len(devices)
    emb_table_a = test_utils.row_initialize_with_padding(
        self.feature_spec_a.table_spec, pad_value=_PAD_VALUE
    )
    emb_table_a_sharded = einops.rearrange(
        emb_table_a,
        '(v c s) f -> c (s v) f',
        c=device_count,
        s=num_sc_per_device,
    )
    emb_table_b = test_utils.row_initialize_with_padding(
        self.feature_spec_b.table_spec, pad_value=_PAD_VALUE
    )
    emb_table_b_sharded = einops.rearrange(
        emb_table_b,
        '(v c s) f -> c (s v) f',
        c=device_count,
        s=num_sc_per_device,
    )

    embedding_variables = {}

    embedding_variables['table_a'] = [
        jax.device_put(
            emb_table_a_sharded[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    embedding_variables['table_b'] = [
        jax.device_put(
            emb_table_b_sharded[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, PartitionSpec(sharding_axis, None))
    embedding_variables['table_a'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_a, padded_dim_a),
            sharding=sharding,
            arrays=embedding_variables['table_a'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )
    embedding_variables['table_b'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_b, padded_dim_b),
            sharding=sharding,
            arrays=embedding_variables['table_b'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )
    sc_module.embedding_table.value = embedding_variables

    def _emb_lookup(*args, **kwargs):
      return embed.embedding_lookup(*args, **kwargs)

    activations = nnx.jit(_emb_lookup)(sc_module, embedding_lookup_input)

    # Check the activation correctness.
    expected_emb_activations = np.broadcast_to(
        np.array(
            [
                [11.0],
                [3.0],
                [9.0],
                [26.0],
                [29.0],
                [31.0],
                [67.0],
                [57.0],
                [15.0],
                [13.0],
                [11.0],
                [8.0],
                [17.0],
                [42.0],
                [30.0],
                [26.0],
            ],
            dtype=np.float32,
        ),
        (_BATCH_SIZE, _DIM_A),
    )
    np.testing.assert_equal(activations[0], expected_emb_activations)

    expected_emb_activations_table_b = np.broadcast_to(
        np.array(
            [
                [110.0],
                [33.0],
                [59.0],
                [26.0],
                [29.0],
                [31.0],
                [67.0],
                [57.0],
                [101.0],
                [120.0],
                [36.0],
                [121.0],
                [60.0],
                [132.0],
                [60.0],
                [26.0],
            ],
            dtype=np.float32,
        ),
        (_BATCH_SIZE, _DIM_B),
    )

    np.testing.assert_equal(activations[1], expected_emb_activations_table_b)

    activations_grad = (
        jnp.ones(  # feature_spec_a
            (_BATCH_SIZE, _DIM_A),
            dtype=jnp.float32,
        ),
        jnp.ones(  # feature_spec_b
            (_BATCH_SIZE, _DIM_B),
            dtype=jnp.float32,
        ),
    )

    m_update_g = {'embedding_table': sc_module.embedding_table.copy()}
    unused_ = embedding_lookup_input
    g = ((m_update_g, unused_), activations_grad)
    res = (sc_module, embedding_lookup_input)

    (grads, _) = nnx.jit(embed.embedding_lookup_bwd)(res, g)
    nnx.update(sc_module, grads)

    expected_grad_table_a = np.full(
        (padded_vocab_a, padded_dim_a), _PAD_VALUE, dtype=np.float32
    )
    expected_grad_table_b = np.full(
        (padded_vocab_b, padded_dim_b), _PAD_VALUE, dtype=np.float32
    )

    for i, array in enumerate(embedding_variables['table_a'][0]):
      col_id = array[0]
      new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      expected_grad_table_a[i, :_DIM_A] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    for i, array in enumerate(embedding_variables['table_b'][0]):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_b, col_id) * 0.01
      )
      expected_grad_table_b[i, :_DIM_B] = np.full(
          (1, _DIM_B), new_col_id, dtype=np.float32
      )
    np.testing.assert_equal(
        sc_module.embedding_table.value['table_a'][0], expected_grad_table_a
    )
    np.testing.assert_equal(
        sc_module.embedding_table.value['table_b'][0], expected_grad_table_b
    )

  @parameterized.named_parameters(
      dict(testcase_name='_with_minibatching', enable_minibatching=True),
      dict(testcase_name='_without_minibatching', enable_minibatching=False),
  )
  def test_forward_and_backward_with_table_stacking(
      self, enable_minibatching: bool
  ):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device()
    sharding_axis = 'x'
    mesh = Mesh(devices, sharding_axis)
    data_sharding = NamedSharding(mesh, PartitionSpec(sharding_axis))

    feature_specs = (self.feature_spec_a, self.feature_spec_c)
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
    )
    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
        mesh=mesh,
        sharding_axis=sharding_axis,
        rngs=nnx.Rngs(params=1337),
        enable_minibatching=enable_minibatching,
    )
    step = 42
    all_reduce_interface = None
    if enable_minibatching:
      all_reduce_interface = embedding.get_all_reduce_interface(
          peer_addresses=[], minibatching_port=self.port
      )
    embedding_lookup_input = embedding.preprocess_sparse_dense_matmul_input(
        (self.input_tensor, self.input_tensor),
        None,  # uniform weights
        feature_specs,
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
        batch_number=step,
        enable_minibatching=enable_minibatching,
        all_reduce_interface=all_reduce_interface,
    )[0]
    embedding_lookup_input = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(data_sharding, x),
        embedding_lookup_input,
    )
    padded_vocab_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_vocab_size
    )
    padded_vocab_c = (
        self.feature_spec_c.table_spec.setting_in_stack.padded_vocab_size
    )
    padded_dim_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_embedding_dim
    )
    stacked_vocab_size = padded_vocab_a + padded_vocab_c
    stacked_embedding_dim = padded_dim_a

    emb_table_a = test_utils.row_initialize_with_padding(
        self.table_spec_a, pad_value=_PAD_VALUE
    )
    emb_table_c = test_utils.row_initialize_with_padding(
        self.table_spec_c, offset=200, pad_value=_PAD_VALUE
    )
    embedding_variables = {}
    sharded_stacked_tables = (
        test_utils.create_per_device_sharded_stacked_tables(
            [emb_table_a, emb_table_c],
            num_devices=mesh.size,
            num_sparsecore_per_device=num_sc_per_device,
            rotation=num_sc_per_device,
        )
    )
    embedding_variables['table_a_table_c'] = [
        jax.device_put(
            sharded_stacked_tables[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, PartitionSpec(sharding_axis, None))
    embedding_variables['table_a_table_c'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(stacked_vocab_size, padded_dim_a),
            sharding=sharding,
            arrays=embedding_variables['table_a_table_c'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )

    # Replace the embedding variables in params with the ones we created.
    sc_module.embedding_table.value = embedding_variables

    def _emb_lookup(*args, **kwargs):
      return embed.embedding_lookup(*args, **kwargs)

    activations = nnx.jit(_emb_lookup)(sc_module, embedding_lookup_input)

    # Check the activation correctness.
    expected_emb_activations = np.broadcast_to(
        np.array(
            [
                [11.0],
                [3.0],
                [9.0],
                [26.0],
                [29.0],
                [31.0],
                [67.0],
                [57.0],
                [15.0],
                [13.0],
                [11.0],
                [8.0],
                [17.0],
                [42.0],
                [30.0],
                [26.0],
            ],
            dtype=np.float32,
        ),
        (_BATCH_SIZE, _DIM_A),
    )
    np.testing.assert_equal(activations[0], expected_emb_activations)

    activations_grad = (
        jnp.ones(  # feature_spec_a
            (_BATCH_SIZE, _DIM_A),
            dtype=jnp.float32,
        ),
        jnp.ones(  # feature_spec_c
            (_BATCH_SIZE, _DIM_C),
            dtype=jnp.float32,
        ),
    )

    m_update_g = {'embedding_table': sc_module.embedding_table.copy()}
    unused_ = embedding_lookup_input
    g = ((m_update_g, unused_), activations_grad)
    res = (sc_module, embedding_lookup_input)

    (grads, _) = nnx.jit(embed.embedding_lookup_bwd)(res, g)
    nnx.update(sc_module, grads)

    expected_grad_table_ac = np.full(
        (stacked_vocab_size, stacked_embedding_dim),
        _PAD_VALUE,
        dtype=np.float32,
    )

    for i, array in enumerate(embedding_variables['table_a_table_c'][0]):
      col_id = array[0]
      embedding_dim = _DIM_A
      if col_id < 200:
        new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      else:
        embedding_dim = _DIM_C
        new_col_id = col_id - (
            count_num(self.input_tensor, col_id - 200) * 0.01
        )
      expected_grad_table_ac[i, :embedding_dim] = np.full(
          (1, embedding_dim), new_col_id, dtype=np.float32
      )

    np.testing.assert_equal(
        sc_module.embedding_table.value['table_a_table_c'][0],
        expected_grad_table_ac,
    )


if __name__ == '__main__':
  absltest.main()
