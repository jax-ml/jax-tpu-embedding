# Copyright 2024 JAX SC Authors.
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

import functools

from absl.testing import absltest
import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
import numpy as np
import tree


_VOC_A = 32
_VOC_B = 64
_VOC_C = 32
_DIM_A = 8
_DIM_B = 16
_DIM_C = 8
_BATCH_SIZE = 16

_EMBED_PARAM = embed.EMBEDDING_PARAM_NAME


def count_num(arr, num):
  """Helper method that counts the number of occurrences of `num` in `arr`."""
  count = 0
  for array in arr:
    for n in array:
      if n == num:
        count += 1
  return count


class EmbeddingLayerTest(absltest.TestCase):
  table_spec_a = embedding_spec.TableSpec(
      vocabulary_size=_VOC_A,
      embedding_dim=_DIM_A,
      initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
      combiner='sum',
      name='table_a',
      max_ids_per_partition=16,
      max_unique_ids_per_partition=16,
  )
  table_spec_b = embedding_spec.TableSpec(
      vocabulary_size=_VOC_B,
      embedding_dim=_DIM_B,
      initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
      combiner='sum',
      name='table_b',
      max_ids_per_partition=16,
      max_unique_ids_per_partition=16,
  )
  table_spec_c = embedding_spec.TableSpec(
      vocabulary_size=_VOC_C,
      embedding_dim=_DIM_C,
      initializer=lambda _, shape: jnp.zeros(shape, dtype=jnp.float32),
      optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.01),
      combiner='sum',
      name='table_c',
      max_ids_per_partition=16,
      max_unique_ids_per_partition=16,
  )
  feature_spec_a = embedding_spec.FeatureSpec(
      table_spec=table_spec_a,
      input_shape=(_BATCH_SIZE, 1),
      output_shape=(
          _BATCH_SIZE,
          table_spec_a.embedding_dim,
      ),
      name='feature_spec_a',
  )
  feature_spec_b = embedding_spec.FeatureSpec(
      table_spec=table_spec_b,
      input_shape=(_BATCH_SIZE, 1),
      output_shape=(
          _BATCH_SIZE,
          table_spec_b.embedding_dim,
      ),
      name='feature_spec_b',
  )
  feature_spec_c = embedding_spec.FeatureSpec(
      table_spec=table_spec_c,
      input_shape=(_BATCH_SIZE, 1),
      output_shape=(
          _BATCH_SIZE,
          table_spec_c.embedding_dim,
      ),
      name='feature_spec_c',
  )
  input_tensor = np.array(
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
  input_tensor_table_b = np.array(
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
  input_weights = np.array(
      [
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
      ],
      dtype=object,
  )
  input_weights_table_b = np.array(
      [
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
          np.array([1.0, 1.0], dtype=np.float32),
      ],
      dtype=object,
  )

  def test_forward_and_backward_with_one_table(self):
    devices = jax.devices()
    feature_specs = (self.feature_spec_a, self.feature_spec_b)
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
    )
    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
    )

    embedding_lookups = sc_module.preprocess_inputs(
        (self.input_tensor, self.input_tensor_table_b),
        (self.input_weights, self.input_weights_table_b),
    )

    padded_vocab_a = 128
    padded_vocab_b = 128
    emb_table_a = (
        np.array([[i for _ in range(_DIM_A)] for i in range(padded_vocab_a)])
        .reshape(padded_vocab_a, _DIM_A)
        .astype(np.float32)
    )
    emb_table_a_sharded = einops.rearrange(
        emb_table_a,
        '(v c s) f -> c (s v) f',
        c=4,
        s=4,
    )
    emb_table_b = (
        np.array([[i for _ in range(_DIM_B)] for i in range(padded_vocab_b)])
        .reshape(padded_vocab_b, _DIM_B)
        .astype(np.float32)
    )
    emb_table_b_sharded = einops.rearrange(
        emb_table_b,
        '(v c s) f -> c (s v) f',
        c=4,
        s=4,
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
    sharding = NamedSharding(sc_module.mesh, P(sc_module.sharding_axis, None))
    embedding_variables['table_a'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_a, _DIM_A),
            sharding=sharding,
            arrays=embedding_variables['table_a'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )
    embedding_variables['table_b'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(padded_vocab_b, _DIM_B),
            sharding=sharding,
            arrays=embedding_variables['table_b'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )

    var_spec = jax.eval_shape(
        sc_module.init,
        jax.random.PRNGKey(0),
        embedding_lookups,
    )

    out_sharding = nn.get_sharding(var_spec, sc_module.mesh)

    params = jax.jit(
        sc_module.init,
        in_shardings=(
            NamedSharding(sc_module.mesh, P()),
            embed.EmbeddingLookups(
                NamedSharding(sc_module.mesh, P(sc_module.sharding_axis)),
                NamedSharding(sc_module.mesh, P(sc_module.sharding_axis)),
                NamedSharding(sc_module.mesh, P(sc_module.sharding_axis)),
                NamedSharding(sc_module.mesh, P(sc_module.sharding_axis)),
            ),
        ),
        out_shardings=out_sharding,
    )(
        jax.random.PRNGKey(0),
        embedding_lookups,
    )

    # Replace the embedding variables in params with the ones we created.
    def check_shape(a, b):
      assert a.shape == b.shape

    jax.tree.map(
        check_shape,
        params['params'][_EMBED_PARAM].value,
        embedding_variables,
    )
    params['params'][_EMBED_PARAM] = params['params'][
        _EMBED_PARAM
    ].replace_boxed(embedding_variables)

    activations = jax.jit(sc_module.apply)(
        params,
        embedding_lookups,
    )

    # Check the activation correctness.
    expected_emb_activations = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_equal(activations[0], expected_emb_activations)

    expected_emb_activations_table_b = np.array(
        [
            [110.0] * 16,
            [33.0] * 16,
            [59.0] * 16,
            [26.0] * 16,
            [29.0] * 16,
            [31.0] * 16,
            [67.0] * 16,
            [57.0] * 16,
            [101.0] * 16,
            [120.0] * 16,
            [36.0] * 16,
            [121.0] * 16,
            [60.0] * 16,
            [132.0] * 16,
            [60.0] * 16,
            [26.0] * 16,
        ],
        dtype=np.float32,
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

    params_updates = jax.jit(
        functools.partial(sc_module.apply, method=sc_module.apply_gradient),
    )(
        params,
        activations_grad,
        embedding_lookups,
    )

    # Updates params with the new embedding variables.
    assert len(params_updates) == 1
    tree.assert_same_structure(
        params_updates[_EMBED_PARAM], params['params'][_EMBED_PARAM].value
    )
    params['params'] = params['params'] | params_updates

    expected_grad_table_a = np.zeros((padded_vocab_a, _DIM_A), dtype=np.float32)
    expected_grad_table_b = np.zeros((padded_vocab_b, _DIM_B), dtype=np.float32)

    for i, array in enumerate(embedding_variables['table_a'][0]):
      col_id = array[0]
      new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      expected_grad_table_a[i] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    for i, array in enumerate(embedding_variables['table_b'][0]):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_b, col_id) * 0.01
      )
      expected_grad_table_b[i] = np.full(
          (1, _DIM_B), new_col_id, dtype=np.float32
      )
    np.testing.assert_equal(
        expected_grad_table_a, params['params'][_EMBED_PARAM]['table_a'][0]
    )
    np.testing.assert_equal(
        expected_grad_table_b, params['params'][_EMBED_PARAM]['table_b'][0]
    )

  def test_forward_and_backward_with_table_stacking(self):
    devices = jax.devices()
    sharding_axis = 'x'
    mesh = jax.sharding.Mesh(devices, sharding_axis)
    feature_specs = (self.feature_spec_a, self.feature_spec_c)
    embedding.auto_stack_tables(
        feature_specs,
        global_device_count=jax.device_count(),
    )
    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
        mesh=mesh,
        sharding_axis=sharding_axis,
    )

    embedding_lookups = sc_module.preprocess_inputs(
        (self.input_tensor, self.input_tensor),
        (self.input_weights, self.input_weights),
    )
    padded_vocab_a = 128
    padded_vocab_c = 128
    stacked_vocab_size = padded_vocab_a + padded_vocab_c
    emb_table_a = test_utils.row_id_initializer(shape=(padded_vocab_a, _DIM_A))
    emb_table_c = test_utils.row_id_initializer(
        shape=(padded_vocab_c, _DIM_C), offset=200
    )
    embedding_variables = {}
    sharded_stacked_tables = (
        test_utils.create_per_device_sharded_stacked_tables(
            [emb_table_a, emb_table_c],
            num_devices=mesh.size,
            num_sparsecore_per_device=4,
            rotation=4,
        )
    )
    embedding_variables['table_a_table_c'] = [
        jax.device_put(
            sharded_stacked_tables[i],
            device=local_device,
        )
        for i, local_device in enumerate(devices)
    ]
    sharding = NamedSharding(mesh, P('x', None))
    embedding_variables['table_a_table_c'] = embedding.EmbeddingVariables(
        table=jax.make_array_from_single_device_arrays(
            shape=(stacked_vocab_size, _DIM_A),
            sharding=sharding,
            arrays=embedding_variables['table_a_table_c'],
        ),
        slot=embedding_spec.SGDSlotVariables(),
    )

    var_spec = jax.eval_shape(
        sc_module.init,
        jax.random.PRNGKey(0),
        embedding_lookups,
    )

    out_sharding = nn.get_sharding(var_spec, mesh)

    params = jax.jit(
        sc_module.init,
        in_shardings=(
            NamedSharding(mesh, P()),
            embed.EmbeddingLookups(
                NamedSharding(mesh, P(sharding_axis)),
                NamedSharding(mesh, P(sharding_axis)),
                NamedSharding(mesh, P(sharding_axis)),
                NamedSharding(mesh, P(sharding_axis)),
            ),
        ),
        out_shardings=out_sharding,
    )(
        jax.random.PRNGKey(0),
        embedding_lookups,
    )

    # Replace the embedding variables in params with the ones we created.
    def check_shape(a, b):
      assert a.shape == b.shape

    jax.tree.map(
        check_shape,
        params['params'][_EMBED_PARAM].value,
        embedding_variables,
    )
    params['params'][_EMBED_PARAM] = params['params'][
        _EMBED_PARAM
    ].replace_boxed(embedding_variables)

    activations = jax.jit(sc_module.apply)(
        params,
        embedding_lookups,
    )

    # Check the activation correctness.
    expected_emb_activations = np.array(
        [
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
            [29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0],
            [31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0, 31.0],
            [67.0, 67.0, 67.0, 67.0, 67.0, 67.0, 67.0, 67.0],
            [57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0, 57.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            [13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0],
            [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            [17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
            [42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0, 42.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0],
        ],
        dtype=np.float32,
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

    params_updates = jax.jit(
        functools.partial(sc_module.apply, method=sc_module.apply_gradient),
    )(
        params,
        activations_grad,
        embedding_lookups,
    )

    # Updates params with the new embedding variables.
    assert len(params_updates) == 1
    tree.assert_same_structure(
        params_updates[_EMBED_PARAM], params['params'][_EMBED_PARAM].value
    )
    params['params'] = params['params'] | params_updates

    expected_grad_table_ac = np.zeros(
        (stacked_vocab_size, _DIM_A), dtype=np.float32
    )

    for i, array in enumerate(embedding_variables['table_a_table_c'][0]):
      col_id = array[0]
      if col_id < 200:
        new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      else:
        new_col_id = col_id - (
            count_num(self.input_tensor, col_id - 200) * 0.01
        )
      expected_grad_table_ac[i] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    np.testing.assert_equal(
        expected_grad_table_ac,
        params['params'][_EMBED_PARAM]['table_a_table_c'][0],
    )


if __name__ == '__main__':
  absltest.main()
