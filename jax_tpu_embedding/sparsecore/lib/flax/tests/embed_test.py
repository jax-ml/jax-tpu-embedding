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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import einops
from flax import core
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.lib.nn.tests import test_utils
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import tree


_VOC_A = 31
_VOC_B = 75
_VOC_C = 33
_DIM_A = 7
_DIM_B = 15
_DIM_C = 6
_BATCH_SIZE = 16
_PAD_VALUE = -1

_EMBED_PARAM = embed.EMBEDDING_PARAM_NAME


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
    self.input_weights = np.array(
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
    self.input_weights_table_b = np.array(
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

  def _row_initialize_with_padding(
      self,
      shape: tuple[int, ...],
      padded_shape: tuple[int, ...],
      offset: int = 0,
      pad_value: float = _PAD_VALUE,
  ):
    array = test_utils.row_id_initializer(shape=shape, offset=offset)
    paddings = tuple((0, y - x) for x, y in zip(shape, padded_shape))
    return np.pad(array, paddings, mode='constant', constant_values=pad_value)

  def _create_embedding_variables(
      self,
      module: embed.SparseCoreEmbed,
      feature_specs: tuple[embedding_spec.FeatureSpec, ...],
      offsets: dict[str, int],
  ) -> core.FrozenDict[str, embedding.EmbeddingVariables]:
    """Creates sharded embedding variables for given feature specs."""
    embedding_variables = {}
    devices = module.mesh.devices.flatten()
    device_count = len(devices)
    num_sc_per_device = module.num_sc_per_device
    sharding = NamedSharding(module.mesh, P(module.sharding_axis, None))

    unique_tables = {}
    for f in feature_specs:
      if f.table_spec.name not in unique_tables:
        unique_tables[f.table_spec.name] = f.table_spec

    for table_name, table_spec in unique_tables.items():
      padded_vocab = table_spec.setting_in_stack.padded_vocab_size
      padded_dim = table_spec.setting_in_stack.padded_embedding_dim

      emb_table = self._row_initialize_with_padding(
          shape=(table_spec.vocabulary_size, table_spec.embedding_dim),
          padded_shape=(padded_vocab, padded_dim),
          offset=offsets.get(table_name, 0),
      )
      emb_table_sharded = einops.rearrange(
          emb_table,
          '(v c s) f -> c (s v) f',
          c=device_count,
          s=num_sc_per_device,
      )
      device_arrays = [
          jax.device_put(emb_table_sharded[i], device=d)
          for i, d in enumerate(devices)
      ]
      embedding_variables[table_name] = embedding.EmbeddingVariables(
          table=jax.make_array_from_single_device_arrays(
              shape=(padded_vocab, padded_dim),
              sharding=sharding,
              arrays=device_arrays,
          ),
          slot=embedding_spec.SGDSlotVariables(),
      )
    return core.freeze(embedding_variables)

  def _initialize_model_variables(
      self,
      module: embed.SparseCoreEmbed,
      embedding_lookup_input: embedding.PreprocessedInput,
      embedding_variables: core.FrozenDict[str, embedding.EmbeddingVariables],
  ):
    """Initializes model variables and replaces embedding table."""
    var_spec = jax.eval_shape(
        module.init,
        jax.random.PRNGKey(0),
        embedding_lookup_input,
    )
    out_sharding = nn.get_sharding(var_spec, module.mesh)
    variables = jax.jit(
        module.init,
        in_shardings=(
            NamedSharding(module.mesh, P()),
            NamedSharding(module.mesh, P(module.sharding_axis)),
        ),
        out_shardings=out_sharding,
    )(
        jax.random.PRNGKey(0),
        embedding_lookup_input,
    )

    # Replace the embedding variables in params with the ones we created.
    def check_shape(a, b):
      assert a.shape == b.shape

    jax.tree.map(
        check_shape,
        variables['params'][_EMBED_PARAM].value,
        embedding_variables,
    )
    variables['params'][_EMBED_PARAM] = variables['params'][
        _EMBED_PARAM
    ].replace_boxed(embedding_variables)
    return variables

  @parameterized.named_parameters(
      dict(testcase_name='_with_minibatching', enable_minibatching=True),
      dict(testcase_name='_without_minibatching', enable_minibatching=False),
  )
  def test_forward_and_backward_with_one_table(self, enable_minibatching: bool):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])

    feature_specs = (self.feature_spec_a, self.feature_spec_b)
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
    )

    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
        enable_minibatching=enable_minibatching,
    )
    step = 42
    embedding_lookup_input = sc_module.preprocess_inputs(
        step,
        (self.input_tensor, self.input_tensor_table_b),
        (self.input_weights, self.input_weights_table_b),
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

    embedding_variables = self._create_embedding_variables(
        sc_module, feature_specs, offsets={'table_a': 0, 'table_b': 0}
    )
    variables = self._initialize_model_variables(
        sc_module, embedding_lookup_input, embedding_variables
    )

    activations = jax.jit(sc_module.apply)(
        variables,
        embedding_lookup_input,
    )

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

    params_updates = jax.jit(
        functools.partial(sc_module.apply, method=sc_module.apply_gradient),
    )(
        variables,
        activations_grad,
        embedding_lookup_input,
    )

    # Updates params with the new embedding variables.
    assert len(params_updates) == 1
    tree.assert_same_structure(
        params_updates[_EMBED_PARAM], variables['params'][_EMBED_PARAM].value
    )
    variables['params'] = variables['params'] | params_updates

    expected_grad_table_a = np.full(
        (padded_vocab_a, padded_dim_a), _PAD_VALUE, dtype=np.float32
    )
    expected_grad_table_b = np.full(
        (padded_vocab_b, padded_dim_b), _PAD_VALUE, dtype=np.float32
    )

    for i, array in enumerate(embedding_variables['table_a'].table):
      col_id = array[0]
      new_col_id = col_id - (count_num(self.input_tensor, col_id) * 0.01)
      expected_grad_table_a[i, :_DIM_A] = np.full(
          (1, _DIM_A), new_col_id, dtype=np.float32
      )

    for i, array in enumerate(embedding_variables['table_b'].table):
      col_id = array[0]
      new_col_id = col_id - (
          count_num(self.input_tensor_table_b, col_id) * 0.01
      )
      expected_grad_table_b[i, :_DIM_B] = np.full(
          (1, _DIM_B), new_col_id, dtype=np.float32
      )
    np.testing.assert_equal(
        variables['params'][_EMBED_PARAM]['table_a'].table,
        expected_grad_table_a,
    )
    np.testing.assert_equal(
        variables['params'][_EMBED_PARAM]['table_b'].table,
        expected_grad_table_b,
    )

  @parameterized.named_parameters(
      dict(testcase_name='_with_minibatching', enable_minibatching=True),
      dict(testcase_name='_without_minibatching', enable_minibatching=False),
  )
  def test_pipelined_forward_and_backward(self, enable_minibatching: bool):
    devices = jax.devices()
    num_sc_per_device = utils.num_sparsecores_per_device(devices[0])

    feature_specs = (self.feature_spec_a,)
    embedding.prepare_feature_specs_for_training(
        feature_specs,
        global_device_count=jax.device_count(),
        num_sc_per_device=num_sc_per_device,
    )

    sc_module = embed.SparseCoreEmbed(
        feature_specs=feature_specs,
        enable_minibatching=enable_minibatching,
        enable_pipelining=True,
    )

    inp0 = sc_module.preprocess_inputs(
        0,
        (self.input_tensor,),
        (self.input_weights,),
    )
    inp1 = sc_module.preprocess_inputs(
        1,
        (self.input_tensor,),
        (self.input_weights,),
    )
    inp2 = sc_module.preprocess_inputs(
        2,
        (self.input_tensor,),
        (self.input_weights,),
    )

    padded_vocab_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_vocab_size
    )
    padded_dim_a = (
        self.feature_spec_a.table_spec.setting_in_stack.padded_embedding_dim
    )

    embedding_variables = self._create_embedding_variables(
        sc_module, feature_specs, offsets={'table_a': 0}
    )
    variables = self._initialize_model_variables(
        sc_module, inp0, embedding_variables
    )

    # step 0
    apply_fn = jax.jit(sc_module.apply, static_argnames=['mutable'])
    apply_grad_fn = jax.jit(
        functools.partial(sc_module.apply, method=sc_module.apply_gradient),
        static_argnames=['mutable'],
    )

    activations0, variables = apply_fn(variables, inp0, mutable=True)
    # Check activations are 0s
    np.testing.assert_allclose(
        activations0[0], jnp.zeros((_BATCH_SIZE, _DIM_A))
    )

    grad0 = (jnp.ones((_BATCH_SIZE, _DIM_A), dtype=jnp.float32),)
    _, variables = apply_grad_fn(variables, grad0, inp0, mutable=True)

    # step 1
    activations1, variables = apply_fn(variables, inp1, mutable=True)
    # in step 1, activations should be from lookup of step 0.
    expected_emb_activations_0 = np.broadcast_to(
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
    np.testing.assert_allclose(activations1[0], expected_emb_activations_0)

    grad1 = (jnp.ones((_BATCH_SIZE, _DIM_A), dtype=jnp.float32),)
    _, variables = apply_grad_fn(variables, grad1, inp1, mutable=True)

    # step 2
    activations2, variables = apply_fn(variables, inp2, mutable=True)
    # in step 2, activations should be from lookup of step 1, which used
    # embedding table updated with step 0 gradients.
    expected_activations_2_list = []
    for i in range(len(self.input_tensor)):
      val = 0
      for embedding_id in self.input_tensor[i]:
        val += embedding_id - count_num(self.input_tensor, embedding_id) * 0.01
      expected_activations_2_list.append([val])
    expected_activations_2 = np.broadcast_to(
        np.array(expected_activations_2_list, dtype=np.float32),
        (_BATCH_SIZE, _DIM_A),
    )
    np.testing.assert_allclose(
        activations2[0], expected_activations_2, rtol=1e-6
    )

    grad2 = (jnp.ones((_BATCH_SIZE, _DIM_A), dtype=jnp.float32),)
    _, variables = apply_grad_fn(variables, grad2, inp2, mutable=True)

    # In step 2 __call__, table update using grad0 and inp0 has been performed.
    # The updated table is in variables['params'][_EMBED_PARAM]

    expected_grad_table_a = np.full(
        (padded_vocab_a, padded_dim_a), _PAD_VALUE, dtype=np.float32
    )

    for i, array in enumerate(embedding_variables['table_a'].table):
      col_id = array[0]
      if col_id != _PAD_VALUE:
        new_col_id = col_id - 2 * (count_num(self.input_tensor, col_id) * 0.01)
        expected_grad_table_a[i, :_DIM_A] = np.full(
            (1, _DIM_A), new_col_id, dtype=np.float32
        )
    np.testing.assert_allclose(
        variables['params'][_EMBED_PARAM].value['table_a'].table,
        expected_grad_table_a,
        rtol=1e-6,
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
    mesh = jax.sharding.Mesh(devices, sharding_axis)
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
        enable_minibatching=enable_minibatching,
    )
    step = 42
    embedding_lookup_input = sc_module.preprocess_inputs(
        step,
        (self.input_tensor, self.input_tensor),
        (self.input_weights, self.input_weights),
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
    padded_dim_c = (
        self.feature_spec_c.table_spec.setting_in_stack.padded_embedding_dim
    )
    stacked_vocab_size = padded_vocab_a + padded_vocab_c
    stacked_embedding_dim = padded_dim_a

    emb_table_a = self._row_initialize_with_padding(
        shape=(_VOC_A, _DIM_A), padded_shape=(padded_vocab_a, padded_dim_a)
    )
    emb_table_c = self._row_initialize_with_padding(
        shape=(_VOC_C, _DIM_C),
        padded_shape=(padded_vocab_c, padded_dim_c),
        offset=200,
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
    sharding = NamedSharding(mesh, P('x', None))
    embedding_variables['table_a_table_c'] = (
        embedding.EmbeddingVariables(
            table=jax.make_array_from_single_device_arrays(
                shape=(stacked_vocab_size, padded_dim_a),
                sharding=sharding,
                arrays=embedding_variables['table_a_table_c'],
            ),
            slot=embedding_spec.SGDSlotVariables(),
        )
    )

    variables = self._initialize_model_variables(
        sc_module, embedding_lookup_input, core.freeze(embedding_variables)
    )

    activations = jax.jit(sc_module.apply)(
        variables,
        embedding_lookup_input,
    )

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

    params_updates = jax.jit(
        functools.partial(sc_module.apply, method=sc_module.apply_gradient),
    )(
        variables,
        activations_grad,
        embedding_lookup_input,
    )

    # Updates params with the new embedding variables.
    assert len(params_updates) == 1
    tree.assert_same_structure(
        params_updates[_EMBED_PARAM], variables['params'][_EMBED_PARAM].value
    )
    variables['params'] = variables['params'] | params_updates

    expected_grad_table_ac = np.full(
        (stacked_vocab_size, stacked_embedding_dim),
        _PAD_VALUE,
        dtype=np.float32,
    )

    for i, array in enumerate(
        embedding_variables['table_a_table_c'].table
    ):
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
        variables['params'][_EMBED_PARAM]['table_a_table_c'].table,
        expected_grad_table_ac,
    )


if __name__ == '__main__':
  absltest.main()
