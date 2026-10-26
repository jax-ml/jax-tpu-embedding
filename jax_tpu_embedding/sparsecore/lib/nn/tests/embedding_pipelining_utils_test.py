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
"""Unit tests for SparseCore embedding pipelining utility functions."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_pipelining_utils
import numpy as np

# The `get_initial_state_shapes` tests shard over a mesh. With one device, every
# sharding is fully replicated.
jax.config.update("jax_num_cpu_devices", 4)


class EmbeddingPipeliningUtilsTest(parameterized.TestCase):

  def test_safe_copy_jax_array(self):
    arr = jnp.array([1.0, 2.0, 3.0])
    copied = embedding_pipelining_utils._safe_copy(arr)
    self.assertIsInstance(copied, jax.Array)
    np.testing.assert_array_equal(copied, arr)

  def test_safe_copy_numpy_array(self):
    arr = np.array([4, 5, 6])
    copied = embedding_pipelining_utils._safe_copy(arr)
    self.assertIsInstance(copied, jax.Array)
    np.testing.assert_array_equal(copied, arr)

  def test_safe_copy_shape_dtype_struct(self):
    struct = jax.ShapeDtypeStruct(shape=(8, 16), dtype=jnp.float32)
    copied = embedding_pipelining_utils._safe_copy(struct)
    self.assertIs(copied, struct)

  def test_safe_copy_abstract_value(self):
    shaped_array = jax.core.ShapedArray(shape=(4,), dtype=jnp.int32)
    copied = embedding_pipelining_utils._safe_copy(shaped_array)
    self.assertIs(copied, shaped_array)

  @parameterized.named_parameters(
      ("none", None),
      ("int", 42),
      ("float", 3.14),
      ("str", "test"),
      ("tuple", (1, 2)),
  )
  def test_safe_copy_non_arrays(self, value):
    copied = embedding_pipelining_utils._safe_copy(value)
    self.assertEqual(copied, value)

  def test_get_initial_state_eval_shape(self):
    pipeline_input = embedding_pipelining_utils.CurrentStepInput(
        sparse_inputs=jnp.zeros((4, 8), dtype=jnp.float32),
        dense_inputs=jnp.zeros((4, 16), dtype=jnp.float32),
    )
    tc_train_state = {
        "params": jnp.zeros((16, 8), dtype=jnp.float32)
    }
    embedding_variables = {
        "table": jnp.zeros((100, 8), dtype=jnp.float32)
    }

    def sc_fwd_fn(sparse_inputs, embedding_variables):
      del embedding_variables
      return sparse_inputs, None

    def tc_fn(
        embedding_activations,
        dense_inputs,
        train_state,
        sc_fwd_aux=None,
    ):
      del dense_inputs, sc_fwd_aux
      emb_grad = embedding_activations
      out = jnp.zeros((4,), dtype=jnp.float32)
      return emb_grad, out, train_state, None

    state_shape = jax.eval_shape(
        functools.partial(
            embedding_pipelining_utils.get_initial_state,
            sc_fwd_function=sc_fwd_fn,
            tc_function=tc_fn,
        ),
        pipeline_input,
        tc_train_state,
        embedding_variables,
    )
    self.assertEqual(state_shape.pipeline_step.shape, ())
    self.assertEqual(
        state_shape.step_before_last_step_inputs.sparse_inputs.shape, (4, 8)
    )

  @parameterized.named_parameters(("unsharded", False), ("sharded", True))
  def test_get_initial_state_shapes_match_arrays(self, sharded):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x")
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    batch = 2 * len(jax.devices())
    pipeline_input = embedding_pipelining_utils.CurrentStepInput(
        sparse_inputs=jax.device_put(
            jnp.zeros((batch, 8), jnp.float32), data_sharding
        ),
        dense_inputs={
            "committed": jax.device_put(
                jnp.zeros((batch, 16), jnp.float32), data_sharding
            ),
            "committed_weak": jax.device_put(jnp.asarray(1.0), replicated),
            "uncommitted_scalar": jnp.zeros((), jnp.float32),
            "numpy": np.zeros((batch, 4), np.int32),
            "python_int": 3,
            "python_float": 1.0,
        },
    )
    tc_train_state = {
        "params": jax.device_put(jnp.zeros((16, 8), jnp.float32), replicated),
        # The trainer step is weakly typed.
        "step": jax.device_put(jnp.asarray(0), replicated),
    }

    def sc_fwd_fn(sparse_inputs, embedding_variables):
      del embedding_variables
      return sparse_inputs * 2, jnp.zeros((), jnp.float32)

    def tc_fn(
        embedding_activations, dense_inputs, train_state, sc_fwd_aux=None
    ):
      del sc_fwd_aux
      loss = jnp.sum(embedding_activations @ train_state["params"].T)
      out = {"loss": loss + dense_inputs["python_int"], "grads": train_state}
      return embedding_activations, out, train_state, train_state["step"]

    input_sharding = data_sharding if sharded else None
    get_state = functools.partial(
        embedding_pipelining_utils.get_initial_state,
        pipeline_input=pipeline_input,
        tc_train_state=tc_train_state,
        embedding_variables={"table": jnp.zeros((100, 8), jnp.float32)},
        sc_fwd_function=sc_fwd_fn,
        tc_function=tc_fn,
        dense_input_sharding=input_sharding,
        sparse_input_sharding=input_sharding,
    )
    state = get_state()
    shapes = embedding_pipelining_utils.get_initial_state_shapes(
        *get_state.args, **get_state.keywords
    )

    def assert_matches(path, value, spec):
      aval = jax.typeof(value)
      self.assertEqual(
          (aval.shape, aval.dtype, aval.weak_type),
          (spec.shape, spec.dtype, spec.weak_type),
          msg=jax.tree_util.keystr(path),
      )
      if spec.sharding is not None:
        self.assertEqual(
            value.sharding, spec.sharding, msg=jax.tree_util.keystr(path)
        )
      elif isinstance(value, jax.Array):
        self.assertFalse(value.committed, msg=jax.tree_util.keystr(path))

    jax.tree.map_with_path(assert_matches, state, shapes)
    # The weak train step does not make the placeholders weak.
    self.assertFalse(shapes.placeholder_tc_aux.weak_type)

  def test_get_initial_state_shapes_from_abstract_inputs(self):
    pipeline_input = embedding_pipelining_utils.CurrentStepInput(
        sparse_inputs=jax.ShapeDtypeStruct((4, 8), jnp.float32),
        dense_inputs=jax.ShapeDtypeStruct((4, 16), jnp.float32),
    )
    tc_train_state = {
        "params": jax.ShapeDtypeStruct((16, 8), jnp.float32),
        "step": jax.ShapeDtypeStruct((), jnp.int32, weak_type=True),
    }

    def sc_fwd_fn(sparse_inputs, embedding_variables):
      del embedding_variables
      return sparse_inputs, None

    def tc_fn(
        embedding_activations, dense_inputs, train_state, sc_fwd_aux=None
    ):
      del dense_inputs, sc_fwd_aux
      return embedding_activations, {}, train_state, train_state["step"]

    shapes = embedding_pipelining_utils.get_initial_state_shapes(
        pipeline_input=pipeline_input,
        tc_train_state=tc_train_state,
        embedding_variables={},
        sc_fwd_function=sc_fwd_fn,
        tc_function=tc_fn,
    )
    self.assertEqual(
        shapes.last_step_inputs.embedding_activations,
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
    )
    self.assertFalse(shapes.placeholder_tc_aux.weak_type)

  def test_get_initial_state_with_pipeline_output_sharding_skips_compile(self):
    pipeline_input = embedding_pipelining_utils.CurrentStepInput(
        sparse_inputs=jnp.zeros((4, 8), dtype=jnp.float32),
        dense_inputs=jnp.zeros((4, 16), dtype=jnp.float32),
    )
    tc_train_state = {"params": jnp.zeros((16, 8), dtype=jnp.float32)}
    embedding_variables = {"table": jnp.zeros((100, 8), dtype=jnp.float32)}

    def sc_fwd_fn(sparse_inputs, embedding_variables):
      del embedding_variables
      return sparse_inputs, None

    def tc_fn(
        embedding_activations,
        dense_inputs,
        train_state,
        sc_fwd_aux=None,
    ):
      del dense_inputs, sc_fwd_aux
      emb_grad = embedding_activations
      out = {
          "loss": jnp.zeros((), dtype=jnp.float32),
          "grads": {"w": jnp.zeros((4, 8), dtype=jnp.float32)},
          "unused": None,
      }
      return emb_grad, out, train_state, None

    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("data",))
    replicated_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    pipeline_output_sharding = {
        "loss": None,
        "grads": replicated_sharding,
        "unused": {"nested": replicated_sharding},
    }

    with mock.patch.object(
        embedding_pipelining_utils,
        "_eval_shape",
        side_effect=AssertionError("_eval_shape should not be called"),
    ):
      state = embedding_pipelining_utils.get_initial_state(
          pipeline_input=pipeline_input,
          tc_train_state=tc_train_state,
          embedding_variables=embedding_variables,
          sc_fwd_function=sc_fwd_fn,
          tc_function=tc_fn,
          pipeline_output_sharding=pipeline_output_sharding,
      )

    self.assertEqual(state.placeholder_output["loss"].shape, ())
    self.assertEqual(state.placeholder_output["grads"]["w"].shape, (4, 8))
    self.assertIsNone(state.placeholder_output["unused"])

if __name__ == "__main__":
  absltest.main()
