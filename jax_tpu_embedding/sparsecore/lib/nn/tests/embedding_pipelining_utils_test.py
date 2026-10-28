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
from jax.experimental import layout
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_pipelining_utils
import numpy as np

# The `get_initial_state_shapes` tests shard over a mesh. With one device, every
# sharding is fully replicated.
jax.config.update("jax_num_cpu_devices", 4)


def _sc_fwd_fn(sparse_inputs, embedding_variables):
  del embedding_variables
  activations = sparse_inputs * 2
  # A weakly typed aux, as from a Python scalar.
  return activations, {"weak": jnp.asarray(0.5), "sum": activations.sum(1)}


def _tc_fn(embedding_activations, dense_inputs, train_state, sc_fwd_aux=None):
  del sc_fwd_aux
  loss = jnp.sum(embedding_activations @ train_state["params"].T)
  output = {
      "loss": loss + dense_inputs["python_int"],
      "weak": train_state["step"] * 1.0,
      "grads": train_state,
  }
  # The trainer passes the weakly typed train step as the TensorCore aux.
  return embedding_activations, output, train_state, train_state["step"]


def _get_initial_state_kwargs(sharded):
  """Returns the arguments of `get_initial_state` with each kind of input."""
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
          "committed_weak_scalar": jax.device_put(
              jnp.asarray(1.0), replicated
          ),
          "uncommitted_scalar": jnp.zeros((), jnp.float32),
          "uncommitted_weak_scalar": jnp.asarray(2.0),
          "uncommitted_weak_vector": jax.lax.broadcast(
              jnp.asarray(1.0), (batch,)
          ),
          "numpy": np.zeros((batch, 4), np.int32),
          "python_int": 3,
          "python_float": 1.0,
      },
  )
  input_sharding = data_sharding if sharded else None
  return dict(
      pipeline_input=pipeline_input,
      tc_train_state={
          "params": jax.device_put(
              jnp.zeros((16, 8), jnp.float32), replicated
          ),
          # The trainer step is weakly typed.
          "step": jax.device_put(jnp.asarray(0), replicated),
      },
      embedding_variables={"table": jnp.zeros((100, 8), jnp.float32)},
      sc_fwd_function=_sc_fwd_fn,
      tc_function=_tc_fn,
      dense_input_sharding=input_sharding,
      sparse_input_sharding=input_sharding,
  )


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
    kwargs = _get_initial_state_kwargs(sharded)
    state = embedding_pipelining_utils.get_initial_state(**kwargs)
    shapes = embedding_pipelining_utils.get_initial_state_shapes(**kwargs)

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

  @parameterized.named_parameters(("unsharded", False), ("sharded", True))
  def test_get_initial_state_weak_types(self, sharded):
    # A placeholder must have the weak type of the value that it stands for, so
    # that the step does not compile again for the first real input.
    kwargs = _get_initial_state_kwargs(sharded)
    state = embedding_pipelining_utils.get_initial_state(**kwargs)

    weak = {
        jax.tree_util.keystr(path)
        for path, x in jax.tree_util.tree_leaves_with_path(state)
        if isinstance(x, jax.Array) and jax.typeof(x).weak_type
    }
    expected = {
        ".pipeline_step",
        ".last_step_inputs.dense_inputs['committed_weak_scalar']",
        ".last_step_inputs.dense_inputs['uncommitted_weak_scalar']",
    }
    if not sharded:
      # `jnp.zeros_like` keeps the weak type, also of a Python scalar.
      expected |= {
          ".last_step_inputs.dense_inputs['uncommitted_weak_vector']",
          ".last_step_inputs.dense_inputs['python_int']",
          ".last_step_inputs.dense_inputs['python_float']",
      }
    # The placeholders that are not inputs are strongly typed, also when the
    # SparseCore or TensorCore function returns weakly typed values.
    self.assertEqual(weak, expected)
    if sharded:
      # A scalar input is its own placeholder.
      dense_inputs = kwargs["pipeline_input"].dense_inputs
      for name in (
          "committed_weak_scalar",
          "uncommitted_weak_scalar",
          "python_int",
          "python_float",
      ):
        self.assertIs(
            state.last_step_inputs.dense_inputs[name], dense_inputs[name]
        )

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

  def test_zeros_from_shape_with_layout_format(self):
    # On TPU, the embedding activations and gradients have the tiled layout of
    # the embedding table. A CPU test cannot make that layout, so it uses a
    # column-major layout, which is not the default and is lost if the format is
    # dropped. Other backends tile every array, so they cannot make it.
    if jax.default_backend() != "cpu":
      self.skipTest("The test layout is valid only on CPU.")
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x")
    )
    device_layout = layout.Layout(major_to_minor=(1, 0), tiling=())
    spec = jax.ShapeDtypeStruct(
        (2 * len(jax.devices()), 8),
        jnp.float32,
        sharding=layout.Format(device_layout, sharding),
    )

    zeros = embedding_pipelining_utils._zeros_from_shape(spec)

    self.assertEqual(zeros.shape, spec.shape)
    self.assertEqual(zeros.dtype, spec.dtype)
    self.assertFalse(jax.typeof(zeros).weak_type)
    self.assertEqual(
        zeros.format,  # pyrefly: ignore[missing-attribute]
        spec.format,
    )
    np.testing.assert_array_equal(zeros, np.zeros(spec.shape, np.float32))


if __name__ == "__main__":
  absltest.main()
