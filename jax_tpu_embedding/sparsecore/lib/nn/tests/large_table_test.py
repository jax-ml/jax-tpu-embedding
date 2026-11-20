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
"""Tests SparseCore embedding table shards larger than 16 GiB per device."""

from collections.abc import Callable
import functools
import math
import os
import shutil

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import orbax.checkpoint as ocp

_ROWS_PER_DEVICE = 12_000_000
_EMBEDDING_DIM = 512
_ROWS_IN_16_GIB = (16 << 30) // (_EMBEDDING_DIM * 4)
# Just past 16 GiB per device so multi-device checkpoints fit in tmpfs.
_CHECKPOINT_ROWS_PER_DEVICE = _ROWS_IN_16_GIB * 33 // 32
_LEARNING_RATE = 0.5
_INITIAL_ACCUMULATOR = 0.1
_BETA_1 = 0.9
_BETA_2 = 0.999
_EPSILON = 1e-8
_HBM_HEADROOM_BYTES = 512 << 20


def _skip_if_insufficient_hbm(
    testcase: absltest.TestCase,
    device: jax.Device,
    rows_per_device: int,
    num_slots: int,
) -> None:
  stats = device.memory_stats()
  if not stats or "bytes_limit" not in stats:
    return
  table_bytes = rows_per_device * _EMBEDDING_DIM * 4
  required_bytes = (1 + num_slots) * table_bytes + _HBM_HEADROOM_BYTES
  if required_bytes > stats["bytes_limit"]:
    testcase.skipTest(
        f"Requires {required_bytes} bytes of HBM per device, got"
        f" {stats['bytes_limit']}."
    )


def _make_table_and_inputs(
    devices: list[jax.Device],
    rows_per_device: int,
    optimizer: embedding_spec.OptimizerSpec,
    seed: int = 0,
) -> tuple[
    Mesh,
    dict[str, embedding_spec.FeatureSpec],
    embedding.PreprocessedInput,
    dict[str, embedding.EmbeddingVariables],
    jax.Array,
]:
  """Builds sharded embedding variables and a batch probing rows past 16 GiB."""
  num_devices = len(devices)
  num_sc_per_device = utils.num_sparsecores_per_device(devices[0])
  mesh = Mesh(np.asarray(devices), "x")
  data_sharding = NamedSharding(mesh, P("x"))
  table_sharding = NamedSharding(mesh, P("x", None))

  rows = (
      _ROWS_IN_16_GIB,
      _ROWS_IN_16_GIB + 1,
      rows_per_device - 2,
      rows_per_device - 1,
  )
  per_device_batch = len(rows) * num_devices
  global_batch = per_device_batch * num_devices

  table_spec = embedding_spec.TableSpec(
      vocabulary_size=rows_per_device * num_devices,
      embedding_dim=_EMBEDDING_DIM,
      initializer=jax.nn.initializers.uniform(scale=1.0),
      optimizer=optimizer,
      combiner="sum",
      name="t",
      max_ids_per_partition=16,
      max_unique_ids_per_partition=16,
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=(global_batch, 1),
      output_shape=(global_batch, _EMBEDDING_DIM),
      name="f",
  )
  feature_specs = {"f": feature_spec}
  embedding.prepare_feature_specs_for_training(
      feature_specs,
      global_device_count=num_devices,
      num_sc_per_device=num_sc_per_device,
  )

  # Under MOD sharding, global ID `i` maps to SparseCore `i % num_shards` at
  # intra-SparseCore row `i // num_shards`, and each device concatenates its
  # `num_sc_per_device` SparseCore shards sequentially along axis 0.
  num_shards = num_devices * num_sc_per_device
  rows_per_shard = rows_per_device // num_sc_per_device
  per_device_ids = []
  for device_idx in range(num_devices):
    for row in rows:
      per_device_ids.append(
          (row % rows_per_shard) * num_shards
          + device_idx * num_sc_per_device
          + row // rows_per_shard
      )
  features = {
      "f": np.tile(
          np.asarray(per_device_ids, dtype=np.int64)[:, None],
          (num_devices, 1),
      )
  }
  preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
      features=features,
      features_weights=None,
      feature_specs=feature_specs,
      local_device_count=num_devices,
      global_device_count=num_devices,
      num_sc_per_device=num_sc_per_device,
      sharding_strategy="MOD",
  )
  preprocessed_inputs = jax.tree.map(
      lambda x: jax.device_put(x, data_sharding),
      preprocessed_inputs,
  )

  embedding_variables = dict(
      embedding.init_embedding_variables(
          rng=jax.random.key(seed),
          table_specs=embedding.get_table_specs(feature_specs),
          global_sharding=table_sharding,
          num_sparsecore_per_device=num_sc_per_device,
          bypass_mesh_check=num_devices != jax.device_count(),
      )
  )
  return (
      mesh,
      feature_specs,
      preprocessed_inputs,
      embedding_variables,
      jnp.asarray(rows, dtype=jnp.int32),
  )


def _make_forward_and_grad_fns(
    mesh: Mesh,
    feature_specs: dict[str, embedding_spec.FeatureSpec],
    embedding_variables: dict[str, embedding.EmbeddingVariables],
):
  num_devices = mesh.size
  data_pspec = P("x")
  table_pspec = P("x", None)
  data_sharding = NamedSharding(mesh, data_pspec)
  var_formats = jax.tree.map(lambda x: x.format, embedding_variables)

  forward_fn = jax.jit(
      jax.shard_map(
          functools.partial(
              embedding.tpu_sparse_dense_matmul,
              global_device_count=num_devices,
              feature_specs=feature_specs,
              sharding_strategy="MOD",
          ),
          mesh=mesh,
          in_specs=(data_pspec, table_pspec),
          out_specs=data_pspec,
          check_vma=False,
      ),
      out_shardings=data_sharding,
  )
  grad_fn = jax.jit(
      jax.shard_map(
          functools.partial(
              embedding.tpu_sparse_dense_matmul_grad,
              feature_specs=feature_specs,
              sharding_strategy="MOD",
          ),
          mesh=mesh,
          in_specs=(data_pspec, data_pspec, table_pspec),
          out_specs=table_pspec,
          check_vma=False,
      ),
      out_shardings=var_formats,
      donate_argnums=(2,),
  )
  return forward_fn, grad_fn


def _shard_rows(arr: jax.Array, row_indices: jax.Array) -> list[np.ndarray]:
  return [
      np.asarray(shard.data[jax.device_put(row_indices, shard.device)])
      for shard in arr.addressable_shards
  ]


class LargeTableTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="sgd_single_device",
          optimizer_factory=lambda: embedding_spec.SGDOptimizerSpec(
              learning_rate=_LEARNING_RATE
          ),
          use_all_devices=False,
      ),
      dict(
          testcase_name="adagrad_single_device",
          optimizer_factory=lambda: embedding_spec.AdagradOptimizerSpec(
              learning_rate=_LEARNING_RATE,
              initial_accumulator_value=_INITIAL_ACCUMULATOR,
          ),
          use_all_devices=False,
      ),
      dict(
          testcase_name="adam_single_device",
          optimizer_factory=lambda: embedding_spec.AdamOptimizerSpec(
              learning_rate=_LEARNING_RATE,
              beta_1=_BETA_1,
              beta_2=_BETA_2,
              epsilon=_EPSILON,
          ),
          use_all_devices=False,
      ),
      dict(
          testcase_name="sgd_all_devices",
          optimizer_factory=lambda: embedding_spec.SGDOptimizerSpec(
              learning_rate=_LEARNING_RATE
          ),
          use_all_devices=True,
      ),
      dict(
          testcase_name="adagrad_all_devices",
          optimizer_factory=lambda: embedding_spec.AdagradOptimizerSpec(
              learning_rate=_LEARNING_RATE,
              initial_accumulator_value=_INITIAL_ACCUMULATOR,
          ),
          use_all_devices=True,
      ),
      dict(
          testcase_name="adam_all_devices",
          optimizer_factory=lambda: embedding_spec.AdamOptimizerSpec(
              learning_rate=_LEARNING_RATE,
              beta_1=_BETA_1,
              beta_2=_BETA_2,
              epsilon=_EPSILON,
          ),
          use_all_devices=True,
      ),
  )
  def test_train_step_past_16_gib(
      self,
      optimizer_factory: Callable[[], embedding_spec.OptimizerSpec],
      use_all_devices: bool,
  ) -> None:
    if use_all_devices and jax.device_count() < 2:
      self.skipTest("Multi-device case requires at least 2 TPU devices.")
    devices = jax.devices() if use_all_devices else jax.devices()[:1]
    num_devices = len(devices)
    optimizer = optimizer_factory()
    _skip_if_insufficient_hbm(
        self, devices[0], _ROWS_PER_DEVICE, optimizer.slot_variables_count()
    )

    mesh, feature_specs, preprocessed_inputs, embedding_variables, rows = (
        _make_table_and_inputs(devices, _ROWS_PER_DEVICE, optimizer)
    )
    table = embedding_variables["t"].table
    for shard in table.addressable_shards:
      self.assertEqual(shard.data.shape, (_ROWS_PER_DEVICE, _EMBEDDING_DIM))

    aliased_rows = rows - _ROWS_IN_16_GIB
    before = _shard_rows(table, rows)
    aliased_before = _shard_rows(table, aliased_rows)

    forward_fn, grad_fn = _make_forward_and_grad_fns(
        mesh, feature_specs, embedding_variables
    )
    activations = forward_fn(preprocessed_inputs, embedding_variables)
    expected_per_device = np.concatenate(before, axis=0)
    expected_activations = np.tile(expected_per_device, (num_devices, 1))
    np.testing.assert_array_equal(
        np.asarray(activations["f"]), expected_activations
    )

    activations_grad = {
        "f": jax.device_put(
            jnp.ones_like(activations["f"]),
            NamedSharding(mesh, P("x")),
        )
    }
    embedding_variables = dict(
        grad_fn(activations_grad, preprocessed_inputs, embedding_variables)
    )
    updated_table = embedding_variables["t"].table
    updated_slots = embedding_variables["t"].slot

    grad = float(num_devices)
    if isinstance(optimizer, embedding_spec.SGDOptimizerSpec):
      step_delta = _LEARNING_RATE * grad
      expected_slots = ()
    elif isinstance(optimizer, embedding_spec.AdagradOptimizerSpec):
      acc = _INITIAL_ACCUMULATOR + grad**2
      step_delta = _LEARNING_RATE * grad / math.sqrt(acc)
      expected_slots = ((updated_slots[0], acc, _INITIAL_ACCUMULATOR),)
    else:
      momentum = (1.0 - _BETA_1) * grad
      velocity = (1.0 - _BETA_2) * (grad**2)
      c_2 = math.sqrt(1.0 - _BETA_2)
      alpha_t = _LEARNING_RATE * c_2 / (1.0 - _BETA_1)
      epsilon_hat = _EPSILON * c_2
      step_delta = alpha_t * momentum / (math.sqrt(velocity) + epsilon_hat)
      expected_slots = (
          (updated_slots[0], momentum, 0.0),
          (updated_slots[1], velocity, 0.0),
      )

    after = _shard_rows(updated_table, rows)
    aliased_after = _shard_rows(updated_table, aliased_rows)
    for d in range(num_devices):
      np.testing.assert_allclose(
          before[d] - after[d],
          np.full_like(before[d], step_delta),
          rtol=1e-5,
          atol=1e-5,
      )
      np.testing.assert_array_equal(aliased_after[d], aliased_before[d])

    for slot_arr, expected_val, init_val in expected_slots:
      slot_rows = _shard_rows(slot_arr, rows)
      slot_aliased = _shard_rows(slot_arr, aliased_rows)
      for d in range(num_devices):
        np.testing.assert_allclose(
            slot_rows[d],
            np.full_like(before[d], expected_val),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            slot_aliased[d], np.full_like(before[d], init_val)
        )

  @parameterized.named_parameters(
      dict(testcase_name="single_device", num_devices=1),
      dict(testcase_name="multi_device", num_devices=2),
  )
  def test_checkpoint_round_trip_past_16_gib(self, num_devices: int) -> None:
    if jax.device_count() < num_devices:
      self.skipTest(f"Requires at least {num_devices} TPU devices.")
    devices = jax.devices()[:num_devices]
    rows_per_device = (
        _CHECKPOINT_ROWS_PER_DEVICE if num_devices > 1 else _ROWS_PER_DEVICE
    )
    _skip_if_insufficient_hbm(self, devices[0], rows_per_device, num_slots=0)

    tmpdir = self.create_tempdir().full_path
    ckpt_bytes = rows_per_device * num_devices * _EMBEDDING_DIM * 4
    free_bytes = shutil.disk_usage(tmpdir).free
    if ckpt_bytes + (2 << 30) > free_bytes:
      self.skipTest(
          f"Requires {ckpt_bytes} bytes in tmpdir, got {free_bytes} free."
      )

    optimizer = embedding_spec.SGDOptimizerSpec(learning_rate=_LEARNING_RATE)
    mesh, feature_specs, preprocessed_inputs, embedding_variables, rows = (
        _make_table_and_inputs(devices, rows_per_device, optimizer, seed=0)
    )
    _, grad_fn = _make_forward_and_grad_fns(
        mesh, feature_specs, embedding_variables
    )
    global_batch = len(rows) * num_devices * num_devices
    activations_grad = {
        "f": jax.device_put(
            jnp.ones((global_batch, _EMBEDDING_DIM), dtype=jnp.float32),
            NamedSharding(mesh, P("x")),
        )
    }
    embedding_variables = dict(
        grad_fn(activations_grad, preprocessed_inputs, embedding_variables)
    )

    table = embedding_variables["t"].table
    aliased_rows = rows - _ROWS_IN_16_GIB
    saved = _shard_rows(table, rows)
    aliased_saved = _shard_rows(table, aliased_rows)
    saved_first_col = [
        np.asarray(s.data[:, 0]) for s in table.addressable_shards
    ]
    abstract_vars = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.format),
        embedding_variables,
    )

    ckpt_path = os.path.join(tmpdir, "ckpt")
    with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
      checkpointer.save(
          ckpt_path, args=ocp.args.StandardSave(embedding_variables)
      )
      del embedding_variables, table
      restored = checkpointer.restore(
          ckpt_path,
          args=ocp.args.StandardRestore(
              item=abstract_vars, support_layout=True
          ),
      )

    restored_table = restored["t"].table
    expected_format = utils.embedding_table_format(mesh, P("x", None))
    self.assertEqual(restored_table.format, expected_format)

    restored_rows = _shard_rows(restored_table, rows)
    restored_aliased = _shard_rows(restored_table, aliased_rows)
    for d, shard in enumerate(restored_table.addressable_shards):
      np.testing.assert_array_equal(
          np.asarray(shard.data[:, 0]), saved_first_col[d]
      )
      np.testing.assert_array_equal(restored_rows[d], saved[d])
      np.testing.assert_array_equal(restored_aliased[d], aliased_saved[d])

    restored = dict(grad_fn(activations_grad, preprocessed_inputs, restored))
    stepped_table = restored["t"].table
    stepped_rows = _shard_rows(stepped_table, rows)
    stepped_aliased = _shard_rows(stepped_table, aliased_rows)
    for d in range(num_devices):
      np.testing.assert_allclose(
          saved[d] - stepped_rows[d],
          np.full_like(saved[d], _LEARNING_RATE * num_devices),
          rtol=1e-5,
          atol=1e-5,
      )
      np.testing.assert_array_equal(stepped_aliased[d], aliased_saved[d])


if __name__ == "__main__":
  absltest.main()
