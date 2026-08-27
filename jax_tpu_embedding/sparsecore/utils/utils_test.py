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
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np


class UtilsTest(parameterized.TestCase):

  def test_shard_emb_table(self):
    table = jnp.arange(8)[:, jnp.newaxis]  # 8x1

    expected = jnp.array([
        # Device 0
        [
            # SC 0: 4k+0
            [0],
            [4],
            # SC 1: 4k+1
            [1],
            [5],
        ],
        # Device 1
        [
            # SC 0: 4k+2
            [2],
            [6],
            # SC 1: 4k+3
            [3],
            [7],
        ],
    ])

    sharded = utils.shard_emb_table(table, num_devices=2, num_sc_per_device=2)
    np.testing.assert_array_equal(sharded, expected)

  def test_unshard_emb_table(self):
    num_devices = 4
    num_sc_per_device = 2
    vocab_size_per_sc = 2
    emb_dim = 4
    # Create an array of row indices: [0, 1, 2, ..., rows-1]
    rows = vocab_size_per_sc * num_devices * num_sc_per_device
    table = jnp.broadcast_to(
        jnp.arange(rows, dtype=jnp.float32)[:, jnp.newaxis], (rows, emb_dim)
    )
    sharded = utils.shard_emb_table(
        table, num_devices=num_devices, num_sc_per_device=num_sc_per_device
    )
    unsharded = utils.unshard_emb_table(sharded, num_sc_per_device)
    np.testing.assert_array_equal(unsharded, table)

  @parameterized.named_parameters(
      ('single_chip_baseline', 512, 1, 4, 1.0, 0.33),
      ('four_chips_multi_hot', 4096, 4, 4, 32.0, None),
      ('sixteen_chips_single_hot', 16384, 16, 4, 1.0, None),
      ('sixty_four_chips_multi_hot', 65536, 64, 4, 16.0, None),
  )
  def test_estimate_preprocessing_parameters_satisfies_compiler_invariant(
      self,
      global_batch_size: int,
      global_device_count: int,
      num_sc_per_device: int,
      valency: float,
      expected_buffer_ratio: float | None,
  ):
    # Arrange & Act
    params = utils.estimate_preprocessing_parameters(
        global_batch_size=global_batch_size,
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        valency=valency,
    )

    # Assert
    buffer_per_sc = (
        params['suggested_coo_buffer_size_per_device'] // num_sc_per_device
    )
    self.assertLessEqual(
        params['max_ids_per_partition'],
        buffer_per_sc,
    )
    self.assertEqual(
        params['suggested_coo_buffer_size_per_device']
        % (8 * num_sc_per_device),
        0,
    )
    self.assertEqual(params['max_ids_per_partition'] % 8, 0)
    self.assertEqual(params['max_unique_ids_per_partition'] % 8, 0)
    if expected_buffer_ratio is not None:
      buffer_ratio = buffer_per_sc / global_batch_size
      self.assertAlmostEqual(buffer_ratio, expected_buffer_ratio, delta=0.05)

  @parameterized.named_parameters(
      ('four_chips_valency_16', 512, 4, 4, 16.0),
      ('sixteen_chips_valency_32', 512, 16, 4, 32.0),
  )
  def test_estimate_preprocessing_parameters_statistical_accuracy(
      self,
      batch_size_per_sc: int,
      global_device_count: int,
      num_sc_per_device: int,
      valency: float,
  ):
    # Arrange
    rng = np.random.default_rng(42)
    global_sc_count = global_device_count * num_sc_per_device
    hot_shard_fraction = 0.20
    total_lookups_per_sc = int(batch_size_per_sc * valency)

    # Act
    params = utils.estimate_preprocessing_parameters(
        global_batch_size=(
            batch_size_per_sc * num_sc_per_device * global_device_count
        ),
        global_device_count=global_device_count,
        num_sc_per_device=num_sc_per_device,
        valency=valency,
        headroom_factor=1.0,
    )

    # Simulate 500 batches to obtain empirical 95th percentile peak
    hot_lookups = rng.binomial(
        total_lookups_per_sc, hot_shard_fraction, size=500
    )
    background_lookups = total_lookups_per_sc - hot_lookups
    observed_peaks = []
    for t in range(500):
      bg = rng.multinomial(
          background_lookups[t], [1.0 / global_sc_count] * global_sc_count
      )
      bg[0] += hot_lookups[t]
      observed_peaks.append(np.max(bg))
    emp_p95 = np.percentile(observed_peaks, 95)

    # Assert
    # Raw formula with 1.0x headroom tracks empirical P95 within 5% error
    pct_error = abs(params['max_ids_per_partition'] - emp_p95) / emp_p95
    self.assertLess(pct_error, 0.05)

  @parameterized.parameters(
      (-10, 1, 4, 1.0),
      (512, 0, 4, 1.0),
      (512, 1, 0, 1.0),
      (512, 1, 4, -1.0),
  )
  def test_estimate_preprocessing_parameters_invalid_arguments_raises(
      self,
      global_batch_size: int,
      global_device_count: int,
      num_sc_per_device: int,
      valency: float,
  ):
    # Arrange, Act & Assert
    with self.assertRaises(ValueError):
      utils.estimate_preprocessing_parameters(
          global_batch_size=global_batch_size,
          global_device_count=global_device_count,
          num_sc_per_device=num_sc_per_device,
          valency=valency,
      )


if __name__ == '__main__':
  absltest.main()
