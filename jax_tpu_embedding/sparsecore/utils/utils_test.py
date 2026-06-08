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


if __name__ == "__main__":
  absltest.main()
