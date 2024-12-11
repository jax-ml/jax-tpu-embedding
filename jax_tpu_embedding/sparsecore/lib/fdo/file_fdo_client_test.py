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

"""Unit tests for file based FDO client."""

import os

from absl.testing import absltest
from absl.testing import absltest
import jax
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
import numpy as np


class NpzFdoClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_dir = self.create_tempdir(
        cleanup=absltest.TempFileCleanup.OFF
    ).full_path

  def _assert_stats_equal(self, actual, expected):
    self.assertLen(actual, len(expected))
    for key in expected:
      self.assertIn(key, actual)
      np.testing.assert_array_equal(expected[key], actual[key])

  def test_record_and_publish_load(self):
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    max_id_stats = {"tab_one": jax.numpy.array([10, 20, 30, 40])}
    max_unique_stats = {"tab_one": jax.numpy.array([1, 2, 3, 4])}
    fdo_client.record(
        {"max_ids": max_id_stats, "max_unique_ids": max_unique_stats}
    )
    fdo_client.publish()
    loaded_max_ids, loaded_max_uniques = fdo_client.load()
    self._assert_stats_equal(loaded_max_ids, max_id_stats)
    self._assert_stats_equal(loaded_max_uniques, max_unique_stats)

  def test_multiple_record(self):
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    fdo_client.record({
        "max_ids": {"tab_one": jax.numpy.array([10, 20, 30, 40])},
        "max_unique_ids": {"tab_one": jax.numpy.array([1, 2, 3, 4])},
    })
    fdo_client.record({
        "max_ids": {"tab_one": jax.numpy.array([10, 20, 30, 40])},
        "max_unique_ids": {"tab_one": jax.numpy.array([1, 2, 3, 4])},
    })
    fdo_client.publish()
    loaded_max_ids, loaded_max_uniques = fdo_client.load()

    self._assert_stats_equal(
        loaded_max_ids,
        {"tab_one": jax.numpy.array([[10, 20, 30, 40], [10, 20, 30, 40]])},
    )
    self._assert_stats_equal(
        loaded_max_uniques,
        {"tab_one": jax.numpy.array([[1, 2, 3, 4], [1, 2, 3, 4]])},
    )

  def test_load_multiple_files(self):
    base_dir = self.create_tempdir().full_path
    jax.numpy.savez(
        os.path.join(base_dir, "fdo_stats_10.npz"),
        **{
            "t_one_max_ids": jax.numpy.array([10, 20, 30, 40]),
            "t_two_max_ids": jax.numpy.array([50, 60, 70, 80]),
            "t_one_max_unique_ids": jax.numpy.array([1, 2, 3, 4]),
            "t_two_max_unique_ids": jax.numpy.array([5, 6, 7, 8]),
        },
    )
    jax.numpy.savez(
        os.path.join(base_dir, "fdo_stats_11.npz"),
        **{
            "t_one_max_ids": jax.numpy.array([20, 10, 40, 30]),
            "t_two_max_ids": jax.numpy.array([60, 60, 80, 70]),
            "t_one_max_unique_ids": jax.numpy.array([2, 1, 4, 3]),
            "t_two_max_unique_ids": jax.numpy.array([6, 5, 8, 7]),
        },
    )

    fdo_client = file_fdo_client.NPZFileFDOClient(base_dir)
    loaded_max_ids, loaded_max_uniques = fdo_client.load()
    self._assert_stats_equal(
        loaded_max_ids,
        {
            "t_one": jax.numpy.array([20, 20, 40, 40]),
            "t_two": jax.numpy.array([60, 60, 80, 80]),
        },
    )
    self._assert_stats_equal(
        loaded_max_uniques,
        {
            "t_one": jax.numpy.array([2, 2, 4, 4]),
            "t_two": jax.numpy.array([6, 6, 8, 8]),
        },
    )

  def test_files_not_found(self):
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    with self.assertRaises(FileNotFoundError):
      _, _ = fdo_client.load()


if __name__ == "__main__":
  absltest.main()
