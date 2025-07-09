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
"""Unit tests for file based FDO client."""

import os

from absl.testing import absltest
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
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
    stats = embedding.SparseDenseMatmulInputStats(
        max_ids_per_partition={"tab_one": np.array([10, 20, 30, 40])},
        max_unique_ids_per_partition={"tab_one": np.array([1, 2, 3, 4])},
        required_buffer_size_per_sc={},
    )
    fdo_client.record(stats)
    fdo_client.publish()
    loaded_stats = fdo_client.load()
    self._assert_stats_equal(
        loaded_stats.max_ids_per_partition, stats.max_ids_per_partition
    )
    self._assert_stats_equal(
        loaded_stats.max_unique_ids_per_partition,
        stats.max_unique_ids_per_partition,
    )
    self._assert_stats_equal(
        loaded_stats.required_buffer_size_per_sc,
        stats.required_buffer_size_per_sc,
    )

  def test_multiple_record(self):
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    stats = embedding.SparseDenseMatmulInputStats(
        max_ids_per_partition={"tab_one": np.array([10, 20, 30, 40])},
        max_unique_ids_per_partition={"tab_one": np.array([1, 2, 3, 4])},
        required_buffer_size_per_sc={"tab_one": np.array([256])},
    )
    fdo_client.record(stats)
    fdo_client.record(stats)  # intentional
    fdo_client.publish()
    stats = fdo_client.load()

    self._assert_stats_equal(
        stats.max_ids_per_partition,
        {"tab_one": np.array([[10, 20, 30, 40], [10, 20, 30, 40]])},
    )
    self._assert_stats_equal(
        stats.max_unique_ids_per_partition,
        {"tab_one": np.array([[1, 2, 3, 4], [1, 2, 3, 4]])},
    )
    self._assert_stats_equal(
        stats.required_buffer_size_per_sc, {"tab_one": np.array([[256], [256]])}
    )

  def test_load_multiple_files(self):
    base_dir = self.create_tempdir().full_path
    np.savez(
        os.path.join(base_dir, "fdo_stats_0_10.npz"),
        **{
            "t_one_max_ids": np.array([10, 20, 30, 40]),
            "t_two_max_ids": np.array([50, 60, 70, 80]),
            "t_one_max_unique_ids": np.array([1, 2, 3, 4]),
            "t_two_max_unique_ids": np.array([5, 6, 7, 8]),
            "t_one_required_buffer_size": np.array([64]),
            "t_two_required_buffer_size": np.array([128]),
        },
    )
    np.savez(
        os.path.join(base_dir, "fdo_stats_1_11.npz"),
        **{
            "t_one_max_ids": np.array([20, 10, 40, 30]),
            "t_two_max_ids": np.array([60, 60, 80, 70]),
            "t_one_max_unique_ids": np.array([2, 1, 4, 3]),
            "t_two_max_unique_ids": np.array([6, 5, 8, 7]),
            "t_one_required_buffer_size": np.array([128]),
            "t_two_required_buffer_size": np.array([256]),
        },
    )

    fdo_client = file_fdo_client.NPZFileFDOClient(base_dir)
    loaded_stats = fdo_client.load()
    self._assert_stats_equal(
        loaded_stats.max_ids_per_partition,
        {
            "t_one": np.array([20, 20, 40, 40]),
            "t_two": np.array([60, 60, 80, 80]),
        },
    )
    self._assert_stats_equal(
        loaded_stats.max_unique_ids_per_partition,
        {
            "t_one": np.array([2, 2, 4, 4]),
            "t_two": np.array([6, 6, 8, 8]),
        },
    )
    self._assert_stats_equal(
        loaded_stats.required_buffer_size_per_sc,
        {"t_one": np.array([128]), "t_two": np.array([256])},
    )

  def test_files_not_found(self):
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    with self.assertRaises(FileNotFoundError):
      _ = fdo_client.load()

  def test_latest_files_by_process(self):
    files = [
        "temp/fdo_dumps/fdo_stats_0_10.npz",
        "temp/fdo_dumps/fdo_stats_0_20.npz",
        "temp/fdo_dumps/fdo_stats_0_30.npz",
        "temp/fdo_dumps/fdo_stats_1_09.npz",
        "temp/fdo_dumps/fdo_stats_0_40.npz",
        "temp/fdo_dumps/fdo_stats_2_10.npz",
    ]
    fdo_client = file_fdo_client.NPZFileFDOClient(self.base_dir)
    latest_files = fdo_client._get_latest_files_by_process(files)
    self.assertEqual(
        latest_files,
        [
            "temp/fdo_dumps/fdo_stats_2_10.npz",
            "temp/fdo_dumps/fdo_stats_1_09.npz",
            "temp/fdo_dumps/fdo_stats_0_40.npz",
        ],
    )


if __name__ == "__main__":
  absltest.main()
