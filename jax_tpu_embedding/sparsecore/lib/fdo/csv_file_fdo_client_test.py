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
"""Unit tests for CSV based FDO client."""

import csv

from absl.testing import absltest
from etils import epath
from jax_tpu_embedding.sparsecore.lib.fdo import csv_file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
import numpy as np


class CsvFdoClientTest(absltest.TestCase):

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

  def test_record_and_publish_load_full_history(self):
    fdo_client = csv_file_fdo_client.CSVFileFDOClient(
        self.base_dir, retain_history=True
    )
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

  def test_record_and_publish_load_retain_history_false(self):
    fdo_client = csv_file_fdo_client.CSVFileFDOClient(
        self.base_dir, retain_history=False
    )
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

  def test_multiple_record_full_history(self):
    fdo_client = csv_file_fdo_client.CSVFileFDOClient(
        self.base_dir, retain_history=True
    )
    stats = embedding.SparseDenseMatmulInputStats(
        max_ids_per_partition={"tab_one": np.array([10, 20, 30, 40])},
        max_unique_ids_per_partition={"tab_one": np.array([1, 2, 3, 4])},
        required_buffer_size_per_sc={"tab_one": np.array([256])},
    )
    fdo_client.record(stats)
    fdo_client.record(stats)
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

  def test_multiple_record_retain_history_false(self):
    fdo_client = csv_file_fdo_client.CSVFileFDOClient(
        self.base_dir, retain_history=False
    )
    stats1 = embedding.SparseDenseMatmulInputStats(
        max_ids_per_partition={"tab_one": np.array([10, 20, 30, 40])},
        max_unique_ids_per_partition={"tab_one": np.array([1, 2, 3, 4])},
        required_buffer_size_per_sc={"tab_one": np.array([256])},
    )
    stats2 = embedding.SparseDenseMatmulInputStats(
        max_ids_per_partition={"tab_one": np.array([5, 25, 15, 45])},
        max_unique_ids_per_partition={"tab_one": np.array([0, 3, 2, 5])},
        required_buffer_size_per_sc={"tab_one": np.array([512])},
    )
    fdo_client.record(stats1)
    fdo_client.record(stats2)
    fdo_client.publish()
    stats = fdo_client.load()

    self._assert_stats_equal(
        stats.max_ids_per_partition,
        {"tab_one": np.array([10, 25, 30, 45])},
    )
    self._assert_stats_equal(
        stats.max_unique_ids_per_partition,
        {"tab_one": np.array([1, 3, 3, 5])},
    )
    self._assert_stats_equal(
        stats.required_buffer_size_per_sc, {"tab_one": np.array([512])}
    )

  def test_load_multiple_files(self):
    base_dir = epath.Path(self.create_tempdir().full_path)

    # File 1
    with open(base_dir / "fdo_stats_0_10.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["key", "values"])
      writer.writerow(["t_one_max_ids", "10 20 30 40"])
      writer.writerow(["t_two_max_ids", "50 60 70 80"])
      writer.writerow(["t_one_max_unique_ids", "1 2 3 4"])
      writer.writerow(["t_two_max_unique_ids", "5 6 7 8"])
      writer.writerow(["t_one_required_buffer_size", "64"])
      writer.writerow(["t_two_required_buffer_size", "128"])

    # File 2
    with open(base_dir / "fdo_stats_1_11.csv", "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["key", "values"])
      writer.writerow(["t_one_max_ids", "20 10 40 30"])
      writer.writerow(["t_two_max_ids", "60 60 80 70"])
      writer.writerow(["t_one_max_unique_ids", "2 1 4 3"])
      writer.writerow(["t_two_max_unique_ids", "6 5 8 7"])
      writer.writerow(["t_one_required_buffer_size", "128"])
      writer.writerow(["t_two_required_buffer_size", "256"])

    fdo_client = csv_file_fdo_client.CSVFileFDOClient(base_dir)
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
    fdo_client = csv_file_fdo_client.CSVFileFDOClient(self.base_dir)
    with self.assertRaises(FileNotFoundError):
      _ = fdo_client.load()


if __name__ == "__main__":
  absltest.main()
