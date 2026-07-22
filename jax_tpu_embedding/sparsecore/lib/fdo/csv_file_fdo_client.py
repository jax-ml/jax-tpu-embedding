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
"""An FDO client implementation that uses CSV files as storage."""

import collections
import csv
import dataclasses
import glob
import itertools
import os
import re
import time
from typing import Mapping

from absl import logging
from etils import epath
import jax
from jax_tpu_embedding.sparsecore.lib.fdo import fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
import numpy as np

_FILE_NAME = 'fdo_stats'
_FILE_EXTENSION = 'csv'
_PARAM_FIELDS = dataclasses.fields(embedding.SparseDenseMatmulInputStats)


class CSVFileFDOClient(fdo_client.FDOClient):
  """FDO client that writes stats to a file in .csv format.

  Usage:
    # Create a FDO client.
    client = CSVFileFDOClient(base_dir='/path/to/base/dir', retain_history=True)

    # Record observed stats from sparse input processing
    _, stats = embedding.preprocess_sparse_dense_matmul_input(...)
    client.record(stats)

    # Publish process local stats to a file.
    client.publish()

    # Load stats from all files in the base_dir.
    stats = client.load()
  """

  def __init__(self, base_dir: epath.PathLike, retain_history: bool = True):
    self._base_dir = epath.Path(base_dir)
    self._retain_history = retain_history
    # We store the params in a dict for easy updation and as an intermediate
    # format between SparseDenseMatmulInputStats and separate files.
    # param_name -> table_name -> stats
    self._params: dict[str, dict[str, np.ndarray]] = {
        field.name: collections.defaultdict(lambda: np.zeros(0, dtype=np.int32))
        for field in _PARAM_FIELDS
    }

  def record(self, data: embedding.SparseDenseMatmulInputStats) -> None:
    """Records stats per process.

    Accumulates the max ids observed per process per sparsecore per device for
    each embedding table.
    Accumulates the max unique ids observed per process per sparsecore per
    device for each embedding table.
    Args:
      data: A mapping representing data to be recorded.
    """
    # We convert the dataclass to dict for easy traversal.
    for param_name, param_value in dataclasses.asdict(data).items():
      if param_name not in self._params:
        logging.warning('Unsupported FDO stats: %s', param_name)
        continue
      for table_name, stats in param_value.items():
        logging.vlog(
            2,
            'Recording observed %s for table: %s -> %s',
            param_name,
            table_name,
            stats,
        )
        stats = np.asarray(stats, dtype=np.int64)
        if table_name not in self._params[param_name]:
          self._params[param_name][table_name] = stats
        else:
          self._params[param_name][table_name] = np.vstack(
              (self._params[param_name][table_name], stats)
          )

    if not self._retain_history:

      def _take_axis_max(x: np.ndarray):
        if not np.isscalar(x) and x.ndim > 1:
          return np.max(x, axis=0)
        return x

      for param_name, param_value in self._params.items():
        self._params[param_name] = {
            k: _take_axis_max(v) for k, v in param_value.items()
        }

  def _generate_file_name(self) -> str:
    """Generates a file name for the stats."""
    # File Format: `fdo_stats_<process_id>_<timestamp>.csv`
    filename = '{}_{}_{}.{}'.format(
        _FILE_NAME, jax.process_index(), time.time_ns(), _FILE_EXTENSION
    )
    return os.fspath(self._base_dir / filename)

  def _get_latest_files_by_process(self, files: list[str]) -> list[str]:
    """Returns the latest file for each process."""
    if not files:
      return []
    dir_path = epath.Path(files[0]).parent
    dir_prefix = len(os.fspath(dir_path)) + 1
    pattern = rf'{_FILE_NAME}_(\d+)_(\d+)\.{_FILE_EXTENSION}'
    file_groups = []
    for file in files:
      match = re.search(pattern, file[dir_prefix:])
      if match:
        file_groups.append((match.group(1), int(match.group(2)), file))
    if not file_groups:
      return []
    # Sort the files in descending order to get latest files first.
    file_groups = sorted(file_groups, reverse=True)
    latest_files = []
    for _, file_group in itertools.groupby(file_groups, key=lambda x: x[0]):
      # Get the first item in the group since already sorted by timestamp
      _, _, file_name = next(file_group)
      latest_files.append(file_name)
    return latest_files

  def publish(self) -> None:
    """Publishes locally accumulated stats to a file in the base_dir."""
    merged_stats = {}
    for field in _PARAM_FIELDS:
      for table_name, stats in self._params[field.name].items():
        merged_stats[f'{table_name}{field.metadata["suffix"]}'] = stats

    file_name = self._generate_file_name()
    logging.info('Write stats to %s', file_name)

    with open(file_name, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['key', 'values'])
      for key, array in merged_stats.items():
        array = array.astype(np.int64)
        if array.ndim == 1:
          writer.writerow([key, ' '.join(map(str, array))])
        else:
          for row in array:
            writer.writerow([key, ' '.join(map(str, row))])

  def _read_from_file(self, files_glob: str) -> Mapping[str, np.ndarray]:
    """Reads stats from files matching files_glob."""
    files = self._get_latest_files_by_process(glob.glob(files_glob))
    if not files:
      raise FileNotFoundError('No stats files found in %s' % files_glob)

    stats = collections.defaultdict(lambda: np.zeros(0, dtype=np.int32))
    for file_name in files:
      logging.info('Reading stats from %s', file_name)
      file_data = collections.defaultdict(list)
      with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
          if not row:
            continue
          key, vals_str = row
          vals = np.fromstring(vals_str, sep=' ', dtype=int)
          file_data[key].append(vals)

      for key, vals_list in file_data.items():
        array = np.vstack(vals_list) if len(vals_list) > 1 else vals_list[0]
        if stats.get(key) is None:
          stats[key] = array
        else:
          stats[key] = np.max(np.vstack((stats[key], array)), axis=0)
    return stats

  def load(self) -> embedding.SparseDenseMatmulInputStats:
    """Loads state of local FDO client from disk."""
    files_glob = os.fspath(
        self._base_dir / '{}*.{}'.format(_FILE_NAME, _FILE_EXTENSION)
    )
    stats = self._read_from_file(files_glob)

    result: dict[str, dict[str, np.ndarray]] = {
        field.name: {} for field in _PARAM_FIELDS
    }
    for key, val in stats.items():
      valid_key = False
      for field in _PARAM_FIELDS:
        if key.endswith(field.metadata['suffix']):
          table_name = key[: -len(field.metadata['suffix'])]
          result[field.name][table_name] = val
          valid_key = True
          break
      if not valid_key:
        raise ValueError(
            f'Unexpected key: {key}, expected to end with'
            f' {[field.metadata["suffix"] for field in _PARAM_FIELDS]}'
        )
    self._params = result
    # Typeshed stubs reject dictionary unpacking of dict[str, np.ndarray] into
    # kwargs expecting dict[str, int].
    return embedding.SparseDenseMatmulInputStats(**result)  # pyrefly: ignore[bad-argument-type]
