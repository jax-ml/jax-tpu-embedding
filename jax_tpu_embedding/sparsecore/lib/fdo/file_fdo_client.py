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

"""An FDO client implementation that uses CSV files as storage."""

import collections
from collections.abc import Mapping
import glob
import os

from absl import logging
import jax
from jax import numpy as jnp
from jax_tpu_embedding.sparsecore.lib.fdo import fdo_client
import numpy as np


_FILE_NAME = 'fdo_stats'
_FILE_EXTENSION = 'npz'
_MAX_ID_STATS_KEY = '_max_ids'
_MAX_UNIQUE_ID_STATS_KEY = '_max_unique_ids'


class NPZFileFDOClient(fdo_client.FDOClient):
  """FDO client that writes stats to a file in .npz format.

  Usage:
    # Create a FDO client.
    client = NPZFileFDOClient(base_dir='/path/to/base/dir')

    # Record observed stats from sparse input processing
    max_ids_per_process, max_uniques_per_process =
      embedding.preprocess_sparse_dense_matmul_input(...)
    client.record(max_ids_per_process, max_uniques_per_process)

    # Publish process local stats to a file.
    client.publish()

    # Load stats from all files in the base_dir.
    max_ids_per_process, max_uniques_per_process = client.load()
  """

  def __init__(self, base_dir: str):
    self._base_dir = base_dir
    self._file = os.path.join(
        base_dir,
        '{}_{}.{}'.format(
            _FILE_NAME, str(jax.process_index()), _FILE_EXTENSION
        ),
    )
    self._max_ids_per_partition = collections.defaultdict(jnp.ndarray)
    self._max_unique_ids_per_partition = collections.defaultdict(jnp.ndarray)

  def record(self, data: Mapping[str, Mapping[str, jnp.ndarray]]) -> None:
    """Records stats per process.

    Accumulates the max ids observed per process per sparsecore per device for
    each embedding table.
    Accumulates the max unique ids observed per process per sparsecore per
    device for each embedding table.
    Args:
      data: A mapping representing data to be recorded.
    """
    if _MAX_ID_STATS_KEY[1:] not in data:
      raise ValueError('Expected stat (max_ids) not found.')
    max_ids_per_process = data[_MAX_ID_STATS_KEY[1:]]
    for table_name, stats in max_ids_per_process.items():
      logging.info(
          'Recording observed max ids for table: %s -> %s', table_name, stats
      )
      if table_name not in self._max_ids_per_partition:
        self._max_ids_per_partition[table_name] = stats
      else:
        self._max_ids_per_partition[table_name] = np.vstack(
            (self._max_ids_per_partition[table_name], stats)
        )
    if _MAX_UNIQUE_ID_STATS_KEY[1:] not in data:
      raise ValueError('Expected stats (max_unique_ids) not found.')
    max_uniques_per_process = data[_MAX_UNIQUE_ID_STATS_KEY[1:]]
    for table_name, stats in max_uniques_per_process.items():
      logging.vlog(
          2,
          'Recording observed max unique ids for table: %s -> %s',
          table_name,
          stats,
      )
      if table_name not in self._max_unique_ids_per_partition:
        self._max_unique_ids_per_partition[table_name] = stats
      else:
        self._max_unique_ids_per_partition[table_name] = np.vstack(
            (self._max_unique_ids_per_partition[table_name], stats)
        )

  def _write_to_file(
      self, stats: Mapping[str, jnp.ndarray], file_name: str
  ) -> None:
    """Writes stats to a npz file."""
    logging.info('Write stats to %s', file_name)
    jax.numpy.savez(file_name, **stats)

  def publish(self) -> None:
    """Publishes locally accmulatedstats to a file in the base_dir.

    Publish is called by each process there by collecting stats from all
    processes.
    """
    merged_stats = {
        f'{table_name}{_MAX_ID_STATS_KEY}': stats
        for table_name, stats in self._max_ids_per_partition.items()
    }
    merged_stats.update({
        f'{table_name}{_MAX_UNIQUE_ID_STATS_KEY}': stats
        for table_name, stats in self._max_unique_ids_per_partition.items()
    })
    self._write_to_file(merged_stats, self._file)

  def _read_from_file(
      self, glob_of_file_name: list[str]
  ) -> Mapping[str, jnp.ndarray]:
    """Reads stats from a npz file."""
    stats = collections.defaultdict(jnp.ndarray)
    if not glob_of_file_name:
      return stats
    for file_name in glob_of_file_name:
      logging.info('Reading stats from %s', file_name)
      loaded = np.load(file_name)
      loaded_d = {key: loaded[key] for key in loaded.files}
      for key, value in loaded_d.items():
        if stats.get(key) is None:
          stats[key] = value
        else:
          stats[key] = np.maximum(stats[key], value)
    return stats

  def load(
      self,
  ) -> tuple[Mapping[str, jnp.ndarray], Mapping[str, jnp.ndarray]]:
    """Loads state of local FDO client from disk.

    Reads all files in the base_dir and aggregates stats.
    Returns:
      A tuple of (max_ids_per_partition, max_unique_ids_per_partition)
    Raises:
      FileNotFoundError: If no stats files are found in the base_dir.
      ValueError: If the stats files do not have expected keys fro max ids and
      max unique ids.
    """
    # read files from base_dir and aggregate stats.
    files_glob = os.path.join(
        self._base_dir, '{}*.{}'.format(_FILE_NAME, _FILE_EXTENSION)
    )
    files = glob.glob(files_glob)
    if not files:
      raise FileNotFoundError('No stats files found in %s' % files_glob)
    stats = self._read_from_file(files)
    max_id_stats, max_unique_id_stats = {}, {}
    for table_name, stats in stats.items():
      if table_name.endswith(f'{_MAX_ID_STATS_KEY}'):
        max_id_stats[table_name[: -len(_MAX_ID_STATS_KEY)]] = stats
      elif table_name.endswith(f'{_MAX_UNIQUE_ID_STATS_KEY}'):
        max_unique_id_stats[table_name[: -len(_MAX_UNIQUE_ID_STATS_KEY)]] = (
            stats
        )
      else:
        raise ValueError(
            f'Unexpected table name and stats key: {table_name}, expected to'
            f' end with {_MAX_ID_STATS_KEY} or {_MAX_UNIQUE_ID_STATS_KEY}'
        )
    self._max_ids_per_partition = max_id_stats
    self._max_unique_ids_per_partition = max_unique_id_stats
    return (max_id_stats, max_unique_id_stats)
