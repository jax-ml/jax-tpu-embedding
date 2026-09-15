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
from typing import Mapping, override

from absl import logging
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
import numpy as np


class CSVFileFDOClient(file_fdo_client.BaseFileFDOClient):
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

  _FILE_EXTENSION: str = 'csv'

  @override
  def _write_to_file(self, stats: Mapping[str, np.ndarray]) -> None:
    """Writes stats to a csv file."""
    file_name = self._generate_file_name()
    logging.info('Write stats to %s', file_name)

    with open(file_name, 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['key', 'values'])
      for key, array in stats.items():
        array = array.astype(np.int64)
        if array.ndim == 1:
          writer.writerow([key, ' '.join(map(str, array))])
        else:
          for row in array:
            writer.writerow([key, ' '.join(map(str, row))])

  @override
  def _read_file(self, file_name: str) -> Mapping[str, np.ndarray]:
    """Reads stats from a single csv file."""
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

    return {
        key: np.vstack(vals_list) if len(vals_list) > 1 else vals_list[0]
        for key, vals_list in file_data.items()
    }
