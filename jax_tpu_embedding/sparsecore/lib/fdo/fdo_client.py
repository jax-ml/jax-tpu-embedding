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
"""Abstract interface for FDO client."""

import abc

from jax_tpu_embedding.sparsecore.lib.nn import embedding


class FDOClient(abc.ABC):
  """Abstract interface for FDO client.

  This class defines the interface for a per process client that interacts with
  the FDO system. An implementation of this class should define how the FDO
  stats are recorded and published to the storage location(disk, database,
  etc.). The load method should return the current aggregated (across all
  processes) stats from the storage location.

  Typical usage:
    1. Create an instance of an implementation of FDOClient.
    2. Call `record` to record the raw stats to the process local FDO client.
    3. (Optional) Repeat a few steps of training.
    3. Call `publish` on the singleton instance to publish the stats to the
    storage location.
    4. Call `load` on the singleton instance to get the aggregated (across all
    processes) stats from the storage location.
  """

  @abc.abstractmethod
  def record(
      self,
      data: embedding.SparseDenseMatmulInputStats,
  ) -> None:
    """Records the raw stats to local memory.

    An implementation of this method defines how the raw stats are processed and
    stored in preparation for publishing to the storage location.

    Args:
      data: Mapping of data stats to be recorded.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def publish(self) -> None:
    """Publishes stats to the storage location.

    An implementation of this method defines how the stats are published to the
    storage location. For instance, this could involve writing the stats to a
    file or updating a database.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def load(
      self,
  ) -> embedding.SparseDenseMatmulInputStats:
    """Loads state of local FDO client and returns the aggregated stats.

    An implementation of this method defines how the stats are aggregated across
    all processes. For instance, this could involve reading the stats from all
    files written by `publish` or a database and then aggregating them.

    Returns:
      A tuple of (max_ids, max_uniques, required_buffer_size) where max_ids
      is a mapping of table name to max ids per partition, max_uniques is a
      mapping of table name to max unique ids per partition, and
      required_buffer_sizes is a mapping of table name to required buffer size
      per sparsecore.
    """
    raise NotImplementedError
