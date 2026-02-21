from collections.abc import Sequence
import enum
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np

class ShardingStrategy(enum.Enum):
  MOD = 1
  DIV = 2

class AllReduceInterface: ...

class MinibatchingNode:
  def __init__(
      self,
      task_id: int,
      num_tasks: int,
      server_addresses: list[str],
      num_local_devices: int,
  ) -> None: ...
  def get_all_reduce_interface(self) -> AllReduceInterface: ...

class SparseDenseMatmulInputStats:
  max_ids_per_partition: dict[str, np.ndarray]
  max_unique_ids_per_partition: dict[str, np.ndarray]
  required_buffer_sizes: dict[str, np.ndarray]
  dropped_id_count: dict[str, int]

# Type alias for the return tuple of the preprocessing functions
PreprocessOutput = tuple[
    dict[str, np.ndarray],  # lhs_row_pointers
    dict[str, np.ndarray],  # lhs_embedding_ids
    dict[str, np.ndarray],  # lhs_sample_ids
    dict[str, np.ndarray],  # lhs_gains
    int,  # num_minibatches
    SparseDenseMatmulInputStats,  # stats
]

def PreprocessSparseDenseMatmulInput(
    features: Sequence[np.ndarray],
    feature_weights: Sequence[np.ndarray] | None,
    feature_specs: Sequence[embedding_spec.FeatureSpec],
    local_device_count: int,
    global_device_count: int,
    *,
    num_sc_per_device: int,
    sharding_strategy: ShardingStrategy = ShardingStrategy.MOD,
    has_leading_dimension: bool = False,
    allow_id_dropping: bool = False,
    batch_number: int = 0,
    enable_minibatching: bool = False,
    all_reduce_interface: AllReduceInterface | None = None
) -> PreprocessOutput: ...
def PreprocessSparseDenseMatmulSparseCooInput(
    indices: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
    dense_shapes: Sequence[np.ndarray],
    feature_specs: Sequence[embedding_spec.FeatureSpec],
    local_device_count: int,
    global_device_count: int,
    *,
    num_sc_per_device: int,
    sharding_strategy: ShardingStrategy = ShardingStrategy.MOD,
    has_leading_dimension: bool = False,
    allow_id_dropping: bool = False,
    batch_number: int = 0,
    enable_minibatching: bool = False,
    all_reduce_interface: AllReduceInterface | None = None
) -> PreprocessOutput: ...
