# Copyright 2022 The jax_tpu_embedding Authors.
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

"""Input utils supports."""

import collections
import itertools
from typing import Any, Callable, NamedTuple, Optional, Tuple, Iterator, Sequence, Dict, List

from flax import jax_utils
import jax
from jax.experimental import multihost_utils
from jax.experimental import PartitionSpec
from jax.experimental.maps import Mesh
from jax_tpu_embedding import pytype_utils
import tensorflow as tf
import tree


Array = jax.Array
PyTree = Any
Shape = Tuple[int, ...]

Nested = pytype_utils.Nested
NestedTfTensor = pytype_utils.NestedTfTensor
TensorType = pytype_utils.TensorType
FeatureConfig = pytype_utils.FeatureConfig


def prepare_devices_data(xs: NestedTfTensor) -> Nested[jax.numpy.ndarray]:
  """Converts device input batches to numpy and split it among local devices.

  Each element of xs should be batched, it will be split into data parallel onto
  local devices.

  Args:
    xs: batches to be reshaped (tree-map'able). It can only be tf.Tensor as
      tf.SparseTensor cannot be converted to numpy().

  Raises:
    ValueError: If any element is not a tf.Tensor.
  Returns:
    re-shaped converted xs in (local_devices, device_batch_size, ...)
  """
  local_device_count = jax.local_device_count()

  def _shard(x):
    if not isinstance(x, tf.Tensor):
      raise ValueError('Value to shard is not a tf.Tensor.')
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (batch_size, ...) to
    # (local_devices, device_batch_size, ...)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_shard, xs)


def make_pmap_array_fn(
    devices: Optional[Sequence[jax.xla.Device]] = None
) -> Callable[..., Nested[Array]]:
  """Example function of creating jax.Array for pmap from local host data.

  Note that, this is a user define function. For this example, we assume user
  provides an iterator yields tf.Tensor, which can be customized.

  Args:
    devices: the list of devices to which the arrays should be put. Defaults to
      the order of devices expected by `jax.pmap`.

  Returns:
    A function takes inputs to devices, returns converted shards of such inputs,
    and put along devices which returns nested structure of jax.Array.
  """

  # TODO(zhonglinhan): remove this when jax.Array is fully used as default.
  if not jax.config.jax_array:
    raise ValueError('Must use jax array as default.')

  devices = devices or jax_utils._pmap_device_order()  # pylint: disable=protected-access

  def _put_sharded(xs):
    return jax.device_put_sharded(list(xs), devices)

  def _create_array_fn(xs: NestedTfTensor) -> Nested[Array]:
    xs = prepare_devices_data(xs)
    return jax.tree_map(_put_sharded, xs)

  return _create_array_fn


def make_pjit_array_fn(
    global_mesh: Mesh,
    pspecs: Nested[PartitionSpec]) -> Callable[..., Nested[Array]]:
  """Example function of creating jax.Array from local host data.

  Args:
    global_mesh: global device mesh that includes data dimension.
    pspecs: partition specs specifies resultant jax.Array.

  Returns:
    A callable function returns a PyTree of jax.Array.
  """

  def _tensor_to_array(x: tf.Tensor) -> jax.numpy.ndarray:
    if not isinstance(x, tf.Tensor):
      raise ValueError('Value to shard is not a tf.Tensor.')
    return x._numpy()  # pylint: disable=protected-access

  def _create_jax_array_fn(xs: NestedTfTensor) -> Nested[Array]:
    host_arrays = jax.tree_util.tree_map(_tensor_to_array, xs)
    return multihost_utils.host_local_array_to_global_array(
        host_arrays, global_mesh, pspecs)
  return _create_jax_array_fn


def split_and_prefetch_to_host_and_devices(
    iterator: Iterator[Any],
    split_fn: Callable[..., Dict[str, Any]],
    host_input_fn: Callable[..., Optional[Any]],
    device_input_fn: Callable[..., Any],
    buffer_size: int = 2):
  """Split input batches for host and devices and prefetch in a local queue.

  This utility takes an iterator and returns a new iterator which fills an on
  prefetch buffer. It has three major steps: 1) mapping yields of iterator with
  split_fn which defines host and device inputs, they are treated differently
  in following. 2) convert and shard device inputs with `prepare_devices_data`
  3) put host and converted device input to prefetch buffer.

  This utility is needed when host side need several steps ahead to feed in
  corresponding inputs than devices.

  Args:
    iterator: an iterator that yields batched inputs for host and devices, first
      dimension of each element is batch size.
    split_fn: A callable function takes elements from iterator, yields splits
      pytree of host and device batches in a dictionary as {'host': ...,
      'device': ...}.
    host_input_fn: a function to be applied on 'host' split from outputs from
      `split_fn`, it can yields converted results of host input split or may not
      need to yield anything when it's only feed input to host.
    device_input_fn: a function that takes 'device' split from `split_fn`,
      convert to device input to jax.Array along applicable devices to be put.
    buffer_size: the size of the prefetch buffer, default number is 2 to avoid
      unnecessary memory allocation.

  Yields:
    A dictionary has `host` and `device`, mapping to inputs as same pytree in
    split_fn yields. Elements of host inputs are original items from the
    iterator, they can be used for host ops like enqueue. Each element of device
    inputs is ndarray sharded to the specified devices.
  """
  iterator = map(split_fn, iterator)
  queue = collections.deque()

  def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(iterator, n):
      queue.append({
          'host': host_input_fn(data['host']),
          'device': device_input_fn(data['device'])
      })

  enqueue(buffer_size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def shard_inputs(inputs: Nested[TensorType],
                 num_shards: int) -> List[Nested[TensorType]]:
  """Shards inputs with same tree def.

  Given inputs with any nested structure, split them into shards with same
  structure on their first dimensions or batch dimensions. `tf.split` and
  requires tf.Tensor inputs' first dimensions can be evenly divided by
  `num_shards`.

  Args:
    inputs: A nested structure of tf.Tensor or tf.SparseTensor.
    num_shards: number of shards to split into for each inputs.

  Raises:
    ValueError: when first dimension of tf.Tensor input is not divisible by
      `num_shards`; or when non-tf.Tensor, non-tf.SparseTensor is given, such
      as tf.RaggedTensor.

  Returns:
    A list of sharded inputs with same original nested structure.
  """
  flatten_inputs, inputs_tree_def = jax.tree_util.tree_flatten(inputs)

  def _shard_fn(xs: TensorType) -> List[TensorType]:
    """Shard inputs from first dimension or batch dimension."""

    if isinstance(xs, tf.Tensor):
      if xs.shape.dims[0] % num_shards != 0:
        raise ValueError('Number of shards should evenly divide the first '
                         'dimension for tf.Tensor inputs.')
      return tf.split(xs, num_or_size_splits=num_shards, axis=0)
    elif isinstance(xs, tf.SparseTensor):
      return tf.sparse.split(xs, num_split=num_shards, axis=0)
    else:
      raise ValueError('Input must be tf.Tensor or tf.SparseTensor.')

  shards = jax.tree_util.tree_map(_shard_fn, flatten_inputs)
  sharded_inputs = []
  for per_shard in zip(*shards):
    sharded_inputs.append(
        jax.tree_util.tree_unflatten(inputs_tree_def, per_shard))
  return sharded_inputs


def tree_flatten_with_names(
    pytree: PyTree) -> Tuple[Sequence[Tuple[str, Any]], PyTree]:
  """Flatten a python tree structure with path name of each leaf.

  Args:
    pytree: A python tree structure.

  Returns:
    - A list of leaf name and value pairs: [(name, value), ...].
    - A tree definition object representing the structure of the flattened tree.
  """

  paths_and_vals = tree.flatten_with_path(pytree)
  paths, vals = zip(*paths_and_vals)

  # Concatenate path to a leaf with '/'.
  names = ('/'.join(str(node) for node in path) for path in paths)

  return list(zip(names, vals)), jax.tree_util.tree_structure(pytree)


class EnqueueOpInput(NamedTuple):
  """Inputs to feed into tpu embedding enqueue op.

  A tuple of attributes converted from input example to feed tpu embedding
  enqueue op. The original input example can be a tf.Tensor or tf.SparseTensor.

  Attributes:
    indices: A tf.Tensor, when input example is tf.SparseTensor, this attribute
      specifies nonzero elements' indices. This attribute is zero tensor only
      when input example is tf.Tensor.
    values: A rank-1 tf.Tensor, when input example is tf.SparseTensor this field
      supplies the values for each element specified in `indices`. While it will
      be reshaped to 1-D from original example when it's tf.Tensor.
    aggregation_weight: A tf.Tensor containing of this training example weight.
  """
  indices: tf.Tensor
  values: tf.Tensor
  aggregation_weight: tf.Tensor


def _process_sparse_data(sparse_input: tf.SparseTensor,
                         weight: Optional[tf.SparseTensor],
                         feature_config: FeatureConfig,
                         path_name: str) -> EnqueueOpInput:
  """Process sparse data input to create input for enqueue op.

  Args:
    sparse_input: A tf.SparseTensor input example.
    weight: If None, set aggregation weights to shape(0,) tensor. It indicates
      all aggregation weights are taken to be 1.0 in backend.
    feature_config: Corresponding tf.tpu.experimental.embedding.FeatureConfig.
    path_name: A path name string for `feature_config`, which is
      represents it's leaf path in its original pytree.

  Raises:
    ValueError: when weight is not a tf.SparseTensor or not None.

  Returns:
    An EnqueueOpInput.
  """
  # Corresponding weight must be NoneType or SparseTensor.
  if not isinstance(weight, (type(None), tf.SparseTensor)):
    raise ValueError(
        f'Weight for {path_name} is not NoneType or tf.SparseTensor for sparse '
        'input.')
  sample_weight = (
      tf.zeros((0,), dtype=tf.float32) if weight is None else tf.cast(
          weight.values, dtype=tf.float32))

  sample_indices_tensor = tf.cast(sparse_input.indices, dtype=tf.int32)

  # Add one dimension to the last axis when tensor shape is rank=2.
  if sparse_input.shape.rank == 2:
    if (not feature_config.output_shape and
        feature_config.max_sequence_length > 0):
      sample_indices_tensor = tf.pad(
          sample_indices_tensor, paddings=[[0, 0], [0, 1]])
  sample_values_tensor = tf.cast(sparse_input.values, dtype=tf.int64)

  return EnqueueOpInput(indices=sample_indices_tensor,
                        values=sample_values_tensor,
                        aggregation_weight=sample_weight)


def prepare_data_to_enqueue(
    flatten_inputs: List[TensorType],
    flatten_weights: List[Optional[TensorType]],
    flatten_feature_configs: Sequence[Tuple[str, FeatureConfig]]
) -> Iterator[EnqueueOpInput]:
  """Convert flatten inputs and weights to feed enqueue op.

  For now we supports tf.Tensor and tf.SparseTensor inputs. tf.SparseTensor has
  attributes of indices and values, which are indices and values as output.
  As for tf.Tensor, there will have no indices tensor, while values are rank 1
  reshaped. Aggregation weights will only be applicable for tf.SparseTensor.

  Args:
    flatten_inputs: A list of flatten inputs for given device to enqueue. Each
      element could be tf.Tensor or tf.SparseTensor.
    flatten_weights: A list of corresponding weights values or None if no weight
      applicable. Weight will be set as tf zero tensor if it's None.
    flatten_feature_configs: A list of flatten feature configs with path string,
      a path string is a concatenated of leaves when flatting a tree structure
      of feature configs.

  Raises:
    ValueError: 1) when lengths of flatten data inputs/weights/feature_configs
      are not same. 2) when flatten weights is not None value for tf.Tensor
      3) Also when flatten inputs is not tf.Tensor or tf.SparseTensor.

  Yields:
    A named tuple EnqueueOpInput.
  """

  # Flatten inputs/weights/feature_configs should have same length.
  if not (len(flatten_inputs) == len(flatten_weights) ==
          len(flatten_feature_configs)):
    raise ValueError(
        f'All flatten data should have same length, but get '
        f'len(flatten_inputs) = {len(flatten_inputs)}, '
        f'len(flatten_weights) = {len(flatten_weights)},'
        f'len(flatten_feature_configs) = {len(flatten_feature_configs)}')

  for tensor_input, weight, feature_config_with_path in zip(
      flatten_inputs, flatten_weights, flatten_feature_configs):

    path_name, feature_config = feature_config_with_path

    if not isinstance(tensor_input, (tf.Tensor, tf.SparseTensor)):
      raise ValueError('Input tensor is neither tf.Tensor nor tf.SparseTensor.')

    elif isinstance(tensor_input, tf.Tensor):
      if weight is not None:
        raise ValueError(
            'Dense input should not have specified weight for input at {}. '
            .format(path_name))

      sample_values_tensor = tf.cast(
          tf.reshape(tensor_input, [-1]), dtype=tf.int64)

      # We have to supply a empty/zero tensor `zeros_int_tensor` for indices
      # to feed enqueue which is required for dense input i.e. tf.Tensor.
      # Aggregation weight in such case will be zeros as well.

      yield EnqueueOpInput(
          indices=tf.zeros((0,), dtype=tf.int32),
          values=sample_values_tensor,
          aggregation_weight=tf.zeros((0,), dtype=tf.float32))

    else:  # should be a tf.SparseTensor input example.
      yield _process_sparse_data(sparse_input=tensor_input,
                                 weight=weight,
                                 feature_config=feature_config,
                                 path_name=path_name)


def enqueue_prefetch(
    enqueue_fn: Callable[..., None]) -> Callable[..., Tuple[()]]:
  """Split host inputs for enqueue op of each device, and execute enqueue.

  Args:
    enqueue_fn: a callable function for tpu_embedding_layer to enqueue.

  Returns:
    A function call to shard inputs for each local device and execute enqueue.
  """

  tf_enqueue_fn = tf.function(enqueue_fn)

  def _prefetch_fn(xs: TensorType) -> Tuple[()]:
    """Apply prefetch input to tf enqueue function.

    Args:
      xs: A tf.Tensor or tf.SparseTensor input slice.

    Returns:
      Empty tuple as enqueue has no output. Prefetch function for host need has
      output to yield.
    """
    xs = shard_inputs(xs, num_shards=jax.local_device_count())
    tf_enqueue_fn(xs)
    return ()

  return _prefetch_fn
