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

"""TPU Embedding CheckpointHandler."""

import dataclasses
import os
from typing import Any, Dict, List, Optional, cast

import jax
from jax.experimental.array_serialization.google.spec import get_tensorstore_spec
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax_ckpt
import tensorstore as ts

Index = tuple[slice, ...]
ParamInfo = orbax_ckpt.pytree_checkpoint_handler.ParamInfo
TypeHandler = orbax_ckpt.type_handlers.TypeHandler
Shape = tuple[int, int]
_DEFAULT_TS_CONTEXT = ts.Context({'file_io_concurrency': {'limit': 128}})


def _get_shard_size(nrows, num_shards, shard_id):
  if shard_id < nrows % num_shards:
    return nrows // num_shards + 1
  return nrows // num_shards


def _get_index(local_nrows, shard_id):
  start = local_nrows * shard_id
  end = local_nrows * (shard_id + 1)
  return (slice(start, end),)


@dataclasses.dataclass
class GlobalHostArray:
  """Host Array data contains sharding info and data for embeddings.

  global_shape: shape of embedding table.
  data: local shard data of embedding.
  shard_id: shard id of local shard, usually is client id.
  num_shards: total number of shards of embeddings, when using multi clients,
    this is number of clients.
  index: index info created by shard id, basically is starting and ending rows.
  dtype: data type of embedding to save to tensor store.
  local_shape: local shape of embedding tensor shard, which is inferred using
    `shard_id`, and `global_shape`.
  """
  global_shape: Shape
  data: np.ndarray
  shard_id: int
  num_shards: int
  index: Index = dataclasses.field(init=False)
  dtype: np.dtype = dataclasses.field(init=False)
  local_shape: Shape = dataclasses.field(init=False)

  def __post_init__(self):
    # TODO(zhonglinhan): need support uneven sharding.
    if self.global_shape[0] % self.num_shards != 0:
      raise ValueError(
          f'Only support evenly sharded, as total number of rows = '
          f'{self.global_shape[0]} and num_shards = {self.num_shards}')
    local_nrows = self.data.shape[0]

    # TODO(zhonglinhan): compute index when not evenly sharded.
    self.index = _get_index(local_nrows, self.shard_id)
    self.dtype = self.data.dtype
    self.local_shape = self.data.shape


@dataclasses.dataclass
class RestoreArgs(orbax_ckpt.RestoreArgs):
  shard_id: Optional[int] = None
  num_shards: Optional[int] = None
  global_shape: Optional[Shape] = None


def _get_metadata(arr: GlobalHostArray) -> Dict[str, Any]:
  dtype = np.dtype(arr.dtype).str
  return {
      'compressor': {
          'id': 'gzip'
      },
      'shape': arr.global_shape,
      'chunks': np.array(np.maximum(1, arr.local_shape)),
      'dtype': dtype,
  }


async def _serialize(arr: GlobalHostArray, tspec, commit_futures=None):
  """Writes a GlobalHostArray data into tensor store."""
  def _spec_has_metadata(tree):
    if not isinstance(tree, dict):
      return False
    return 'metadata' in tree or any(
        _spec_has_metadata(subtree) for _, subtree in tree.items())

  if not _spec_has_metadata(tspec):
    tspec['metadata'] = _get_metadata(arr)

  if jax.process_index() == 0:
    open_future = ts.open(
        ts.Spec(tspec),
        create=True,
        open=True,
        context=_DEFAULT_TS_CONTEXT)
    # Asynchronous case.
    if commit_futures is not None:
      assert isinstance(commit_futures, list)
      commit_futures.append(open_future)
    else:
      await open_future

  ts_writer = await ts.open(
      ts.Spec(tspec),
      open=True,
      assume_metadata=True,
      context=_DEFAULT_TS_CONTEXT)

  write_future = ts_writer[arr.index].write(arr.data)
  await write_future.copy
  return [write_future.commit]


async def _deserialize(restore_args: RestoreArgs,
                       tspec: Dict[str, Any]) -> GlobalHostArray:
  """Deserialize embedding tensor shard."""

  if restore_args is None:
    raise ValueError('Restore args for embedding checkpointing cannot be None.')

  if restore_args.dtype is not None:
    tspec = {
        'base': tspec,
        'driver': 'cast',
        'dtype': jnp.dtype(restore_args.dtype).name,
    }

  t = await ts.open(ts.Spec(tspec), open=True, context=_DEFAULT_TS_CONTEXT)
  global_shape = restore_args.global_shape
  global_shape = t.shape if global_shape is None else global_shape

  if restore_args.shard_id is None:
    raise ValueError('shard_id cannot be None.')
  if restore_args.num_shards is None:
    raise ValueError('num_shards cannot be None.')
  shard_id = restore_args.shard_id
  num_shards = restore_args.num_shards

  new_shard_size = _get_shard_size(
      nrows=global_shape[0], num_shards=num_shards, shard_id=shard_id)
  index = _get_index(
      local_nrows=new_shard_size,
      shard_id=shard_id)

  new_shard_shape = (new_shard_size,) + tuple(global_shape[1:])
  out = np.zeros(new_shard_shape, dtype=t.dtype.numpy_dtype)
  requested_domain = ts.IndexTransform(input_shape=global_shape)[index].domain
  restricted_domain = t.domain.intersect(requested_domain)

  await ts.array(out)[ts.d[:].translate_to[requested_domain.origin]
                     ][restricted_domain].write(t[restricted_domain])

  return GlobalHostArray(
      global_shape=global_shape,
      data=out,
      shard_id=shard_id,
      num_shards=num_shards)


class GlobalHostArrayHandler(TypeHandler):
  """Serialize/deserialize logic to allow integration with PyTreeCheckpointHandler.
  """

  def _get_json_tspec(
      self,
      info: ParamInfo,
      value: Optional[GlobalHostArray] = None) -> Dict[str, Any]:
    if info.path is None:
      raise ValueError('Must construct serialization path.')
    path = os.fspath(info.path)
    tspec = get_tensorstore_spec(path)
    if value is not None:
      tspec['metadata'] = _get_metadata(value)  # pylint: disable=protected-access
      # del tspec['metadata']['dtype']
    return tspec

  async def serialize(
      self,
      value: GlobalHostArray,
      info: ParamInfo,
      args: Optional[orbax_ckpt.SaveArgs] = None) -> List[orbax_ckpt.Future]:
    """See superclass documentation."""
    commit_futures = []
    await _serialize(
        value, self._get_json_tspec(info, value), commit_futures=commit_futures)
    return commit_futures

  async def deserialize(
      self,
      info: ParamInfo,
      args: Optional[orbax_ckpt.RestoreArgs] = None) -> GlobalHostArray:
    """See superclass documentation."""
    if args is None:
      raise ValueError('RestoreArgs cannot be None.')
    args = cast(RestoreArgs, args)
    arr = await _deserialize(args, self._get_json_tspec(info))
    return arr


orbax_ckpt.type_handlers.register_type_handler(GlobalHostArray,
                                               GlobalHostArrayHandler())
