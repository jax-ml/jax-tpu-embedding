# Copyright 2024 The jax_tpu_embedding Authors.
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

import asyncio
import dataclasses
import os
from typing import Any, Dict, Optional, Sequence, Union, cast

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorstore as ts


Index = tuple[slice, ...]
ParamInfo = ocp.type_handlers.ParamInfo
TypeHandler = ocp.type_handlers.TypeHandler
Shape = Union[tuple[int, int], tuple]


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
    if self.global_shape and self.global_shape[0] % self.num_shards != 0:
      raise ValueError(
          f'Only support evenly sharded, as total number of rows = '
          f'{self.global_shape[0]} and num_shards = {self.num_shards}')

    if self.data.shape:
      local_nrows = self.data.shape[0]
      # TODO(zhonglinhan): compute index when not evenly sharded.
      self.index = _get_index(local_nrows, self.shard_id)
    else:
      self.index = ()
    self.dtype = self.data.dtype
    self.local_shape = self.data.shape


async def _serialize(arr: GlobalHostArray, tspec, ts_context: ts.Context):
  """Writes a GlobalHostArray data into tensor store."""
  commit_futures = []

  if jax.process_index() == 0:
    open_future = ts.open(
        ts.Spec(tspec), create=True, open=True, context=ts_context
    )
    commit_futures.append(open_future)

  ts_writer = await ts.open(
      ts.Spec(tspec), open=True, assume_metadata=True, context=ts_context
  )

  write_future = ts_writer[arr.index].write(arr.data)
  await write_future.copy
  commit_futures += [write_future.commit]
  return commit_futures


async def _deserialize(
    restore_args: 'GlobalHostArrayRestoreArgs',
    tspec: Dict[str, Any],
    ts_context: ts.Context,
) -> GlobalHostArray:
  """Deserialize embedding tensor shard."""

  if restore_args is None:
    raise ValueError('Restore args for embedding checkpointing cannot be None.')

  tspec = ocp.type_handlers.get_cast_tspec_deserialize(tspec, restore_args)

  t = await ts.open(ts.Spec(tspec), open=True, context=ts_context)
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


def _array_metadata_from_tensorstore(
    t: Any, info: ParamInfo
) -> ocp.metadata.ArrayMetadata:
  # TODO(b/284185400): Set sharding property.
  return ocp.metadata.ArrayMetadata(
      name=info.name,
      directory=info.parent_dir,
      shape=t.shape,
      sharding=None,
      dtype=jnp.dtype(t.dtype.name),
  )


class GlobalHostArrayHandler(TypeHandler):
  """Serialize/deserialize logic to allow integration with PyTreeCheckpointHandler.
  """

  def typestr(self) -> str:
    return 'GlobalHostArray'

  async def metadata(
      self, infos: Sequence[ParamInfo]
  ) -> Sequence[ocp.metadata.ArrayMetadata]:
    open_ops = []
    for info in infos:
      tspec = ocp.type_handlers._get_json_tspec(
          info,
          use_ocdbt=info.is_ocdbt_checkpoint,
      )
      open_ops.append(
          ts.open(
              ts.Spec(tspec),
              open=True,
              context=info.ts_context,
          )
      )
    tensorstores = await asyncio.gather(*open_ops)
    return [
        _array_metadata_from_tensorstore(t, info)
        for t, info in zip(tensorstores, infos)
    ]

  async def serialize(
      self,
      values: Sequence[GlobalHostArray],
      infos: Sequence[ParamInfo],
      args: Optional[Sequence[ocp.SaveArgs]] = None,
  ) -> Sequence[ocp.Future]:
    """See superclass documentation."""
    args = args or [ocp.SaveArgs()] * len(values)
    ocp.type_handlers.check_input_arguments(args)

    async def _serialize_values():
      serialize_ops = []
      for value, info, arg in zip(values, infos, args):
        if not info.is_ocdbt_checkpoint:
          # TODO(b/340292264): Raise the following error instead.
          logging.warning('This handler should only support OCDBT format.')
        tspec = ocp.type_handlers.get_json_tspec_write(
            info=info,
            use_ocdbt=info.is_ocdbt_checkpoint,
            global_shape=value.global_shape,
            local_shape=value.local_shape,
            dtype=value.dtype,
            process_index=jax.process_index(),
            arg=arg,
        )

        tspec = ocp.type_handlers.get_cast_tspec_serialize(tspec, value, arg)
        serialize_ops.append(_serialize(value, tspec, info.ts_context))
      return await asyncio.gather(*serialize_ops)

    return await _serialize_values()

  async def deserialize(
      self,
      infos: Sequence[ParamInfo],
      args: Optional[Sequence['GlobalHostArrayRestoreArgs']] = None,
  ) -> Sequence[GlobalHostArray]:
    """See superclass documentation."""
    if args is None:
      raise ValueError('RestoreArgs cannot be None.')
    ocp.type_handlers.check_input_arguments(infos, args)

    async def _deserialize_values():
      deserialize_ops = []
      for info, arg in zip(infos, args):
        if not info.is_ocdbt_checkpoint:
          await ocp.type_handlers._assert_parameter_files_exist(  # pylint: disable=protected-access
              info.path, metadata_key=None, use_zarr3=info.use_zarr3
          )
        gha_restore_args = cast(GlobalHostArrayRestoreArgs, arg)
        if not isinstance(gha_restore_args, GlobalHostArrayRestoreArgs):
          raise TypeError(
              'Restore args must be of type GlobalHostArrayRestoreArgs'
          )
        tspec = ocp.type_handlers._get_json_tspec(
            info, use_ocdbt=info.is_ocdbt_checkpoint
        )
        tspec = ocp.type_handlers.get_cast_tspec_deserialize(tspec, arg)
        deserialize_ops.append(
            _deserialize(gha_restore_args, tspec, info.ts_context)
        )
      return await asyncio.gather(*deserialize_ops)

    return await _deserialize_values()


ocp.type_handlers.register_type_handler(
    GlobalHostArray, GlobalHostArrayHandler()
)


@dataclasses.dataclass
class GlobalHostArrayRestoreArgs(ocp.RestoreArgs):
  shard_id: Optional[int] = None
  num_shards: Optional[int] = None
  global_shape: Optional[Shape] = None
