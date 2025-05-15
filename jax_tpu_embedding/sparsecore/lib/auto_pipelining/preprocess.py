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
"""Preprocessor for autopipelining."""

import jax.extend as jex

from jax_tpu_embedding.sparsecore.lib.auto_pipelining import utils


def _has_permitive(eqn: jex.core.JaxprEqn, primitive_name_prefix: str) -> bool:
  """Checks if a JaxprEqn contains a primitive with the given prefix.

  This function recursively checks the equation and any nested Jaxprs (e.g., in
  conditionals or loops) for the presence of the primitive.

  Args:
    eqn: The JaxprEqn to check.
    primitive_name_prefix: The prefix of the primitive name to search for.

  Returns:
    True if the primitive is found, False otherwise.
  """
  if eqn.primitive.name.startswith(primitive_name_prefix):
    return True
  for param in eqn.params.values():
    if isinstance(param, jex.core.ClosedJaxpr) or isinstance(
        param, jex.core.Jaxpr
    ):
      if any(_has_permitive(eqn, primitive_name_prefix) for eqn in param.eqns):
        return True
  return False


def _has_embedding_lookup(eqn: jex.core.JaxprEqn) -> bool:
  """Checks if a JaxprEqn contains an embedding lookup operation."""
  return _has_permitive(eqn, utils.EMBEDDING_LOOKUP_PRIMITIVE_PREFIX)


def _has_embedding_update(eqn: jex.core.JaxprEqn) -> bool:
  """Checks if a JaxprEqn contains an embedding update operation."""
  return _has_permitive(eqn, utils.EMBEDDING_UPDATE_PRIMITIVE_PREFIX)


def _inline_custom_vjp(jaxpr: jex.core.Jaxpr) -> jex.core.Jaxpr:
  """Inlines embedding lookup inside custom_vjp_call_jaxpr."""
  eqns = []
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == 'custom_vjp_call' and _has_embedding_lookup(
        eqn
    ):
      eqns.extend(
          utils.inline_jaxpr(
              eqn.params['call_jaxpr'].jaxpr, eqn.invars, eqn.outvars
          )
      )
    else:
      eqns.append(eqn)
  return jaxpr.replace(eqns=eqns)


def _validate_embedding_lookup(eqn: jex.core.JaxprEqn) -> None:
  """Validates whether the embedding lookups can be transformed."""
  # shard_map should be on the top level so that we can combine them.
  assert (
      eqn.primitive.name == utils.SHARD_MAP_PRIMITIVE_NAME
  ), 'Embedding lookup should be wrapped directly by shard_map'
  # lookup primitive should be the first equation in the shard_map, for easy
  # check in the later transformations.
  jaxpr = eqn.params['jaxpr']
  lookup_eqn = jaxpr.eqns[0]
  assert lookup_eqn.primitive.name.startswith(
      utils.EMBEDDING_LOOKUP_PRIMITIVE_PREFIX
  ), (
      'The first equation in the shard_map is not an embedding lookup. '
      f'Got {lookup_eqn.primitive.name}'
  )

  # Embedding table should be the last input of the lookup shard_map.
  # Slot variables are not used for embedding lookup.
  assert (
      lookup_eqn.invars[-1] == jaxpr.invars[utils.EMBEDDING_LOOKUP_DATA_LEN]
  ), 'Embedding table should be the last input of the lookup shard_map'


def _validate_embedding_update(eqn: jex.core.JaxprEqn) -> None:
  """Validates whether the embedding updates can be transformed."""
  # shard_map should be on the top level so that we can combine them.
  assert (
      eqn.primitive.name == utils.SHARD_MAP_PRIMITIVE_NAME
  ), 'Embedding update should be wrapped directly by shard_map'
  # update primitive should be the last equation in the shard_map, for easy
  # check in the later transformations.
  jaxpr = eqn.params['jaxpr']
  update_eqn = jaxpr.eqns[-1]
  assert update_eqn.primitive.name.startswith(
      utils.EMBEDDING_UPDATE_PRIMITIVE_PREFIX
  ), (
      'The last equation in the shard_map is not an embedding update. '
      f'Got {update_eqn.primitive.name}'
  )

  # The embedding table should be the last input of the update shard_map.
  # Used when passing updates from dense to SC backward.
  embed_tables = jaxpr.invars[utils.EMBEDDING_UPDATE_DATA_LEN :]
  assert (
      update_eqn.invars[4:][: len(embed_tables)] == embed_tables
  ), 'Embedding table should be the last input of the update shard_map'

  # Embedding table should be the only output of the update shard_map.
  # This is used when we combine the lookup and update shard_maps.
  assert (
      update_eqn.outvars == jaxpr.outvars
  ), 'Embedding table should be the first output of the update shard_map'


def validate_jaxpr(jaxpr: jex.core.Jaxpr) -> None:
  """Validates the structure of the Jaxpr for auto-pipelining."""
  for eqn in jaxpr.eqns:
    has_lookup = _has_embedding_lookup(eqn)
    has_update = _has_embedding_update(eqn)
    assert not (
        has_lookup and has_update
    ), 'Embedding lookup and update should not be in the same equation'
    if has_lookup:
      _validate_embedding_lookup(eqn)
    if has_update:
      _validate_embedding_update(eqn)


def preprocess(jaxpr: jex.core.Jaxpr) -> jex.core.Jaxpr:
  """Preprocesses the Jaxpr for auto-pipelining."""
  jaxpr = _inline_custom_vjp(jaxpr)
  validate_jaxpr(jaxpr)
  return jaxpr
