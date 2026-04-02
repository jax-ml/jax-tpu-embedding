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
"""Utils for auto pipelining."""

from collections.abc import Iterable
import itertools

import jax
import jax.extend as jex


EMBEDDING_LOOKUP_PRIMITIVE_PREFIX = 'sparse_dense_matmul_csr'
EMBEDDING_UPDATE_PRIMITIVE_PREFIX = 'sparse_dense_matmul_grad'
CUSTOM_VJP_CALL_PRIMITIVE_NAME = 'custom_vjp_call_jaxpr'
SHARD_MAP_PRIMITIVE_NAME = 'shard_map'

# The number of data inputs of the embedding lookup shard_map.
# They are row_pointers, embedding_ids, sample_ids, gains, num_minibatches.
EMBEDDING_LOOKUP_DATA_LEN = 5
# The number of data inputs of the embedding update shard_map.
# They are:
#   gradients, row_pointers, embedding_ids, sample_ids, gains, num_minibatches.
EMBEDDING_UPDATE_DATA_LEN = EMBEDDING_LOOKUP_DATA_LEN + 1


def is_embedding_lookup(eqn: jex.core.JaxprEqn) -> bool:
  if eqn.primitive.name != SHARD_MAP_PRIMITIVE_NAME:
    return False
  jaxpr = eqn.params['jaxpr']
  for sub_eqn in jaxpr.eqns:
    if sub_eqn.primitive.name.startswith(EMBEDDING_LOOKUP_PRIMITIVE_PREFIX):
      return True
  return False


def is_embedding_update(eqn: jex.core.JaxprEqn) -> bool:
  if eqn.primitive.name != SHARD_MAP_PRIMITIVE_NAME:
    return False
  jaxpr = eqn.params['jaxpr']
  for sub_eqn in jaxpr.eqns:
    if sub_eqn.primitive.name.startswith(EMBEDDING_UPDATE_PRIMITIVE_PREFIX):
      return True
  return False


def get_embedding_lookup_eqn(
    eqns: list[jex.core.JaxprEqn],
) -> jex.core.JaxprEqn:
  for eqn in eqns:
    if eqn.primitive.name.startswith(EMBEDDING_LOOKUP_PRIMITIVE_PREFIX):
      return eqn
  assert False, 'No embedding lookup found in the given eqns.'


def replace_embedding_lookup_eqn(
    eqns: list[jex.core.JaxprEqn], new_lookup_eqn: jex.core.JaxprEqn,
) -> list[jex.core.JaxprEqn]:
  result = []
  for eqn in eqns:
    if eqn.primitive.name.startswith(EMBEDDING_LOOKUP_PRIMITIVE_PREFIX):
      result.append(new_lookup_eqn)
    else:
      result.append(eqn)
  return result


def lookup_params(
    eqn: jex.core.JaxprEqn,
) -> tuple[list[jax.core.Atom], list[jax.core.Atom]]:
  return (
      eqn.invars[:EMBEDDING_LOOKUP_DATA_LEN],
      eqn.invars[EMBEDDING_LOOKUP_DATA_LEN:],
  )


def update_params(
    eqn: jex.core.JaxprEqn,
) -> tuple[list[jax.core.Atom], list[jax.core.Atom], list[jax.core.Atom]]:
  """Separates update shard_map inputs into data, tables, and unmapped."""
  jaxpr = eqn.params['jaxpr']

  # Find all update primitive calls.
  update_eqns = []
  for eq in jaxpr.eqns:
    if eq.primitive.name.startswith(EMBEDDING_UPDATE_PRIMITIVE_PREFIX):
      update_eqns.append(eq)

  if not update_eqns:
    raise ValueError('No embedding update primitive found in shard_map')

  # Table count mapping by optimizer name substring.
  optimizer_tables_count = {
      'adam': 3,
      'adagrad_momentum': 3,
      'adagrad': 2,
      'ftrl': 3,
      'sgd': 1,
      'laprop': 3,
      'f2a': 1,
  }

  prim_tables_invars = []
  for update_eqn in update_eqns:
    prim_name = update_eqn.primitive.name
    num_tables = 1
    for opt, count in optimizer_tables_count.items():
      if opt in prim_name:
        num_tables = count
        break
    prim_tables_invars.extend(update_eqn.invars[5 : 5 + num_tables])

  # Map inner variables to outer variables.
  tables = []
  for var in prim_tables_invars:
    if isinstance(var, jex.core.Var):
      try:
        index = jaxpr.invars.index(var)
        tables.append(eqn.invars[index])
      except ValueError:
        pass

  # Remove duplicates while preserving order.
  tables = list(dict.fromkeys(tables))

  # Find the indices of tables in eqn.invars
  table_indices = []
  for t in tables:
    try:
      table_indices.append(eqn.invars.index(t))
    except ValueError:
      pass

  if not table_indices:
    # Fallback to old logic if no tables found via primitives.
    all_tables_and_slots = eqn.invars[EMBEDDING_UPDATE_DATA_LEN:]
    in_specs = eqn.params['in_specs'][EMBEDDING_UPDATE_DATA_LEN:]
    tables = [
        v for v, s in zip(all_tables_and_slots, in_specs) if s is not None
    ]
    unmapped = [v for v, s in zip(all_tables_and_slots, in_specs) if s is None]
    return eqn.invars[:EMBEDDING_UPDATE_DATA_LEN], tables, unmapped

  min_table_idx = min(table_indices)
  data_inputs = eqn.invars[:min_table_idx]
  unmapped = [var for var in eqn.invars[min_table_idx:] if var not in tables]

  return data_inputs, tables, unmapped


def clone_vars(var_list: Iterable[jex.core.Var]) -> list[jex.core.Var]:
  return [jex.core.Var(var.aval) for var in var_list]


def inline_jaxpr(
    jaxpr: jex.core.Jaxpr,
    invars: list[jex.core.Var],
    outvars: list[jex.core.Var],
) -> list[jex.core.JaxprEqn]:
  """Inlines a jaxpr with given invars and outvars."""
  assert not set(jaxpr.invars).intersection(
      jaxpr.outvars
  ), 'Returning invars directly as outvars is not supported.'
  assert not jaxpr.constvars, 'Jaxpr with consts is not supported.'
  assert len(invars) == len(jaxpr.invars)
  assert len(outvars) == len(jaxpr.outvars)

  var_mapping = {
      var: val
      for var, val in itertools.chain(
          zip(jaxpr.invars, invars), zip(jaxpr.outvars, outvars)
      )
  }

  def _translate_outvar(var: jex.core.Var) -> jex.core.Var:
    return var_mapping.setdefault(var, clone_vars([var])[0])

  def _translate_invar(var: jex.core.Var) -> jex.core.Var:
    return var if isinstance(var, jex.core.Literal) else var_mapping[var]

  return [
      eqn.replace(
          invars=[_translate_invar(var) for var in eqn.invars],
          outvars=[_translate_outvar(var) for var in eqn.outvars],
      )
      for eqn in jaxpr.eqns
  ]
