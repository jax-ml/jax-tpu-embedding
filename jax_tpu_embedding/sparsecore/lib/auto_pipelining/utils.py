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


EMBEDDING_LOOKUP_PRIMITIVE_PREFIX = 'sparse_dense_matmul_csr'
EMBEDDING_UPDATE_PRIMITIVE_PREFIX = 'sparse_dense_matmul_grad'
CUSTOM_VJP_CALL_PRIMITIVE_NAME = 'custom_vjp_call_jaxpr'
SHARD_MAP_PRIMITIVE_NAME = 'shard_map'

# The number of data inputs of the embedding lookup shard_map.
# They are row_pointers, embedding_ids, sample_ids, gains.
EMBEDDING_LOOKUP_DATA_LEN = 4
# The number of data inputs of the embedding update shard_map.
# They are row_pointers, embedding_ids, sample_ids, gains, gradients.
EMBEDDING_UPDATE_DATA_LEN = EMBEDDING_LOOKUP_DATA_LEN + 1


def is_embedding_lookup(eqn: jax.core.JaxprEqn) -> bool:
  if eqn.primitive.name != SHARD_MAP_PRIMITIVE_NAME:
    return False
  jaxpr = eqn.params['jaxpr']
  return jaxpr.eqns[0].primitive.name.startswith(
      EMBEDDING_LOOKUP_PRIMITIVE_PREFIX
  )


def is_embedding_update(eqn: jax.core.JaxprEqn) -> bool:
  if eqn.primitive.name != SHARD_MAP_PRIMITIVE_NAME:
    return False
  jaxpr = eqn.params['jaxpr']
  return jaxpr.eqns[-1].primitive.name.startswith(
      EMBEDDING_UPDATE_PRIMITIVE_PREFIX
  )


def lookup_params(
    eqn: jax.core.JaxprEqn,
) -> tuple[list[jax.core.Atom], list[jax.core.Atom]]:
  return (
      eqn.invars[:EMBEDDING_LOOKUP_DATA_LEN],
      eqn.invars[EMBEDDING_LOOKUP_DATA_LEN:],
  )


def update_params(
    eqn: jax.core.JaxprEqn,
) -> tuple[list[jax.core.Atom], list[jax.core.Atom]]:
  return (
      eqn.invars[:EMBEDDING_UPDATE_DATA_LEN],
      eqn.invars[EMBEDDING_UPDATE_DATA_LEN:],
  )


def clone_vars(var_list: Iterable[jax.core.Var]) -> list[jax.core.Var]:
  return [jax.core.Var(var.suffix, var.aval) for var in var_list]


def inline_jaxpr(
    jaxpr: jax.core.Jaxpr,
    invars: list[jax.core.Var],
    outvars: list[jax.core.Var],
) -> list[jax.core.JaxprEqn]:
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

  def _translate_outvar(var: jax.core.Var) -> jax.core.Var:
    return var_mapping.setdefault(var, clone_vars([var])[0])

  def _translate_invar(var: jax.core.Var) -> jax.core.Var:
    return var if isinstance(var, jax.core.Literal) else var_mapping[var]

  return [
      eqn.replace(
          invars=[_translate_invar(var) for var in eqn.invars],
          outvars=[_translate_outvar(var) for var in eqn.outvars],
      )
      for eqn in jaxpr.eqns
  ]
