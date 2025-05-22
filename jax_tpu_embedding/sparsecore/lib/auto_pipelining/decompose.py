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
"""Decomposer to split a Jaxpr into SparseCore forward, dense and SparseCore backward."""

import dataclasses
from typing import Any

import jax
import jax.extend as jex
from jax.interpreters import partial_eval as pe
from jax_tpu_embedding.sparsecore.lib.auto_pipelining import preprocess
from jax_tpu_embedding.sparsecore.lib.auto_pipelining import utils


Carry = Any
Activations = list[jax.Array]
Updates = list[jax.Array]


@dataclasses.dataclass
class FunctionRunner:
  """A helper class to run parts of a step function."""

  result_structure: jax.tree_util.PyTreeDef
  carry_structure: jax.tree_util.PyTreeDef
  literals: list[jex.core.Literal]

  lookup_jaxpr: jex.core.Jaxpr
  update_jaxpr: jex.core.Jaxpr
  update_lookup_jaxpr: jex.core.Jaxpr
  dense_jaxpr: jex.core.Jaxpr

  update_params_len: int
  activations_len: int

  def _run_jaxpr(self, jaxpr, *args):
    flatten_args = jax.tree.leaves(args)
    return jax.core.eval_jaxpr(jaxpr, self.literals, *flatten_args)

  def embedding_lookup(self, carry: Carry, *args) -> Activations:
    return self._run_jaxpr(self.lookup_jaxpr, carry, *args)

  def dense(self, activations: Activations, carry: Carry, *args):
    jaxpr_res = self._run_jaxpr(self.dense_jaxpr, activations, carry, *args)
    updates = jaxpr_res[: self.update_params_len]
    dense_results = jaxpr_res[self.update_params_len :]
    res = jax.tree.unflatten(self.result_structure, dense_results)
    return updates, *res

  def embedding_update(self, updates: Updates, carry: Carry) -> Carry:
    res = self._run_jaxpr(self.update_jaxpr, updates, carry)
    return jax.tree.unflatten(self.carry_structure, res)

  def embedding_update_lookup(
      self, updates: Updates, carry: Carry, *args
  ) -> tuple[Activations, Carry]:
    res = self._run_jaxpr(self.update_lookup_jaxpr, updates, carry, *args)
    activations = res[: self.activations_len]
    flatten_carry = res[self.activations_len :]
    return activations, jax.tree.unflatten(self.carry_structure, flatten_carry)

  def none_result(self):
    """Creates a result filled with None values, matching the output structure."""
    return jax.tree.unflatten(
        self.result_structure,
        [None] * self.result_structure.num_leaves,
    )


def _combine_shard_maps(
    lookup_eqn: jex.core.JaxprEqn,
    update_eqn: jex.core.JaxprEqn,
) -> jex.core.JaxprEqn:
  """Combines the lookup and update shard_maps into a single shard_map."""
  # The update primitive is the last equation in the Jaxpr.
  update_jaxpr = update_eqn.params['jaxpr']
  # The lookup primitive is the first equation in the Jaxpr.
  lookup_jaxpr = lookup_eqn.params['jaxpr']

  # Feed the updated embedding table to the lookup primitive.
  updated_tables = update_jaxpr.eqns[-1].outvars
  lookup_primitive_eqn = lookup_jaxpr.eqns[0]
  lookup_primitive_eqn = lookup_primitive_eqn.replace(
      invars=lookup_primitive_eqn.invars[:-1] + [updated_tables[0]]
  )

  # Combine the sub Jaxpr.
  jaxpr = jex.core.Jaxpr(
      constvars=update_jaxpr.constvars + lookup_jaxpr.constvars,
      invars=update_jaxpr.invars + lookup_jaxpr.invars[:-1],
      outvars=update_jaxpr.outvars + lookup_jaxpr.outvars,
      eqns=update_jaxpr.eqns + [lookup_primitive_eqn] + lookup_jaxpr.eqns[1:],
  )

  # Combine shard_map parameters.
  updated_params = update_eqn.params | {
      'jaxpr': jaxpr,
      'in_specs': (
          update_eqn.params['in_specs'] + lookup_eqn.params['in_specs'][:-1]
      ),
      'out_specs': (
          update_eqn.params['out_specs'] + lookup_eqn.params['out_specs']
      ),
  }

  # Combine the shard_map.
  return update_eqn.replace(
      params=updated_params,
      invars=update_eqn.invars + lookup_eqn.invars[:-1],
      outvars=update_eqn.outvars + lookup_eqn.outvars,
  )


def decompose(
    closed_jaxpr: jex.core.ClosedJaxpr,
    result_structure: jax.tree_util.PyTreeDef,
    carry_structure: jax.tree_util.PyTreeDef,
) -> FunctionRunner:
  """Decomposes a Jaxpr into SparseCore forward, dense and SparseCore backward.

  Args:
    closed_jaxpr: The ClosedJaxpr of the training step function.
    result_structure: The PyTree structure of the training step's output.
    carry_structure: The PyTree structure of the carry (training state).

  Returns:
    A FunctionRunner object that can execute the decomposed parts of the
    training step.
  """
  carry_len = carry_structure.num_leaves
  jaxpr = preprocess.preprocess(closed_jaxpr.jaxpr)

  activation_inputs: list[jax.core.Atom] = []
  activation_outputs: list[jax.core.Atom] = []

  update_param_inputs: list[jax.core.Atom] = []
  update_param_outputs: list[jax.core.Atom] = []

  lookups: dict[jax.core.Atom, jex.core.JaxprEqn] = {}
  updates: dict[jax.core.Atom, jex.core.JaxprEqn] = {}

  ##############################################################################
  # Construct a full jaxpr that takes / returns a different set of activations
  # and update params.
  ##############################################################################
  eqns = []
  emb_slot_vars = []
  for eqn in jaxpr.eqns:
    if utils.is_embedding_lookup(eqn):
      activation_inputs.extend(eqn.outvars)
      eqn = eqn.replace(outvars=utils.clone_vars(eqn.outvars))
      activation_outputs.extend(eqn.outvars)
      _, embedding_tables = utils.lookup_params(eqn)
      assert (
          embedding_tables[0] not in lookups
      ), 'Duplicate embedding lookups for the same table.'
      lookups[embedding_tables[0]] = eqn
    elif utils.is_embedding_update(eqn):
      # The last argument is excluded because it's the embedding table.
      update_params, embedding_tables = utils.update_params(eqn)
      emb_slot_vars.extend(embedding_tables)
      update_param_outputs.extend(update_params)
      cloned_params = utils.clone_vars(update_params)
      eqn = eqn.replace(invars=cloned_params + embedding_tables)
      update_param_inputs.extend(cloned_params)
      assert (
          embedding_tables[0] not in updates
      ), 'Duplicate embedding updates for the same table.'
      updates[embedding_tables[0]] = eqn
    eqns.append(eqn)

  full_jaxpr = jaxpr.replace(
      eqns=eqns,
      invars=activation_inputs + update_param_inputs + jaxpr.invars,
      outvars=activation_outputs + update_param_outputs + jaxpr.outvars,
      debug_info=None,
  )
  act_len = len(activation_inputs)
  update_len = len(update_param_inputs)
  invar_len = len(jaxpr.invars)
  outvar_len = len(jaxpr.outvars)

  ##############################################################################
  # Find the embedding tables in the input.
  ##############################################################################
  assert (
      lookups.keys() == updates.keys()
  ), 'Embedding updates and lookups must be paired.'
  del lookups
  # Create a temporary Jaxpr that returns all embedding tables and slot
  # variables.
  jaxpr_tables = jaxpr.replace(outvars=emb_slot_vars, debug_info=None)
  # Use dead code elimination to find which inputs are used to compute the
  # embedding tables.
  _, table_inputs = pe.dce_jaxpr(
      jaxpr_tables, used_outputs=tuple(True for _ in jaxpr_tables.outvars)
  )
  # Verify that embedding tables are in the carry.
  non_table_inputs = [not table for table in table_inputs]
  assert all(
      non_table_inputs[carry_len:]
  ), 'Embedding tables should all be in the carry.'

  # Identify outputs that correspond to embedding tables.
  table_outputs = table_inputs[:carry_len] + [False] * (outvar_len - carry_len)
  non_table_outputs = [not table for table in table_outputs]

  ##############################################################################
  # Build lookup jaxpr. It takes the same input but returns activations.
  ##############################################################################
  lookup_jaxpr = jaxpr.replace(outvars=activation_inputs, debug_info=None)
  lookup_jaxpr = pe.dce_jaxpr(
      lookup_jaxpr, tuple(True for _ in lookup_jaxpr.outvars), instantiate=True
  )[0]
  assert all(
      not utils.is_embedding_update(eqn) for eqn in lookup_jaxpr.eqns
  ), 'In the train step function, embedding lookup should before update.'

  ##############################################################################
  # Build dense. In addition to the original inputs and outputs, it takes
  # activations as input, and embedding updates as output.
  ##############################################################################
  inps = [True] * act_len + [False] * update_len + non_table_inputs
  oups = [False] * act_len + [True] * update_len + non_table_outputs
  dense_jaxpr, used_inputs = pe.dce_jaxpr(full_jaxpr, oups, inps)
  assert used_inputs == inps, 'Dense layer should not use the embedding table.'

  # Copy unmodified embedding tables from input to output.
  dense_jaxpr = dense_jaxpr.replace(
      invars=activation_inputs + jaxpr.invars,
      outvars=update_param_outputs
      + [
          invar if is_table else outvar
          for invar, outvar, is_table in zip(
              jaxpr.invars, jaxpr.outvars, table_outputs
          )
      ],
  )
  assert all(
      not utils.is_embedding_lookup(eqn) and not utils.is_embedding_update(eqn)
      for eqn in dense_jaxpr.eqns
  ), 'Dense layer should not use the embedding table.'

  ##############################################################################
  # Build lookup_update. It runs an embedding lookup then update.
  ##############################################################################
  inps = [False] * act_len + [True] * update_len + [True] * invar_len
  oups = [True] * act_len + [False] * update_len + table_outputs
  lookup_update_jaxpr, used_inputs = pe.dce_jaxpr(
      full_jaxpr, used_outputs=oups, instantiate=inps
  )
  assert (
      used_inputs == inps
  ), 'There should not be embedding table manipulation after update.'

  # Copy other part of carry from input to output.
  lookup_update_jaxpr = lookup_update_jaxpr.replace(
      outvars=activation_outputs
      + [
          outvar if is_table else invar
          for invar, outvar, is_table in zip(
              jaxpr.invars, jaxpr.outvars, table_outputs[:carry_len]
          )
      ]
  )

  ##############################################################################
  # Build update_lookup. It updates an embedding table, then lookup.
  ##############################################################################
  eqns = []
  for eqn in lookup_update_jaxpr.eqns:
    if utils.is_embedding_lookup(eqn):
      _, embed_tables = utils.lookup_params(eqn)
      # Combine the lookup equation with its corresponding update equation.
      eqns.append(_combine_shard_maps(eqn, updates[embed_tables[0]]))
    elif not utils.is_embedding_update(eqn):
      # Keep non-lookup and non-update equations as they are.
      eqns.append(eqn)
  update_lookup_jaxpr = lookup_update_jaxpr.replace(eqns=eqns)

  ##############################################################################
  # Build update. It only updates an embedding table.
  ##############################################################################
  inps = [True] * (update_len + carry_len) + [False] * (invar_len - carry_len)
  oups = [False] * act_len + [True] * carry_len
  update_jaxpr, used_inputs = pe.dce_jaxpr(
      lookup_update_jaxpr, used_outputs=oups, instantiate=inps
  )
  assert (
      used_inputs == inps
  ), 'There should not be embedding table manipulation after update.'
  assert all(
      not utils.is_embedding_lookup(eqn) for eqn in update_jaxpr.eqns
  ), 'In the train step function, embedding update should before lookup.'

  ##############################################################################
  # Assemble all the results.
  ##############################################################################
  return FunctionRunner(
      result_structure=result_structure,
      carry_structure=carry_structure,
      literals=closed_jaxpr.literals,
      lookup_jaxpr=lookup_jaxpr,
      update_jaxpr=update_jaxpr,
      update_lookup_jaxpr=update_lookup_jaxpr,
      dense_jaxpr=dense_jaxpr,
      update_params_len=update_len,
      activations_len=act_len,
  )
