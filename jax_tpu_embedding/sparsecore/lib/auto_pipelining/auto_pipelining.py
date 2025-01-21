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
"""Autopipelining decorator.

This decorator is used to automatically transform a train step function to use
SparseCore pipelining.

Example:

```python
@auto_pipelining.auto_pipelining
def train_step(carry: Carry, ...) -> tuple[Carry, ...]:
  ...
  return new_carry, ...
```

With the decorator, the `train_step` function will be transformed to take
pipelining state as the first argument, before the carry. It also returns a new
pipelining state as the first return value.

```python
def train_step(state: EmbeddingPipeliningState, carry: Carry, ...) ->
    tuple[EmbeddingPipeliningState, Carry, ...]:
  ...
  return new_pipelining_state, new_carry, ...
```

**Notice** that we expect that the train step function returns multiple values,
with carry as the first value. We don't support the train_step that only returns
the carry. This is because the carry may be defined as a NamedTuple. We can not
diferentiate a single carry result, vs multiple results.


# Carry

The training function should take carry as the first argument, and return a new
carry as the first return value.

The carry should include all the training states. Specifically, embedding tables
should be part of carry.

Usually, the carry should be donated to the train step function in each step.
And the function should return a new updated carry with exactly the same
structure. There's usually only one carry instance at one time.

On the other hand, args should be the "data" fed into each train step. It
should be ok for the pipelining system to keep them for up to three steps.

A typical training loop looks like this:

```
carry = init_carry()
for (data1, data2) in dataset:
  carry, loss, metrics = train_step(carry, data1, data2)
```

With pipelining, the training loop should be modified to pass pipelining state:

```
carry = init_carry()
state = auto_pipelining.init_state()
for (data1, data2) in dataset:
  state, (carry, loss, metrics) = train_step(state, carry, data1, data2)
```

This carry assumption may be avoidable, but it greatly simplifies the
implementation and use of auto-pipelining system:

1. Training data and training state are naturally separated. We know what part
of the input should be reused across steps, and what part is step specific.

2. For all training state arrays, we can easily map them from inputs to the
outputs. This helps create a "no-op" training step. This is useful for the first
and last few steps, where we only run part of the SparseCore forward, dense,
or the SparseCore backward.


# Limitations on embedding lookups and updates

1. There should be only one embedding lookup and embedding update for each
embedding table.

2. The embedding lookups and updates should not be wrapped in any ops, e.g.
`jax.lax.cond`, or nested `jax.jit`.

3. Each embedding lookup and update shard_map should not contain other
operations.

4. Embedding table should be the last argument to lookup and update shard_map.

The easiest way to comply to these limitations is to use our Flax API.
"""

import dataclasses
import functools
import logging
import os
from typing import Any, Callable, Concatenate, NamedTuple, Optional, ParamSpec, TypeVar

import jax
import jax.extend as jex
from jax_tpu_embedding.sparsecore.lib.auto_pipelining import decompose


Carry = decompose.Carry


def dump_jaxpr(jaxpr: jex.core.ClosedJaxpr | jex.core.Jaxpr, name: str) -> None:
  """Dumps a Jaxpr to a file, if the TEST_UNDECLARED_OUTPUTS_DIR is set."""
  dirname = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
  if not dirname:
    return
  dump_path = os.path.join(dirname, f'jaxpr_{name}.txt')

  logging.info('Writing %s Jaxpr to %s', name, dump_path)

  with open(dump_path, 'w') as f:
    f.write(jaxpr.pretty_print(source_info=True))


class EmbeddingPipeliningState(NamedTuple):
  activations_1: Optional[list[jax.Array]] = None
  args_1: Any = None
  updates_0: Optional[list[jax.Array]] = None


@dataclasses.dataclass
class PipeliningFunction:
  train_step_func: Callable[..., Any]
  finalize: Callable[..., Any]

  def __call__(self, state: EmbeddingPipeliningState, *args_2):
    return self.train_step_func(state, *args_2)


def init_state() -> EmbeddingPipeliningState:
  """Initializes the EmbeddingPipeliningState."""
  return EmbeddingPipeliningState()


def _run(
    runner: decompose.FunctionRunner,
    state: EmbeddingPipeliningState,
    carry: Carry,
    args_2,
):
  """Runs one step of the pipelined training loop."""
  if args_2 is not None:
    if state.updates_0 is not None:
      # Run combined update and lookup (SparseCore backward then forward).
      activations_2, carry = runner.embedding_update_lookup(
          state.updates_0, carry, *args_2
      )
    else:
      # Run lookup only (SparseCore forward).
      activations_2 = runner.embedding_lookup(carry, *args_2)
  else:
    activations_2 = None
    if state.updates_0 is not None:
      # Run update only (SparseCore backward).
      carry = runner.embedding_update(state.updates_0, carry)

  if state.args_1 is not None and state.activations_1 is not None:
    # Run dense layer.
    updates_1, carry, *func_results = runner.dense(
        state.activations_1, carry, *state.args_1
    )
  else:
    # No dense computation needed, create empty results.
    updates_1 = None
    _, *func_results = runner.none_result()

  # Prepare data for the next step.
  next_state = EmbeddingPipeliningState(
      activations_1=activations_2,
      # During finalization, args_2 is None. Keep the last args_1 for tracing
      # This can be removed if we move to one program setting.
      args_1=args_2 or state.args_1,
      updates_0=updates_1,
  )

  return next_state, (carry, *func_results)


def finalize_pipelining(
    fn: Callable[..., Any], state: EmbeddingPipeliningState, carry: Carry
):
  """Finalizes the pipelining run."""
  assert isinstance(
      fn, PipeliningFunction
  ), 'The function is not auto-pipelining wrapped.'
  state, (carry, *res) = fn.finalize(state, carry)
  _, (carry, *_) = fn.finalize(state, carry)
  return carry, *res


FunctionArgs = ParamSpec('FunctionArgs')
FunctionResults = TypeVar('FunctionResults')


def auto_pipelining(
    fn: Callable[FunctionArgs, FunctionResults],
    static_argnums: tuple[int, ...] = (),
) -> Callable[
    Concatenate[EmbeddingPipeliningState, FunctionArgs],
    tuple[EmbeddingPipeliningState, FunctionResults],
]:
  """A decorator to transform a function for SparseCore pipelining."""

  def _build_runner(carry: Carry, args_2):
    """Compile the function if it's not compiled yet."""
    closed_jaxpr, res_shape = jax.make_jaxpr(
        fn, static_argnums=static_argnums, return_shape=True
    )(carry, *args_2)
    dump_jaxpr(closed_jaxpr, 'original')
    assert isinstance(res_shape, tuple), 'train_step should return a tuple.'
    assert jax.tree.structure(carry) == jax.tree.structure(res_shape[0]), (
        'The structure of the first argument (carry) should be the same as'
        ' the first return value.'
    )

    res_structure = jax.tree.structure(res_shape)
    carry_structure = jax.tree.structure(carry)
    runner = decompose.decompose(closed_jaxpr, res_structure, carry_structure)

    dump_jaxpr(runner.lookup_jaxpr, 'lookup')
    dump_jaxpr(runner.dense_jaxpr, 'dense')
    dump_jaxpr(runner.update_jaxpr, 'update')
    dump_jaxpr(runner.update_lookup_jaxpr, 'update_lookup')

    return runner

  @functools.wraps(fn)
  def train_pipeline(state: EmbeddingPipeliningState, carry: Carry, *args_2):
    runner = _build_runner(carry, args_2)
    return _run(runner, state, carry, args_2)

  def finalize_pipeline(state: EmbeddingPipeliningState, carry: Carry):
    assert state.args_1, (
        'Args_1 should not be None. This is likely caused by a bug in the'
        ' auto-pipelining system.'
    )

    runner = _build_runner(carry, state.args_1)
    return _run(runner, state, carry, None)

  return PipeliningFunction(jax.jit(train_pipeline), jax.jit(finalize_pipeline))
