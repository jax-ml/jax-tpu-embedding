# Copyright 2024 JAX SC Authors.
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

"""Utils for checkpointing and restoring embedding variables."""

from typing import Any, Dict

import jax
from jax_tpu_embedding.sparsecore.lib.nn import embedding


def convert_orbax_restored_dict_to_embedding_variables(
    init_embedding_variables: Dict[str, Any],
    restored_dict: Dict[str, Any],
) -> Dict[str, Any]:
  """Convert the restored dict to EmbeddingVariables.

  Args:
    init_embedding_variables: the namedtuple from the initialized embedding
      variables
    restored_dict: the dict from the restored checkpoint

  Returns:
    the dictionary with EmbeddingVariables
  """

  def convert_dict_to_embedding_variables(
      reference: embedding.EmbeddingVariables, input_dict: Dict[str, Any]
  ) -> embedding.EmbeddingVariables:
    """Convert the restored dict to EmbeddingVariables.

    Args:
      reference: the namedtuple from the initialized embedding variables
      input_dict: the dict from the restored checkpoint

    Returns:
      the EmbeddingVariables namedtuple

    Raises:
      ValueError: if the input dict is not in the expected format
    """
    if (
        not isinstance(input_dict, dict)
        or "table" not in input_dict
        or "slot" not in input_dict
    ):
      raise ValueError(
          "Unexpected type for conversion to EmbeddingVariables:"
          f" input={input_dict}"
      )

    # create the slot namedtuple
    if input_dict["slot"] is None:
      slot = (type(reference.slot))()
    elif isinstance(input_dict["slot"], dict):
      slot = (type(reference.slot))(**input_dict["slot"])
    else:
      raise ValueError(
          "Unexpected slot type for conversion to EmbeddingVariables:"
          f" input={input_dict}"
      )

    return embedding.EmbeddingVariables(
        table=input_dict["table"],
        slot=slot,
    )

  return jax.tree.map(
      convert_dict_to_embedding_variables,
      init_embedding_variables,
      restored_dict,
      is_leaf=lambda x: isinstance(x, embedding.EmbeddingVariables),
  )
