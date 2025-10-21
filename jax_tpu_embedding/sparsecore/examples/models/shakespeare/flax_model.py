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
"""Shakespeare model using embedding layer."""

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

Nested = embedding.Nested


################################################################################
# Define the model.
################################################################################
class Model(nn.Module):
  """Shakespeare model using embedding layer."""

  feature_specs: Nested[embedding_spec.FeatureSpec]
  global_batch_size: int
  vocab_size: int
  seq_len: int
  embedding_size: int
  feature_name: str = 'shakespeare_feature'
  mesh: jax.sharding.Mesh | None = None
  sharding_axis: str = 'sparsecore_sharding'

  def add_sharding_constraint(self, x: jax.Array, names: tuple[str | None]):
    # Add a sharding constraint to the array.
    #
    # Add a sharding constraint to the array to ensure that the sharding
    # information is not lost during compilation. This may not be necessary but
    # it helps SPMD and ensures that the sharding information is as expected.
    #
    # Args:
    #   x: The array to add the sharding constraint to.
    #   names: The mesh axes for the partition spec.
    #
    # Returns:
    #   The array with the sharding constraint added.
    return jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*names)
        ),
    )

  @nn.compact
  def __call__(self, embedding_lookup_inputs: embedding.PreprocessedInput):
    # Run the embedding layer.
    x = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
    )(embedding_lookup_inputs)

    # Unpack the activations.
    x = x[self.feature_name]
    x = jnp.reshape(x, (self.global_batch_size, -1))
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    # Apply the dense portion of the model.
    x = nn.Dense(self.embedding_size)(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))
    x = nn.Dense(self.vocab_size)(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    return x
