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

from flax import nnx
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax.nnx import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

Nested = embedding.Nested


################################################################################
# Define the model.
################################################################################
class Model(nnx.Module):
  """Shakespeare model using embedding layer."""

  def __init__(
      self,
      *,
      feature_specs: Nested[embedding_spec.FeatureSpec],
      global_batch_size: int,
      vocab_size: int,
      seq_len: int,
      embedding_size: int,
      enable_minibatching: bool,
      mesh: jax.sharding.Mesh,
      sharding_axis: str,
  ):
    self.feature_name = 'shakespeare_feature'
    assert len(feature_specs) == 1, 'Shakespeare model expects one feature.'
    assert self.feature_name in feature_specs, (
        'Shakespeare model expects feature named "%s".' % self.feature_name
    )

    self.feature_specs = feature_specs
    self.global_batch_size = global_batch_size
    self.vocab_size = vocab_size
    self.seq_len = seq_len
    self.embedding_size = embedding_size
    self.enable_minibatching = enable_minibatching
    self.mesh = mesh
    self.sharding_axis = sharding_axis
    rngs = nnx.Rngs(params=42)
    self.embedding_layer = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
        rngs=rngs,
        enable_minibatching=enable_minibatching,
    )
    e = self.embedding_size
    v = self.vocab_size
    s = self.seq_len
    self.dense_layer_1 = nnx.Linear(
        in_features=s * e,
        out_features=e,
        rngs=rngs,
    )
    self.dense_layer_2 = nnx.Linear(
        in_features=e,
        out_features=v,
        rngs=rngs,
    )

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

  def __call__(self, embedding_lookup_inputs: embedding.PreprocessedInput):
    # Run the embedding layer.
    x = self.embedding_layer(embedding_lookup_inputs)

    # Unpack the activations.
    x = x[self.feature_name]
    x = jnp.reshape(x, (self.global_batch_size, -1))
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    # Apply the dense portion of the model.
    x = self.dense_layer_1(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))
    x = self.dense_layer_2(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    return x
