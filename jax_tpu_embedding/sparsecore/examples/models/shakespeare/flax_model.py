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

"""Shakespeare model using embedding layer."""

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.flax import embed
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec


shard_map = jax.experimental.shard_map.shard_map
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
  table_name: str = 'shakespeare_table'
  feature_name: str = 'shakespeare_feature'

  @nn.compact
  def __call__(self, embedding_lookups: embed.EmbeddingLookups):
    x = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
    )(embedding_lookups)

    # Unpack the activations.
    x = x[self.feature_name]
    x = jnp.reshape(x, (self.global_batch_size, -1))

    # Apply the model.
    x = nn.Dense(self.embedding_size)(x)
    x = nn.Dense(self.vocab_size)(x)

    return x
