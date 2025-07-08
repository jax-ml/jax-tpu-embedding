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
"""Shakespeare next word predictor model."""

from typing import Any, Mapping

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import optax


class Model(nn.Module):
  """A simple model that predicts the next word in a sequence of words.

  Attributes:
    global_batch_size: The number of examples in the global batch.
    vocab_size: The number of unique words in the vocabulary.
    seq_len: The length of the sequences in the global batch.
    embedding_size: The dimension of the embedding vectors.
    table_name: The name of the embedding table.
    feature_name: The name of the embedding feature.
  """

  global_batch_size: int
  vocab_size: int
  seq_len: int
  embedding_size: int
  table_name: str = 'shakespeare_table'
  feature_name: str = 'shakespeare_feature'

  def create_feature_specs(
      self,
  ) -> Mapping[str, embedding_spec.FeatureSpec]:
    """Creates the feature specs for the Shakespeare model.

    Returns:
      The feature specs for the Shakespeare model.
    """
    table_spec = embedding_spec.TableSpec(
        vocabulary_size=self.vocab_size,
        embedding_dim=self.embedding_size,
        initializer=jax.nn.initializers.zeros,
        optimizer=embedding_spec.SGDOptimizerSpec(),
        combiner='sum',
        name=self.table_name,
        max_ids_per_partition=5000,
        max_unique_ids_per_partition=5000,
    )
    feature_spec = embedding_spec.FeatureSpec(
        table_spec=table_spec,
        input_shape=(self.global_batch_size * self.seq_len, 1),
        output_shape=(
            self.global_batch_size * self.seq_len,
            self.embedding_size,
        ),
        name=self.feature_name,
    )
    feature_specs = nn.FrozenDict({self.feature_name: feature_spec})
    return feature_specs

  @nn.compact
  def __call__(self, emb_activations: Mapping[str, jax.Array]):
    # Unpack the activations.
    x = emb_activations[self.feature_name]
    x = jnp.reshape(x, (x.shape[0], -1))
    # Apply the model.
    x = nn.Dense(self.embedding_size)(x)
    x = nn.Dense(self.vocab_size)(x)
    return x


def loss(
    model: nn.Module,
    params: Any,
    emb_activations: Mapping[str, jax.Array],
    labels: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Applies the embedding activations to model and returns loss.

  Args:
    model: The model being trained.
    params: The parameters of the model.
    emb_activations: The embedding activations that will be applied.
    labels: The integer labels corresponding to the embedding activations.

  Returns:
    The loss.
  """
  logits = model.apply(params, emb_activations)
  xentropy = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=labels
  )
  return jnp.mean(xentropy), logits
