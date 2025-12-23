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
import functools
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import flax_model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.flax.linen import embed_optimizer
from jax_tpu_embedding.sparsecore.lib.nn import embedding
import numpy as np
import optax


Nested = embedding.Nested
FLAGS = flags.FLAGS
np.set_printoptions(threshold=np.inf)


################################################################################
# Define the shakespeare test.
################################################################################
class ShakespeareTest(absltest.TestCase):
  """Input processing and convergence of basic model using JAX SC libraries."""

  def test_shakespeare_model_loss_convergence(self):
    FLAGS.vocab_size = 4096
    FLAGS.batch_size = 8
    FLAGS.embedding_size = 16
    config = shakespeare_config.get_config()

    mesh = jax.sharding.Mesh(config.global_devices, config.sharding_axis)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(config.sharding_axis)
    )

    feature_batches, label_batches = shakespeare_config.get_batches(config)
    feature_specs = shakespeare_config.create_feature_specs(config)

    # Construct the model.
    model = shakespeare_model.Model(
        feature_specs=feature_specs,
        global_batch_size=config.global_batch_size,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        embedding_size=config.embedding_size,
        mesh=mesh,
        feature_name=config.feature_name,
        sharding_axis=config.sharding_axis,
    )

    first_model_input, _ = shakespeare_config.process_inputs(
        config, feature_specs, -1, feature_batches[0], data_sharding
    )
    params = model.init(jax.random.key(42), first_model_input)

    # Create optimizer.
    tx = embed_optimizer.create_optimizer_for_sc_model(
        params,
        optax.sgd(learning_rate=config.learning_rate),
    )
    opt_state = tx.init(params)

    # Define the forward pass function.
    @functools.partial(
        jax.jit,
        out_shardings=(
            nn.meta.get_sharding(params, mesh),
            None,
            None,
        ),
        donate_argnums=(0),
    )
    def train_step(
        params: Any,
        embedding_lookup_inputs: embedding.PreprocessedInput,
        labels: jax.Array,
        opt_state,
    ):
      def forward_pass(params, embedding_lookups, labels):
        logits = model.apply(params, embedding_lookups)
        xentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        return jnp.mean(xentropy), logits

      # Run model forward/backward pass.
      train_step_fn = jax.value_and_grad(
          forward_pass, has_aux=True, allow_int=True
      )

      (loss_val, unused_logits), grads = train_step_fn(
          params, embedding_lookup_inputs, labels
      )

      updates, opt_state = tx.update(grads, opt_state)
      params = embed_optimizer.apply_updates_for_sc_model(params, updates)

      return params, opt_state, loss_val

    step = 0
    for features, labels in zip(feature_batches, label_batches):
      features = shakespeare_config.local_slice(config, features)
      labels = shakespeare_config.device_slice(config, labels, data_sharding)

      processed_input_tensor, _ = shakespeare_config.process_inputs(
          config, feature_specs, step, features, data_sharding
      )

      params, opt_state, loss_val = train_step(
          params, processed_input_tensor, labels, opt_state
      )

      logging.info(
          'Step %s: loss=%s, params is %s',
          step,
          loss_val,
          jax.tree.map(jnp.sum, params),
      )

      step += 1


if __name__ == '__main__':
  absltest.main()
