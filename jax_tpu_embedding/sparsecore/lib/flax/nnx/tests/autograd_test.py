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
from absl import flags
from absl import logging
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import flax_nnx_model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.flax.nnx import embed
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

    mesh = Mesh(config.global_devices, (config.sharding_axis,))
    data_sharding = NamedSharding(mesh, PartitionSpec(config.sharding_axis))
    feature_batches, label_batches = shakespeare_config.get_batches(config)
    feature_specs = shakespeare_config.create_feature_specs(config)

    # Construct the model.
    model = shakespeare_model.Model(
        feature_specs=feature_specs,
        global_batch_size=config.global_batch_size,
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        embedding_size=config.embedding_size,
        enable_minibatching=False,
        mesh=mesh,
        feature_name=config.feature_name,
        sharding_axis=config.sharding_axis,
    )

    # Create optimizer.
    tx = optax.sgd(learning_rate=config.learning_rate)
    optimizer = embed.PartitionedOptimizer(model, tx)

    model_sharding = embed.get_named_sharding(model, mesh)
    optimizer_sharding = embed.get_named_sharding(optimizer, mesh)

    # Define the forward pass function.
    @nnx.jit(
        in_shardings=(
            model_sharding,
            optimizer_sharding,
            data_sharding,
            data_sharding,
        ),
        donate_argnames=[
            'model',
            'optimizer',
        ],
    )
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        embedding_lookup_inputs: embedding.PreprocessedInput,
        labels: jax.Array,
    ):
      def loss_fn(mdl, inputs, labels):
        logits = mdl(inputs)
        xentropy = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        )
        return jnp.mean(xentropy), logits

      train_step_fn = nnx.value_and_grad(loss_fn, has_aux=True, allow_int=True)

      (loss_val, unused_logits), grads = train_step_fn(
          model, embedding_lookup_inputs, labels
      )

      optimizer.update(model, grads)

      return loss_val

    step = 0
    for features, labels in zip(feature_batches, label_batches):
      features = shakespeare_config.local_slice(config, features)
      labels = shakespeare_config.device_slice(config, labels, data_sharding)

      processed_input_tensor, _ = shakespeare_config.process_inputs(
          config, feature_specs, step, features, data_sharding
      )

      loss_val = train_step(model, optimizer, processed_input_tensor, labels)

      logging.info(
          'Step %s: loss=%s, params is %s',
          step,
          loss_val,
          jax.tree.map(jnp.sum, model.embedding_layer.embedding_table.value),
      )

      step += 1


if __name__ == '__main__':
  absltest.main()
