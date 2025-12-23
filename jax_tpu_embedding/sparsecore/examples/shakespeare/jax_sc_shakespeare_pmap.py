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
"""An example Shakespeare model that uses the SparseCore embedding API."""

# pylint: disable=g-importing-member
import collections
from functools import partial
import os
from typing import Any, Mapping

from absl import app
from absl import flags
from absl import logging
from clu import metrics
from clu import parameter_overview
import flax
from flax import linen as nn
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
import numpy as np
import optax


np.set_printoptions(threshold=np.inf)
Nested = embedding.Nested
_FLAGS = flags.FLAGS
info = logging.info


@flax.struct.dataclass
class TrainState:
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """
  params: Any
  opt_state: optax.OptState


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  # train_accuracy: metrics.Accuracy
  # learning_rate: metrics.LastValue.from_output("learning_rate")
  train_loss: metrics.Average.from_output('loss')
  train_loss_std: metrics.Std.from_output('loss')


def create_train_state(
    config: shakespeare_config.Config,
    rng: jax.Array,
) -> tuple[
    nn.Module,
    optax.GradientTransformation,
    TrainState,
    embedding.Nested[embedding_spec.FeatureSpec],
]:
  """Create and initialize the model.

  Args:
    config: The model configuration.
    rng: JAX PRNG Key.

  Returns:
    The model, optimizer,  initial train state, table specs, and feature specs.
  """
  model = shakespeare_model.Model(
      global_batch_size=config.global_batch_size,
      vocab_size=config.vocab_size,
      seq_len=config.seq_len,
      embedding_size=config.embedding_size,
      feature_name=config.feature_name,
  )

  # Global embedding activations. We can change this to local arrays and use
  # make_array_from_single_device_arrays as above.
  init_emb_activations = {
      config.feature_name: jnp.zeros((
          config.global_batch_size,
          config.seq_len,
          1,
          config.embedding_size,
      ))
  }

  params = model.init(rng, init_emb_activations)
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.adam(learning_rate=config.learning_rate)
  feature_specs = shakespeare_config.create_feature_specs(config)
  return (
      model,
      optimizer,
      TrainState(
          params=params,
          opt_state=optimizer.init(params),
      ),
      feature_specs,
  )


def run_model():
  """Runs the model including input processing and training."""

  flags.FLAGS.vocab_size = 256
  flags.FLAGS.batch_size = 32
  flags.FLAGS.num_steps = 1000
  config = shakespeare_config.get_config()

  # Shard the local embedding table onto the local devices.
  global_mesh = Mesh(
      np.array(config.global_devices), axis_names=[config.sharding_axis]
  )
  data_sharding = NamedSharding(global_mesh, P(config.sharding_axis, None))
  embedding_sharding = NamedSharding(
      global_mesh, P(config.sharding_axis, None, None)
  )

  # Note 1: InputProcessing is currently global so all the input batch features
  # and labels are global as well. We'll have to use local slices in
  # multi-process environments.
  #
  # Note 2: The input processor expects 2-d lookups, so we increase the batch
  # size and reshape the results.
  feature_batches, label_batches = shakespeare_config.get_batches(config)

  # Initialize the model.
  model, optimizer, train_state, feature_specs = create_train_state(
      config, jax.random.key(42)
  )

  table_specs = {
      f.table_spec.name: f.table_spec for f in jax.tree.leaves(feature_specs)
  }
  emb_variables = embedding.init_embedding_variables(
      jax.random.key(13),
      table_specs,
      embedding_sharding,
      config.num_sc_per_device,
  )

  def train_step_fn(
      global_device_count: int,
      model: nn.Module,
      optimizer,
      feature_specs,
      train_state: TrainState,
      preprocessed_inputs,
      emb_variables: Mapping[str, embedding.EmbeddingVariables],
      labels,
  ) -> tuple[
      TrainState, metrics.Collection, Mapping[str, embedding.EmbeddingVariables]
  ]:
    """Performs a single training step at the device level."""
    # Sparse forward pass - embedding lookup.
    with jax.named_scope('sc_forward_pass'):
      tpu_sparse_dense_matmul = partial(
          embedding.tpu_sparse_dense_matmul,
          global_device_count=global_device_count,
          feature_specs=feature_specs,
          sharding_strategy='MOD',
      )
      emb_act = tpu_sparse_dense_matmul(
          preprocessed_inputs,
          emb_variables,
      )

    # Dense forward + backward pass.
    emb_act = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (config.device_batch_size, -1)), emb_act
    )
    with jax.named_scope('tc_pass'):
      loss_grad_fn = jax.value_and_grad(
          partial(shakespeare_model.loss, model), argnums=(0, 1), has_aux=True
      )

      (loss, logits), grads = loss_grad_fn(train_state.params, emb_act, labels)

    with jax.named_scope('tc_update'):
      (dense_grad, emb_grad) = jax.lax.pmean(grads, axis_name='batch')

      updates, new_opt_state = optimizer.update(
          dense_grad, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)

    emb_grad = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, model.embedding_size)), emb_grad
    )

    # Sparse backward pass - embedding update.
    with jax.named_scope('sc_backward_pass'):
      tpu_sparse_dense_matmul_grad = partial(
          embedding.tpu_sparse_dense_matmul_grad,
          feature_specs=feature_specs,
          sharding_strategy='MOD',
      )
      emb_variables = tpu_sparse_dense_matmul_grad(
          emb_grad,
          preprocessed_inputs,
          emb_variables,
      )

    new_train_state = train_state.replace(
        params=new_params, opt_state=new_opt_state
    )

    metrics_update = TrainMetrics.gather_from_model_output(
        loss=loss,
        logits=logits,
        labels=labels,
    )

    return new_train_state, metrics_update, emb_variables

  # Distributed training.
  info('Replicating train_state')
  train_state = flax_utils.replicate(train_state, config.local_devices)
  parameter_overview.log_parameter_overview(train_state.params)
  p_train_step_fn = jax.pmap(
      partial(
          train_step_fn,
          config.num_global_devices,
          model,
          optimizer,
          feature_specs,
      ),
      axis_name='batch',
  )

  train_metrics = None
  step = -1
  for features, labels in zip(feature_batches, label_batches):
    step += 1
    shakespeare_config.step_header(step)

    # ----------------------------------------------------------------------
    # SC input processing.
    # ----------------------------------------------------------------------
    # These are currently global batches so each task needs to offset into
    # the data for it's local slice.
    features = shakespeare_config.local_slice(config, features)
    labels = jnp.reshape(
        shakespeare_config.local_slice(config, labels),
        (config.num_local_devices, -1),
    )

    # Preprocess the inputs.
    preprocessed_inputs, _ = shakespeare_config.process_inputs(
        config,
        feature_specs,
        step,
        features,
        data_sharding,
        has_leading_dimension=True,
    )

    # ----------------------------------------------------------------------
    # Optionally dump the Jaxpr for the train step.
    if _FLAGS.dump_dir and step == 0:
      dirname = (
          os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
          if _FLAGS.dump_dir == 'sponge'
          else _FLAGS.dump_dir
      )
      if not dirname:
        logging.warning(
            'Cannot write Jaxpr to sponge, TEST_UNDECLARED_OUTPUTS_DIR not set'
        )
        continue
      jaxpr = jax.make_jaxpr(p_train_step_fn)(
          train_state,
          preprocessed_inputs,
          emb_variables,
          labels,
      )
      jaxpr_fn = os.path.join(
          dirname, f'jaxpr-pmap-train_step_fn-process_{config.process_id}.txt'
      )
      info('Writing train_step_fn Jaxpr to %s', jaxpr_fn)
      with open(jaxpr_fn, 'w') as f:
        f.write(jaxpr.pretty_print())

    train_state, metrics_update, emb_variables = p_train_step_fn(
        train_state,
        preprocessed_inputs,
        emb_variables,
        labels,
    )
    metrics_update = flax_utils.unreplicate(metrics_update)
    train_metrics = (
        metrics_update
        if train_metrics is None
        else train_metrics.merge(metrics_update)
    )

    if (step + 1) % config.log_frequency == 0:
      m = train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(train_state.params)

    if (step + 1) % config.loss_reset_frequency == 0:
      train_metrics = None


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_model()


if __name__ == '__main__':
  app.run(main)
