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
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax


np.set_printoptions(threshold=np.inf)
Nested = embedding.Nested


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


_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size.')

_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'Global batch size.'
)

_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')

_SEQ_LEN = flags.DEFINE_integer(
    'sequence_length', 16, 'Sequence length of context words.'
)

_NUM_TABLES = flags.DEFINE_integer(
    'num_tables', 1, 'Number of tables to create.'
)

_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 1000, 'Number of steps to train for.'
)

_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs',
    1,
    'Number of separate epochs.',
)

_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 8, 'Embedding size.')
_EMBEDDING_INIT = flags.DEFINE_enum(
    'embedding_init', 'normal', ['normal', 'row_id'], 'Embedding initializer.'
)

_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency', 10, 'Frequency to log metrics.'
)
_LOSS_RESET_FREQUENCY = flags.DEFINE_integer(
    'loss_window', 10, 'Number of steps to average loss over.'
)

_FLAGS = flags.FLAGS

info = logging.info
vlog1 = partial(logging.vlog, 1)
vlog3 = partial(logging.vlog, 3)


def create_train_state(
    rng: jax.Array,
    global_device_count: int,
    num_sc_per_device: int,
    global_batch_size: int,
    vocab_size: int,
    seq_len: int,
    embedding_size: int,
) -> tuple[
    nn.Module,
    optax.GradientTransformation,
    TrainState,
    embedding.Nested[embedding_spec.FeatureSpec],
]:
  """Create and initialize the model.

  Args:
    rng: JAX PRNG Key.
    global_device_count: The JAX global device count.
    num_sc_per_device: The number of sparsecores per device.
    global_batch_size: global batch size.
    vocab_size: embedding vocabulary size.
    seq_len: sequence length.
    embedding_size: embedding dimension.

  Returns:
    The model, optimizer,  initial train state, table specs, and feature specs.
  """
  model = shakespeare_model.Model(
      global_batch_size=global_batch_size,
      vocab_size=vocab_size,
      seq_len=seq_len,
      embedding_size=embedding_size,
  )

  # Global embedding activations. We can change this to local arrays and use
  # make_array_from_single_device_arrays as above.
  init_emb_activations = {
      model.feature_name: jnp.zeros((
          global_batch_size,
          seq_len,
          1,
          embedding_size,
      ))
  }

  params = model.init(rng, init_emb_activations)
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.adam(learning_rate=_LEARNING_RATE.value)
  feature_specs = model.create_feature_specs()
  embedding.prepare_feature_specs_for_training(
      feature_specs,
      global_device_count=global_device_count,
      num_sc_per_device=num_sc_per_device,
  )
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
  local_devices = jax.local_devices()
  global_devices = jax.devices()
  num_global_devices = len(global_devices)
  num_local_devices = len(local_devices)
  num_sc_per_device = utils.num_sparsecores_per_device(global_devices[0])

  num_processes = jax.process_count()
  process_id = jax.process_index()
  info(
      'num devices: local = %s, global = %s',
      num_local_devices,
      num_global_devices,
  )
  info('process_id = %s, num_processes = %s', process_id, num_processes)
  info('local_devices [len=%s] = %s', len(local_devices), local_devices)
  info('global_devices [len=%s] = %s', len(global_devices), global_devices)

  # Shard the local embedding table onto the local devices.
  info('Local Devices: %s', local_devices)
  local_mesh = Mesh(np.array(local_devices), axis_names=['device'])
  info('Local Mesh: %s', local_mesh)
  global_mesh = Mesh(np.array(global_devices), axis_names=['device'])
  global_emb_sharding = NamedSharding(global_mesh, P('device', None, None))

  # Note 1: InputProcessing is currently global so all the input batch features
  # and labels are global as well. We'll have to use local slices in
  # multi-process environments.
  #
  # Note 2: The input processor expects 2-d lookups, so we increase the batch
  # size and reshape the results.
  local_batch_size = _GLOBAL_BATCH_SIZE.value // num_processes
  device_batch_size = _GLOBAL_BATCH_SIZE.value // num_global_devices
  info(
      'batch sizes: global=%s, local=%s, device=%s',
      _GLOBAL_BATCH_SIZE.value,
      local_batch_size,
      device_batch_size,
  )

  per_sc_vocab_size = _VOCAB_SIZE.value // num_sc_per_device
  if per_sc_vocab_size < 8 or per_sc_vocab_size % 8 != 0:
    raise ValueError(
        'Vocabulary size must be a multiple of 8 per SC: VOCAB_SIZE ='
        f' {_VOCAB_SIZE.value}, num_scs = {num_sc_per_device}'
    )

  word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
  vlog1('word_ids len is %s', len(word_ids))
  feature_batches, label_batches = shakespeare_data.word_id_batches(
      word_ids,
      _NUM_STEPS.value,
      _GLOBAL_BATCH_SIZE.value,
      _SEQ_LEN.value,
      _NUM_TABLES.value,
  )
  feature_batches = feature_batches['words_0']
  vlog1('feature_batches len = %s', len(feature_batches))
  info('feature_batches[0] shape = %s', feature_batches[0].shape)
  vlog1('label_batches len = %s', len(label_batches))
  vlog1('label_batches[0] shape = %s', label_batches[0].shape)

  # Initialize the model.
  model, optimizer, train_state, feature_specs = create_train_state(
      jax.random.key(42),
      num_global_devices,
      num_sc_per_device,
      _GLOBAL_BATCH_SIZE.value,
      _VOCAB_SIZE.value,
      _SEQ_LEN.value,
      _EMBEDDING_SIZE.value,
  )
  vlog3('train_state is %s', train_state)

  table_specs = {
      f.table_spec.name: f.table_spec for f in jax.tree.leaves(feature_specs)
  }
  emb_variables = embedding.init_embedding_variables(
      jax.random.key(13), table_specs, global_emb_sharding, num_sc_per_device
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
        lambda x: jnp.reshape(x, (device_batch_size, -1)), emb_act
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
  vlog1('Replicating train_state')
  train_state = flax_utils.replicate(train_state, local_devices)
  parameter_overview.log_parameter_overview(train_state.params)
  p_train_step_fn = jax.pmap(
      partial(
          train_step_fn,
          num_global_devices,
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

    vlog1('*' * 70)
    vlog1('* STEP = %s', step)
    vlog1('*' * 70)

    # ----------------------------------------------------------------------
    # SC input processing.
    # ----------------------------------------------------------------------
    # These are currently global batches so each task needs to offset into
    # the data for it's local slice.
    features = features[
        process_id * local_batch_size : (process_id + 1) * local_batch_size
    ]
    features = np.reshape(features, (-1, 1))

    # Pack the features into a tree structure.
    feature_structure = jax.tree.structure(feature_specs)
    features = jax.tree_util.tree_unflatten(feature_structure, [features])

    # Preprocess the inputs.
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        features,
        None,  # uniform weights
        feature_specs,
        local_device_count=global_mesh.local_mesh.size,
        global_device_count=global_mesh.size,
        num_sc_per_device=num_sc_per_device,
        sharding_strategy='MOD',
        has_leading_dimension=True,
        batch_number=step,
    )

    # TODO(patn): This (local_slice)will go away once the input processor is
    # updated to only produce local batches.
    local_slice = lambda x: jax.lax.slice_in_dim(
        x,
        process_id * (x.shape[0] // num_processes),
        (process_id + 1) * (x.shape[0] // num_processes),
    )

    # Labels are global so we need to offset into the global batch for this
    # local slice.
    labels_sharded = jnp.reshape(
        labels,
        (
            num_global_devices,
            device_batch_size,
        ),
    )
    labels_sharded = local_slice(labels_sharded)

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
          labels_sharded,
      )
      jaxpr_fn = os.path.join(
          dirname, f'jaxpr-pmap-train_step_fn-process_{process_id}.txt'
      )
      info('Writing train_step_fn Jaxpr to %s', jaxpr_fn)
      with open(jaxpr_fn, 'w') as f:
        f.write(jaxpr.pretty_print())

    train_state, metrics_update, emb_variables = p_train_step_fn(
        train_state,
        preprocessed_inputs,
        emb_variables,
        labels_sharded,
    )
    metrics_update = flax_utils.unreplicate(metrics_update)
    train_metrics = (
        metrics_update
        if train_metrics is None
        else train_metrics.merge(metrics_update)
    )

    if (step + 1) % _LOG_FREQUENCY.value == 0:
      m = train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(train_state.params)

    if (step + 1) % _LOSS_RESET_FREQUENCY.value == 0:
      train_metrics = None


def main(argv: collections.abc.Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_model()


if __name__ == '__main__':
  app.run(main)
