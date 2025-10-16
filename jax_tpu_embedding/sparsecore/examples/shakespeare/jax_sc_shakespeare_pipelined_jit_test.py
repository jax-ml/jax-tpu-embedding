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
"""An integration test for the Shakespeare model with SparseCore embedding pipelining API.

This test runs the entire training loop, including input processing and
training. It uses the SparseCore embedding pipelining API to improve
performance.

This is not a typical unit test, but it is a valuable tool for testing the
integration of different components of the SparseCore embedding API.
"""

from functools import partial  # pylint: disable=g-importing-member
from typing import Any

from absl import flags
from absl import logging
from absl.testing import absltest
from clu import metrics
from clu import parameter_overview
import flax
from flax import linen as nn
import jax
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import model as shakespeare_model
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_pipelining_utils as ep_utils
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax


jnp.set_printoptions(threshold=np.inf, linewidth=np.inf)
NestedFeatureSpecs = embedding.Nested[embedding_spec.FeatureSpec]
NestedEmbeddingVariables = embedding.Nested[embedding.EmbeddingVariables]
NestedArray = embedding.Nested[jax.Array]
PyTree = Any


@flax.struct.dataclass
class TrainState:
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """

  params: Any
  opt_state: optax.OptState
  step: jax.Array


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  """Training metrics.

  This class defines the training metrics that will be collected during
  training.
  """

  # train_accuracy: metrics.Accuracy
  # learning_rate: metrics.LastValue.from_output("learning_rate")
  train_loss: metrics.Average.from_output('loss')
  train_loss_std: metrics.Std.from_output('loss')


@flax.struct.dataclass
class ShakespeareModelDenseInput:
  """Input data for the Shakespeare model."""

  labels: jax.Array | None = None


ShakesperaeModelPipelineCurrentStepInput = ep_utils.CurrentStepInput[
    ep_utils.DefaultSparseInputs, ShakespeareModelDenseInput
]


@flax.struct.dataclass
class ShakespeareModelOutput:
  """Model output data."""

  metrics_update: TrainMetrics | None = None


# Instance of PipelineState for the Shakespeare model.
ShakespeareModelPipelineState = ep_utils.PipelineState[
    ep_utils.DefaultSparseInputs,
    ShakespeareModelDenseInput,
    ep_utils.DefaultEmbeddingActivations,
    ep_utils.DefaultEmbeddingGradients,
    ep_utils.DefaultTcAux,
    ep_utils.DefaultScFwdAux,
    ShakespeareModelOutput,
]

_VOCAB_SIZE = flags.DEFINE_integer(
    'vocab_size', 512, 'Vocabulary size.', lower_bound=512
)

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

_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 8, 'Embedding size.')

_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency', 10, 'Frequency to log metrics.'
)
_METRICS_RESET_FREQUENCY = flags.DEFINE_integer(
    'metrics_reset_frequency', 10, 'Number of steps to average loss over.'
)

info = logging.info
vlog1 = partial(logging.vlog, 1)


def create_train_state(
    rng: jax.Array,
    global_device_count: int,
    num_sc_per_device: int,
    global_batch_size: int,
    vocab_size: int,
    seq_len: int,
    embedding_size: int,
) -> tuple[
    nn.Module, optax.GradientTransformation, TrainState, NestedFeatureSpecs
]:
  """Creates and initializes the model, optimizer, train state, and feature specs.

  Args:
    rng: JAX PRNG Key.
    global_device_count: The number of global devices (chips). Typically
      `mesh.size`.
    num_sc_per_device: The number of sparsecores per device.
    global_batch_size: global batch size.
    vocab_size: embedding vocabulary size.
    seq_len: sequence length.
    embedding_size: embedding dimension.

  Returns:
    The model, optimizer, initial train state, and feature specs.
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
      model.feature_name: jnp.zeros(
          (global_batch_size, seq_len * embedding_size)
      )
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
          params=params, opt_state=optimizer.init(params), step=jnp.array(0)
      ),
      feature_specs,
  )


class EmbeddingPipelineTest(absltest.TestCase):
  """Embedding pipeline test."""

  def setUp(self):
    super().setUp()

    # Get number of global and local devices
    self.num_global_devices = jax.device_count()
    self.num_local_devices = jax.local_device_count()
    self.num_processes = jax.process_count()
    self.process_id = jax.process_index()
    self.local_devices = jax.local_devices()
    self.global_devices = jax.devices()
    self.num_sc_per_device = utils.num_sparsecores_per_device(
        self.global_devices[0]
    )

    # Log device information
    info(
        'local_devices [len=%s] = %s',
        len(self.local_devices),
        self.local_devices,
    )
    info(
        'global_devices [len=%s] = %s',
        len(self.global_devices),
        self.global_devices,
    )
    info(
        'num devices: local = %s, global = %s',
        self.num_local_devices,
        self.num_global_devices,
    )
    info(
        'process_id = %s, num_processes = %s',
        self.process_id,
        self.num_processes,
    )

    # Define sharding specs
    self.pd = P('device')  # Device sharding.
    self.pe = P('device', None)  # PartitionSpec for embedding tables.

    # Create global mesh and sharding
    self.global_mesh = Mesh(
        np.array(self.global_devices), axis_names=['device']
    )
    self.global_sharding = NamedSharding(self.global_mesh, self.pd)
    self.global_emb_sharding = NamedSharding(self.global_mesh, self.pe)
    self.global_emb_layout = jax_layout.Format(
        jax_layout.Layout(major_to_minor=(0, 1), tiling=((8,),)),
        self.global_emb_sharding,
    )
    self.replicated_sharding = NamedSharding(self.global_mesh, P())

    self.train_metrics = TrainMetrics.empty()

    # Define sharding for model input and output data
    # pytype: disable=wrong-arg-types
    # creating pytree sharding with value types as sharding
    self.pipeline_input_sharding = self.global_sharding
    self.output_sharding = ShakespeareModelOutput(
        metrics_update=self.replicated_sharding
    )

    # Define sharding for pipeline state
    self.pipeline_state_sharding = ep_utils.get_pipeline_state_sharding(
        pipeline_state_cls=ShakespeareModelPipelineState,
        sparse_input_sharding=self.global_sharding,
        dense_input_sharding=self.global_sharding,
        pipeline_output_sharding=self.output_sharding,
        tc_aux_sharding=self.replicated_sharding,
    )
    # pytype: enable=wrong-arg-types

    # Initialize the model.
    self.model, self.optimizer, self.train_state, self.feature_specs = (
        create_train_state(
            jax.random.key(42),
            self.num_global_devices,
            self.num_sc_per_device,
            _GLOBAL_BATCH_SIZE.value,
            _VOCAB_SIZE.value,
            _SEQ_LEN.value,
            _EMBEDDING_SIZE.value,
        )
    )

    # Define SparseCore forward and backward functions
    self.sc_fwd_function = ep_utils.get_default_sc_fwd_function(
        self.feature_specs, self.global_mesh
    )
    self.sc_bwd_function = ep_utils.get_default_sc_bwd_function(
        self.feature_specs, self.global_mesh
    )

    # Define batch sizes
    # Note 1: InputProcessing is currently global so all the input batch
    # features and labels are global as well.
    #
    # Note 2: The input processor expects 2-d lookups, so we scale-up the batch
    # size and reshape the results.

    self.local_batch_size = _GLOBAL_BATCH_SIZE.value // self.num_processes
    self.device_batch_size = _GLOBAL_BATCH_SIZE.value // self.num_global_devices
    info(
        'batch sizes: global=%s, local=%s, device=%s',
        _GLOBAL_BATCH_SIZE.value,
        self.local_batch_size,
        self.device_batch_size,
    )

    # Define per SparseCore vocabulary size
    self.per_sc_vocab_size = _VOCAB_SIZE.value // self.num_sc_per_device
    if self.per_sc_vocab_size < 8 or self.per_sc_vocab_size % 8 != 0:
      raise ValueError(
          'Vocabulary size must be a multiple of 8 per SC: VOCAB_SIZE ='
          f' {_VOCAB_SIZE.value}, num_scs = {self.num_sc_per_device}'
      )

    # Load Shakespeare data
    self.word_ids = shakespeare_data.load_shakespeare(_VOCAB_SIZE.value)
    vlog1('word_ids len = %s', len(self.word_ids))
    self.feature_batches, self.label_batches = shakespeare_data.word_id_batches(
        self.word_ids,
        _NUM_STEPS.value,
        _GLOBAL_BATCH_SIZE.value,
        _SEQ_LEN.value,
        _NUM_TABLES.value,
    )
    self.feature_batches = self.feature_batches['words_0']
    vlog1('feature_batches len = %s', len(self.feature_batches))
    vlog1('feature_batches[0] shape = %s', self.feature_batches[0].shape)
    vlog1('label_batches len = %s', len(self.label_batches))
    vlog1('label_batches[0] shape = %s', self.label_batches[0].shape)

    # Define table specs
    self.table_specs = {
        f.table_spec.name: f.table_spec
        for f in jax.tree.leaves(self.feature_specs)
    }

    # Initialize embedding variables
    self.emb_variables = embedding.init_embedding_variables(
        jax.random.key(13),
        self.table_specs,
        self.global_emb_sharding,
        self.num_sc_per_device,
    )

    # NOTE: donation to avoid copies (results in 6 copies from 20). Before:
    #   http://screen/5hBmNR9YPC7DbhE After: http://screen/84U79owzSmas7Uk
    # NOTE: specifying the shardings reduces the copies to 2.

    # Define training step function
    self.train_step = jax.jit(
        self._train_step,
        in_shardings=(
            self.pipeline_input_sharding,
            self.replicated_sharding,
            self.global_emb_layout,
            self.pipeline_state_sharding,
        ),
        out_shardings=(
            self.output_sharding,
            self.replicated_sharding,
            self.replicated_sharding,
            self.global_emb_layout,
            self.pipeline_state_sharding,
        ),
        donate_argnums=(0, 1, 2, 3),
        static_argnames=['fake_tc_step'],
    )
    self.fake_tc_train_step = jax.jit(
        self._train_step,
        in_shardings=(
            self.pipeline_input_sharding,
            self.replicated_sharding,
            self.global_emb_layout,
            self.pipeline_state_sharding,
        ),
        out_shardings=(
            self.output_sharding,
            self.replicated_sharding,
            self.replicated_sharding,
            self.global_emb_layout,
            self.pipeline_state_sharding,
        ),
        donate_argnums=(0, 1, 2, 3),
        static_argnames=['fake_tc_step'],
    )

    # Distributed training.
    parameter_overview.log_parameter_overview(self.train_state.params)

  def _tc_forward_backward_pass(
      self,
      emb_act: ep_utils.DefaultEmbeddingActivations,
      dense_inputs: ShakespeareModelDenseInput,
      train_state: TrainState,
      sc_fwd_aux: ep_utils.DefaultScFwdAux,
  ) -> tuple[
      ep_utils.DefaultEmbeddingActivations,
      ShakespeareModelOutput,
      TrainState,
      ep_utils.DefaultTcAux,
  ]:
    """TensorCore forward + backward pass."""
    del sc_fwd_aux

    emb_act = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (self.model.global_batch_size, -1)),
        emb_act,
    )
    with jax.named_scope('tc_value_and_grad'):
      loss_grad_fn = jax.value_and_grad(
          partial(shakespeare_model.loss, self.model),
          argnums=(0, 1),
          has_aux=True,
      )

      (loss, logits), (dense_grad, emb_grad) = loss_grad_fn(
          train_state.params,
          emb_act,
          dense_inputs.labels,
      )

    with jax.named_scope('tc_update'):
      updates, new_opt_state = self.optimizer.update(
          dense_grad, train_state.opt_state, train_state.params
      )
      new_params = optax.apply_updates(train_state.params, updates)

    emb_grad = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, self.model.embedding_size)), emb_grad
    )

    train_state = train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
        step=train_state.step + 1,
    )

    metrics_update = TrainMetrics.single_from_model_output(
        loss=loss,
        logits=logits,
        labels=dense_inputs.labels,
    )

    model_output = ShakespeareModelOutput(metrics_update=metrics_update)
    return emb_grad, model_output, train_state, None

  def _prepare_batch(
      self, step: int
  ) -> ShakesperaeModelPipelineCurrentStepInput:
    """Prepares a batch for training."""
    features_i = self.feature_batches[step]
    labels_i = self.label_batches[step]

    # These are currently global batches so each task needs to offset into the
    # data for it's local slice.
    labels = labels_i[
        self.process_id
        * self.local_batch_size : (self.process_id + 1)
        * self.local_batch_size
    ]
    labels = jax.make_array_from_process_local_data(
        self.global_sharding, labels
    )

    # Each input preprocessing processes the current process's slice of the
    # global batch.
    features = features_i[
        self.process_id
        * self.local_batch_size : (self.process_id + 1)
        * self.local_batch_size
    ]
    features = np.reshape(features, (-1, 1))
    features_weights = np.ones(features.shape, dtype=np.float32)

    # Pack the features into a tree structure.
    feature_structure = jax.tree.structure(self.feature_specs)
    features = jax.tree_util.tree_unflatten(feature_structure, [features])
    features_weights = jax.tree_util.tree_unflatten(
        feature_structure, [features_weights]
    )

    # Preprocess the inputs and build JAX global views of the data.
    def make_global_view(x: PyTree) -> PyTree:
      """Makes a global view of this process's slice of preprocessed data."""
      return jax.tree.map(
          lambda y: jax.make_array_from_process_local_data(
              self.global_sharding, y
          ),
          x,
      )

    # Note: eventually the preprocessing will dynamically adjust the limits
    #   and, hence, update the feature specs. For now, we just use the
    #   original specs.
    preprocessed_inputs, _ = embedding.preprocess_sparse_dense_matmul_input(
        features=features,
        features_weights=features_weights,
        feature_specs=self.feature_specs,
        local_device_count=self.global_mesh.local_mesh.size,
        global_device_count=self.global_mesh.size,
        num_sc_per_device=self.num_sc_per_device,
        sharding_strategy='MOD',
        batch_number=step,
    )
    preprocessed_inputs = make_global_view(preprocessed_inputs)
    return ShakesperaeModelPipelineCurrentStepInput(
        dense_inputs=ShakespeareModelDenseInput(labels=labels),
        sparse_inputs=preprocessed_inputs,
    )

  def _train_step(
      self,
      pipeline_input: ShakesperaeModelPipelineCurrentStepInput,
      train_state: TrainState,
      embedding_variables: NestedEmbeddingVariables,
      pipeline_state: ShakespeareModelPipelineState,
      fake_tc_step: bool,
  ) -> tuple[
      ShakespeareModelOutput,
      ep_utils.DefaultScBwdAux,
      TrainState,
      NestedEmbeddingVariables,
      ShakespeareModelPipelineState,
  ]:
    """Performs a single training step."""

    return ep_utils.step(
        pipeline_input=pipeline_input,
        tc_train_state=train_state,
        embedding_variables=embedding_variables,
        pipeline_state=pipeline_state,
        sc_fwd_function=self.sc_fwd_function,
        tc_function=self._tc_forward_backward_pass,
        sc_bwd_function=self.sc_bwd_function,
        fake_tc_step=fake_tc_step,
    )

  def _post_train_step(self, step: int, model_output: ShakespeareModelOutput):
    """Performs any special handling or consumption of the train step outputs."""

    if ep_utils.is_output_valid(step, _NUM_STEPS.value):
      self.train_metrics = self.train_metrics.merge(model_output.metrics_update)

    if (step + 1) % _LOG_FREQUENCY.value == 0:
      m = self.train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(self.train_state.params)

    if (step + 1) % _METRICS_RESET_FREQUENCY.value == 0:
      self.train_metrics = TrainMetrics.empty()

  def test_run_shakespeare_pipelined_model(self):
    """Runs the Shakespeare model with SparseCore embedding pipelining.

    This test trains the model for a number of steps and then checks that the
    final loss is less than a certain threshold.
    """

    pipeline_state: ShakespeareModelPipelineState = None
    pipeline_input: ShakesperaeModelPipelineCurrentStepInput = None
    train_step = None
    fake_tc_train_step = None

    num_steps = _NUM_STEPS.value

    step_counter = 0
    while self.train_state.step < num_steps:

      with jax.profiler.StepTraceAnnotation(
          'train', step_num=self.train_state.step
      ):
        vlog1('*' * 70)
        vlog1('* STEP = %s', step_counter)
        vlog1('*' * 70)

        if step_counter < num_steps:
          pipeline_input = self._prepare_batch(step_counter)
        else:
          # Use dummy input based on last input
          pipeline_input = jax.tree.map(jnp.zeros_like, pipeline_input)

        pipeline_input = jax.device_put(
            pipeline_input, self.pipeline_input_sharding
        )

        if step_counter == 0:
          pipeline_state = ep_utils.get_initial_state(
              pipeline_input=pipeline_input,
              tc_train_state=self.train_state,
              embedding_variables=self.emb_variables,
              sc_fwd_function=self.sc_fwd_function,
              tc_function=self._tc_forward_backward_pass,
          )
          pipeline_state = jax.device_put(
              pipeline_state, self.pipeline_state_sharding
          )
          # compiles the function for given inputs and complains if they
          # change (prevents re-tracing)
          fake_tc_train_step = self.train_step.lower(
              pipeline_input,
              self.train_state,
              self.emb_variables,
              pipeline_state,
              True,
          ).compile()
          train_step = self.train_step.lower(
              pipeline_input,
              self.train_state,
              self.emb_variables,
              pipeline_state,
              False,
          ).compile()

        state_structure = jax.tree.structure(pipeline_state)

        train_step_fn = (
            train_step
            if ep_utils.is_output_valid(step_counter, num_steps)
            else fake_tc_train_step
        )
        assert train_step_fn is not None
        (
            model_output,
            _,  # sc_bwd_aux
            new_train_state,
            new_embedding_variables,
            new_pipeline_state,
        ) = train_step_fn(
            pipeline_input,
            self.train_state,
            self.emb_variables,
            pipeline_state,
        )

        self.assertEqual(
            state_structure,
            jax.tree.structure(new_pipeline_state),
            'pipeline_state and new_pipeline_state must have the same'
            ' structure',
        )
        pipeline_state = new_pipeline_state
        self.train_state = new_train_state
        self.emb_variables = new_embedding_variables

        self._post_train_step(step_counter, model_output)

        step_counter += 1

    final_loss = self.train_metrics.compute()['train_loss']
    logging.info('final_loss: %s', final_loss)
    # Deterministic with initial seed
    np.testing.assert_allclose(final_loss, 0.00204, atol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
