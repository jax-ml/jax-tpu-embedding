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
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.sharding import NamedSharding  # pylint: disable=g-importing-member
from jax.sharding import PartitionSpec as P  # pylint: disable=g-importing-member
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import config as shakespeare_config
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

info = logging.info


def create_train_state(
    config: shakespeare_config.Config, rng: jax.Array
) -> tuple[
    nn.Module, optax.GradientTransformation, TrainState, NestedFeatureSpecs
]:
  """Creates and initializes the model, optimizer, train state, and feature specs.

  Args:
    config: The model configuration.
    rng: JAX PRNG Key.

  Returns:
    The model, optimizer, initial train state, and feature specs.
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
      config.feature_name: jnp.zeros(
          (config.global_batch_size, config.seq_len * config.embedding_size)
      )
  }

  params = model.init(rng, init_emb_activations)
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.adam(learning_rate=config.learning_rate)
  feature_specs = shakespeare_config.create_feature_specs(config)

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

    flags.FLAGS.vocab_size = 512
    flags.FLAGS.batch_size = 32
    flags.FLAGS.num_steps = 1000
    self.config = shakespeare_config.get_config()

    # Get number of global and local devices
    self.num_global_devices = self.config.num_global_devices
    self.num_local_devices = self.config.num_local_devices
    self.num_processes = self.config.num_processes
    self.process_id = self.config.process_id
    self.local_devices = self.config.local_devices
    self.global_devices = self.config.global_devices
    self.num_sc_per_device = self.config.num_sc_per_device

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
    self.pd = P(self.config.sharding_axis)  # Device sharding.
    self.pe = P(self.config.sharding_axis, None)  # For embedding tables.

    # Create global mesh and sharding
    self.global_mesh = Mesh(
        np.array(self.global_devices), axis_names=[self.config.sharding_axis]
    )
    self.global_sharding = NamedSharding(self.global_mesh, self.pd)
    self.global_emb_sharding = NamedSharding(self.global_mesh, self.pe)
    self.global_emb_layout = utils.embedding_table_format(
        self.global_emb_sharding.mesh, self.global_emb_sharding.spec
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
        create_train_state(self.config, jax.random.key(42))
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

    self.local_batch_size = self.config.local_batch_size

    # Load Shakespeare data
    self.feature_batches, self.label_batches = shakespeare_config.get_batches(
        self.config
    )

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
    labels = shakespeare_config.device_slice(
        self.config, labels_i, self.global_sharding
    )
    features = shakespeare_config.local_slice(self.config, features_i)

    # Note: eventually the preprocessing will dynamically adjust the limits
    #   and, hence, update the feature specs. For now, we just use the
    #   original specs.
    preprocessed_inputs, _ = shakespeare_config.process_inputs(
        self.config,
        self.feature_specs,
        step,
        features,
        self.global_sharding,
    )
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

    if ep_utils.is_output_valid(step, self.config.num_steps):
      self.train_metrics = self.train_metrics.merge(model_output.metrics_update)

    if (step + 1) % self.config.log_frequency == 0:
      m = self.train_metrics.compute()
      info('Step %s: Loss = %s', step, m['train_loss'])
      parameter_overview.log_parameter_overview(self.train_state.params)

    if (step + 1) % self.config.loss_reset_frequency == 0:
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

    num_steps = self.config.num_steps

    step_counter = 0
    while self.train_state.step < num_steps:

      with jax.profiler.StepTraceAnnotation(
          'train', step_num=self.train_state.step
      ):
        shakespeare_config.step_header(step_counter)

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
    np.testing.assert_allclose(final_loss, 0.000225, atol=1e-5)


if __name__ == '__main__':
  jax.config.update('jax_threefry_partitionable', False)
  absltest.main()
