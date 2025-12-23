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
"""An example Shakespeare model that uses the SparseCore embedding API.

This is all the generic stuff that's common across all the Shakespeare models.
Generally, it's not very interesting so it's hidden here to keep the
other files more readable.
"""

# pylint: disable=g-importing-member
import dataclasses
import pprint

from absl import flags
from absl import logging
from flax import linen as nn
import jax
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np

Nested = embedding.Nested


_SHARDING_AXIS = 'device'
_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 2048, 'Vocabulary size.')
_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 256, 'Global batch size.'
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.005, 'Learning rate.')
_SEQ_LEN = flags.DEFINE_integer(
    'sequence_length', 16, 'Sequence length of context words.'
)
_NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 100, 'Number of steps to train for.'
)
_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 8, 'Embedding size.')
_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency', 10, 'Frequency to log metrics.'
)
_LOSS_RESET_FREQUENCY = flags.DEFINE_integer(
    'loss_window', 10, 'Number of steps to average loss over.'
)
_ENABLE_MINIBATCHING = flags.DEFINE_bool(
    'enable_minibatching',
    False,
    'If set, minibatching will be enabled.',
)


# Config: A namedtuple that holds all the configuration metadata.
@dataclasses.dataclass(frozen=True)
class Config:
  """Configuration for the Shakespeare model."""

  global_devices: list[jax.Device]
  local_devices: list[jax.Device]
  num_global_devices: int
  num_local_devices: int
  num_sc_per_device: int
  num_processes: int
  process_id: int
  sharding_axis: str
  vocab_size: int
  global_batch_size: int
  local_batch_size: int
  device_batch_size: int
  seq_len: int
  embedding_size: int
  learning_rate: float
  num_steps: int
  log_frequency: int
  loss_reset_frequency: int
  enable_minibatching: bool
  feature_name: str = 'shakespeare_feature'
  table_name: str = 'shakespeare_table'


def get_config() -> Config:
  """Returns the configuration for the Shakespeare model.

  This is a dumping ground for all configuration options. It's really just meant
  to minimize the amount of code in the trainer and model files.
  """
  local_devices = jax.local_devices()
  global_devices = jax.devices()
  num_global_devices = len(global_devices)
  num_local_devices = len(local_devices)
  num_sc_per_device = utils.num_sparsecores_per_device(global_devices[0])

  num_processes = jax.process_count()
  process_id = jax.process_index()
  logging.info(
      'process_id %s:: num devices: local = %s, global = %s',
      process_id,
      num_local_devices,
      num_global_devices,
  )

  config = Config(
      global_devices=global_devices,
      local_devices=local_devices,
      num_global_devices=num_global_devices,
      num_local_devices=num_local_devices,
      num_sc_per_device=num_sc_per_device,
      num_processes=num_processes,
      process_id=process_id,
      sharding_axis=_SHARDING_AXIS,
      vocab_size=_VOCAB_SIZE.value,
      global_batch_size=_GLOBAL_BATCH_SIZE.value,
      local_batch_size=_GLOBAL_BATCH_SIZE.value // num_processes,
      device_batch_size=_GLOBAL_BATCH_SIZE.value // num_global_devices,
      seq_len=_SEQ_LEN.value,
      embedding_size=_EMBEDDING_SIZE.value,
      learning_rate=_LEARNING_RATE.value,
      num_steps=_NUM_STEPS.value,
      log_frequency=_LOG_FREQUENCY.value,
      loss_reset_frequency=_LOSS_RESET_FREQUENCY.value,
      enable_minibatching=_ENABLE_MINIBATCHING.value,
  )
  logging.info('Shakespeare Config = %s', pprint.pformat(config))

  per_sc_vocab_size = config.vocab_size // config.num_sc_per_device
  if per_sc_vocab_size < 8 or per_sc_vocab_size % 8 != 0:
    raise ValueError(
        'Vocabulary size must be a multiple of 8 per SC: VOCAB_SIZE ='
        f' {config.vocab_size}, num_scs = {config.num_sc_per_device}'
    )

  return config


def local_slice(config: Config, x: embedding.ArrayLike) -> embedding.ArrayLike:
  """Batch data is read for the global model. This creates a local slice."""
  return x[
      config.process_id
      * config.local_batch_size : (config.process_id + 1)
      * config.local_batch_size
  ]


def device_slice(
    config: Config,
    x: embedding.ArrayLike,
    data_sharding: jax.sharding.NamedSharding,
) -> embedding.ArrayLike:
  """Like local_slice, but creates an on-device JAX array."""
  return jax.make_array_from_process_local_data(
      data_sharding, local_slice(config, x)
  )


def process_inputs(
    config: Config,
    feature_specs: Nested[embedding_spec.FeatureSpec],
    batch_number: int,
    feature_batch: embedding.ArrayLike,
    data_sharding: jax.sharding.NamedSharding | None,
    has_leading_dimension: bool = False,
) -> tuple[embedding.PreprocessedInput, embedding.SparseDenseMatmulInputStats]:
  """Preprocess a Shakespeare batch into PreprocessedInput and stats.

  Args:
    config: The configuration.
    feature_specs: The feature specs.
    batch_number: The batch number.
    feature_batch: The feature batch.
    data_sharding: The NamedSharding for the data.
    has_leading_dimension: Whether the feature batch has a leading dimension.

  Returns:
    A tuple of PreprocessedInput and SparseDenseMatmulInputStats.
  """
  features = np.reshape(feature_batch, (-1, 1))

  # Pack the features into a tree structure.
  feature_structure = jax.tree.structure(feature_specs)
  features = jax.tree_util.tree_unflatten(feature_structure, [features])
  processed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
      features=features,
      features_weights=None,  # uniform weights
      feature_specs=feature_specs,
      local_device_count=config.num_local_devices,
      global_device_count=config.num_global_devices,
      num_sc_per_device=config.num_sc_per_device,
      sharding_strategy='MOD',
      has_leading_dimension=has_leading_dimension,
      batch_number=batch_number,
  )
  if data_sharding is not None:
    processed_inputs = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(data_sharding, x),
        processed_inputs,
    )
  return processed_inputs, stats


def create_feature_specs(config: Config) -> Nested[embedding_spec.FeatureSpec]:
  """Creates the feature specs for the Shakespeare model.

  Args:
    config: The configuration.

  Returns:
    A Nested structure of FeatureSpecs.
  """
  ## Embedding API: TableSpec and FeatureSpec creation
  table_spec = embedding_spec.TableSpec(
      vocabulary_size=config.vocab_size,
      embedding_dim=config.embedding_size,
      initializer=jax.nn.initializers.normal(),
      optimizer=embedding_spec.AdamOptimizerSpec(
          learning_rate=config.learning_rate
      ),
      combiner='sum',
      name='shakespeare_table',
      max_ids_per_partition=64,
      max_unique_ids_per_partition=64,
  )
  feature_spec = embedding_spec.FeatureSpec(
      table_spec=table_spec,
      input_shape=(config.global_batch_size * config.seq_len, 1),
      output_shape=(
          config.global_batch_size * config.seq_len,
          config.embedding_size,
      ),
      name='shakespeare_feature',
  )
  feature_specs = {feature_spec.name: feature_spec}
  # This call will take care of stacking features and other automatable
  # configuration settings.
  embedding.prepare_feature_specs_for_training(
      feature_specs,
      global_device_count=config.num_global_devices,
      num_sc_per_device=config.num_sc_per_device,
  )
  feature_specs = nn.FrozenDict({feature_spec.name: feature_spec})
  return feature_specs


def get_batches(config: Config) -> tuple[list[list[int]], list[list[int]]]:
  """Returns the batches for the Shakespeare model."""
  word_ids = shakespeare_data.load_shakespeare(config.vocab_size)
  num_tables = 1
  feature_batches, label_batches = shakespeare_data.word_id_batches(
      word_ids,
      config.num_steps,
      config.global_batch_size,
      config.seq_len,
      num_tables,
  )
  feature_batches = feature_batches['words_0']
  return feature_batches, label_batches


def step_header(step: int) -> str:
  """Returns a header for a step."""
  line = f'* STEP = {step} '
  line = line + '*' * (80 - len(line))
  return line
