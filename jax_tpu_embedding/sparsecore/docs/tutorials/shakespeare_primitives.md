# The JAX SC Shakespeare Example - Primitives

_An example Shakespeare model that uses the low-level primitives of the SparseCore embedding API_

Using some text from the Shakespeare corpus, we're going to construct a model that predicts the next word for a given sequence of words. This tutorial is based on the working example in [.../examples/shakespeare/jax_sc_shakespeare_jit.py](https://github.com/jax-ml/jax-tpu-embedding/blob/main/jax_tpu_embedding/sparsecore/examples/shakespeare/jax_sc_shakespeare_jit.py) using the low-level primitves for the embedding lookup and gradient updates.

In this example we'll walk through the construction and configuration of the model paying particular attention to the embedding aspects.

This example domonstrates the following features of the JAX SC API:
- Input preprocessing
- TableSpec and FeatureSpec creation
- `jax.jit` + `shard_map` execution
- Direct execution of embedding lookup and gradient update for embeddings
- FDO (feedback directed optimization)

# Front matter

We start off with some basic imports and some settings. In a real example, these settings would come from flags or other model configuration.


```python
from functools import partial
import os
from pprint import pformat, pprint
from typing import Any, Mapping

from clu import metrics
from clu import parameter_overview
import flax
from flax import linen as nn
import jax
from jax.experimental.layout import Format
from jax.experimental.layout import Layout
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax

np.set_printoptions(threshold=np.inf)
Nested = embedding.Nested
```


```python
VOCAB_SIZE = 1024  # Maximum number of unique words.
GLOBAL_BATCH_SIZE = 32
LEARNING_RATE = 0.005
SEQ_LEN = 16  # Sequence length of context words.
NUM_TABLES = 1  # Number of tables to create.
NUM_STEPS = 100
NUM_EPOCHS = 1
EMBEDDING_SIZE = 8
EMBEDDING_INIT = 'normal'
LOG_FREQUENCY = 10
LOSS_RESET_FREQUENCY = 10  # Number of steps to average loss over.
CHECKPOINT_DIR = None  # If set, checkpoints will be written to the directory.
CHECKPOINT_INTERVAL = 500
CHECKPOINT_RESUME = True
CHECKPOINT_MAX_TO_KEEP = 5
FDO_DIR = '/tmp'
FDO_FREQUENCY = 15  # How frequently, in steps, to update FDO stats
```


```python
# Device and cluster configuration
local_devices = jax.local_devices()
global_devices = jax.devices()
num_global_devices = len(global_devices)
num_local_devices = len(local_devices)
num_sc_per_device = utils.num_sparsecores_per_device(global_devices[0])
num_processes = jax.process_count()
process_id = jax.process_index()

# PartitionSpecs for the model and embedding tables.
pd = P('device')  # Device sharding.
pe = P('device', None)  # PartitionSpec for embedding tables.

# Create the global mesh and shardings.
global_mesh = Mesh(np.array(global_devices), axis_names=['device'])
global_sharding = NamedSharding(global_mesh, pd)
global_emb_sharding = NamedSharding(global_mesh, pe)
```


```python
print(
    f'num devices: local = {num_local_devices}, global = {num_global_devices}'
)
print(f'process_id = {process_id}, num_processes = {num_processes}')
print(f'local_devices = {pformat(local_devices)}')
print(f'global_devices = {pformat(global_devices)}')
```


```python
# Define Train State and Metrics classes.
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
```

# The Shakespeare Model

We're basing this on the example  model in [.../examples/models/shakespeare/model.py](https://github.com/jax-ml/jax-tpu-embedding/blob/main/jax_tpu_embedding/sparsecore/examples/models/shakespeare/model.py)

The Shakespeare model consists of an embedding layer and two dense layers. This Flax model takes the already computed embedding activations which are of length `embedding_size`. Since any word may be predicted, the output of the dense model is the same size as the embedding vocabulary.


```python
# Flax Model
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

  @nn.compact
  def __call__(self, emb_activations: Mapping[str, jax.Array]):
    # Unpack the activations.
    x = emb_activations[self.feature_name]
    x = jnp.reshape(x, (x.shape[0], -1))
    # Apply the model.
    x = nn.Dense(self.embedding_size)(x)
    x = nn.Dense(self.vocab_size)(x)
    return x


model = Model(
    global_batch_size=GLOBAL_BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    embedding_size=EMBEDDING_SIZE,
)
```

## The Loss Function

Here we define a simple loss function for our model.


```python
# The loss function
def loss_fn(
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
```

# Embedding API: TableSpec and FeatureSpec creation

A <project:#TableSpec> describes a single embedding table which will be used by one or more sparse features in the model. We configure it's size (vocabulary size and embedding dimension), initialization method, and the optimizer to use for the weight update. The `combiner` parameter determines how multivalent features are combined into a single output activation. The `max_ids_per_partition` and `max_unique_ids_per_partition` parameters are described in <project:../parameters.rst>.

A <project:#FeatureSpec> describes the batch input data for a given feature. We configure which table to use for the embedding lookup and the input and output shapes. In this example we're using a sequence of `SEQ_LEN` words for this feature so the input data has `GLOBAL_BATCH_SIZE * SEQ_LEN` tokens.

In general, models may have multiple tables and multiple features. Furthermore, multiple features may share a single embedding table.

More information on embedding specifications can be found in <project:../embedding.rst>.


```python
# TableSpec
table_spec = embedding_spec.TableSpec(
    vocabulary_size=model.vocab_size,
    embedding_dim=model.embedding_size,
    initializer=jax.nn.initializers.zeros,
    optimizer=embedding_spec.SGDOptimizerSpec(),
    combiner='sum',
    name=model.table_name,
    max_ids_per_partition=64,
    max_unique_ids_per_partition=64,
)
pprint(table_spec)
```

    TableSpec(name='shakespeare_table',
              vocabulary_size=1024,
              embedding_dim=8,
              initializer=<function zeros at 0x70b5cb9c5da0>,
              optimizer=SGDOptimizerSpec(),
              combiner='sum',
              max_ids_per_partition=64,
              max_unique_ids_per_partition=64,
              suggested_coo_buffer_size_per_device=None,
              quantization_config=None,
              _stacked_table_spec=None,
              _setting_in_stack=TableSettingInStack(stack_name='shakespeare_table',
                                                    padded_vocab_size=1024,
                                                    padded_embedding_dim=8,
                                                    row_offset_in_shard=0,
                                                    shard_rotation=0))



```python
# FeatureSpec
feature_spec = embedding_spec.FeatureSpec(
    table_spec=table_spec,
    input_shape=(model.global_batch_size * model.seq_len, 1),
    output_shape=(
        model.global_batch_size * model.seq_len,
        model.embedding_size,
    ),
    name=model.feature_name,
)
feature_specs = nn.FrozenDict({model.feature_name: feature_spec})

# This call will take care of stacking features and other automatable
# configuration settings.
embedding.prepare_feature_specs_for_training(
    feature_specs,
    global_device_count=num_global_devices,
    num_sc_per_device=num_sc_per_device,
)
for fs in feature_specs.values():
  pprint(fs)

table_specs = {
    f.table_spec.name: f.table_spec for f in jax.tree.leaves(feature_specs)
}
```

    FeatureSpec(name='shakespeare_feature',
                table_spec=TableSpec(name='shakespeare_table',
                                     vocabulary_size=1024,
                                     embedding_dim=8,
                                     initializer=<function zeros at 0x70b5cb9c5da0>,
                                     optimizer=SGDOptimizerSpec(),
                                     combiner='sum',
                                     max_ids_per_partition=64,
                                     max_unique_ids_per_partition=64,
                                     suggested_coo_buffer_size_per_device=None,
                                     quantization_config=None,
                                     _stacked_table_spec=StackedTableSpec(stack_name='shakespeare_table',
                                                                          stack_vocab_size=1024,
                                                                          stack_embedding_dim=8,
                                                                          optimizer=SGDOptimizerSpec(),
                                                                          combiner='sum',
                                                                          total_sample_count=np.int64(512),
                                                                          max_ids_per_partition=64,
                                                                          max_unique_ids_per_partition=64,
                                                                          suggested_coo_buffer_size_per_device=None,
                                                                          quantization_config=None),
                                     _setting_in_stack=TableSettingInStack(stack_name='shakespeare_table',
                                                                           padded_vocab_size=1024,
                                                                           padded_embedding_dim=8,
                                                                           row_offset_in_shard=0,
                                                                           shard_rotation=0)),
                input_shape=(512, 1),
                output_shape=(512, 8),
                _id_transformation=FeatureIdTransformation(row_offset=0,
                                                           col_offset=0,
                                                           col_shift=0))


# Model Initialization

Here, we initialize the model. We start by creating a zero-array for the embedding activations which are then used as inputs to initialize the dense model. We also initialize the emebdding tables with the initialization functions specified in the `TableSpec`s.


```python
# Initialize the model and training state.

# Global embedding activations. We can change this to local arrays and use
# make_array_from_single_device_arrays as above.
init_emb_activations = {
    model.feature_name: jnp.zeros((
        model.global_batch_size,
        model.seq_len,
        1,
        model.embedding_size,
    ))
}
rng = jax.random.key(42)
params = model.init(rng, init_emb_activations)
print(f'params =\n {parameter_overview.get_parameter_overview(params)}')

optimizer = optax.adam(learning_rate=LEARNING_RATE)

train_state = TrainState(
    params=params,
    opt_state=optimizer.init(params),
)

emb_variables = embedding.init_embedding_variables(
    jax.random.key(13), table_specs, global_emb_sharding, num_sc_per_device
)

# Outsharding the embedding variables. Note that arrays on the SparseCore are
# expected to have row major layout while the default layout is column major.
# Here, we explicitly set the layout to row major.
emb_var_outsharding = utils.embedding_table_format(
    global_emb_sharding.mesh, global_emb_sharding.spec
)
```

    params =
     +-----------------------+-----------+---------+-------+---------+--------+
    | Name                  | Shape     | Dtype   | Size  | Mean    | Std    |
    +-----------------------+-----------+---------+-------+---------+--------+
    | params/Dense_0/bias   | (8,)      | float32 | 8     | 0.0     | 0.0    |
    | params/Dense_0/kernel | (128, 8)  | float32 | 1,024 | 0.00226 | 0.0887 |
    | params/Dense_1/bias   | (1024,)   | float32 | 1,024 | 0.0     | 0.0    |
    | params/Dense_1/kernel | (8, 1024) | float32 | 8,192 | 0.00592 | 0.354  |
    +-----------------------+-----------+---------+-------+---------+--------+
    Total: 10,248 -- 40,992 bytes


# Model Data

For this simple model, we preload all the data into fixed arrays.


```python
# Preload the model data

# Note: The input processor expects 2-d lookups, so we scale-up the batch
# size and reshape the results.

local_batch_size = GLOBAL_BATCH_SIZE // num_processes
device_batch_size = GLOBAL_BATCH_SIZE // num_global_devices
print(
    f'batch sizes: global={GLOBAL_BATCH_SIZE}, local={local_batch_size},'
    f' device={device_batch_size}'
)

per_sc_vocab_size = VOCAB_SIZE // num_sc_per_device
if per_sc_vocab_size < 8 or per_sc_vocab_size % 8 != 0:
  raise ValueError(
      'Vocabulary size must be a multiple of 8 per SC: VOCAB_SIZE ='
      f' {VOCAB_SIZE}, num_scs = {num_sc_per_device}'
  )

word_ids = shakespeare_data.load_shakespeare(VOCAB_SIZE)
print(f'word_ids len = {len(word_ids)}')
feature_batches, label_batches = shakespeare_data.word_id_batches(
    word_ids,
    NUM_STEPS,
    GLOBAL_BATCH_SIZE,
    SEQ_LEN,
    NUM_TABLES,
)
feature_batches = feature_batches['words_0']
print(f'feature_batches len = {len(feature_batches)}')
print(f'feature_batches[0] shape = {feature_batches[0].shape}')
print(f'label_batches len = {len(label_batches)}')
print(f'label_batches[0] shape = {label_batches[0].shape}')
```

    batch sizes: global=32, local=32, device=8
    word_ids len = 454
    feature_batches len = 1000
    feature_batches[0] shape = (32, 16, 1)
    label_batches len = 1000
    label_batches[0] shape = (32,)


# The Training Step Function

The training step function takes a step by
1. Performing the embedding lookup.
2. Passing the activations as inputs to the dense model to:
  - (a) compute the loss and dense gradients and,
  - (b) update the dense weights.
3. Using the back propagated dense gradients to update the embedding weights.


```python
@partial(
    jax.jit,
    static_argnums=(0, 1, 2, 3),
    out_shardings=(
        None,
        None,
        emb_var_outsharding,
    ),
    donate_argnums=(6),
)
def train_step_fn(
    mesh: jax.sharding.Mesh,
    model: nn.Module,
    optimizer,
    feature_specs,
    train_state: TrainState,
    preprocessed_inputs,
    emb_variables,
    labels,
) -> tuple[TrainState, TrainMetrics, Nested[jax.Array]]:
  """Performs a single training step at the chip level."""

  ##############################################################################
  # Sparse forward pass - embedding lookup.
  ##############################################################################
  tpu_sparse_dense_matmul = partial(
      embedding.tpu_sparse_dense_matmul,
      global_device_count=num_global_devices,
      feature_specs=feature_specs,
      sharding_strategy='MOD',
  )
  tpu_sparse_dense_matmul = jax.shard_map(
      tpu_sparse_dense_matmul,
      mesh=mesh,
      in_specs=(pd, pe),
      out_specs=pd,
      check_vma=False,
  )
  emb_act = tpu_sparse_dense_matmul(
      preprocessed_inputs,
      emb_variables,
  )

  ##############################################################################
  # Dense forward + backward pass.
  ##############################################################################
  emb_act = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (model.global_batch_size, -1)), emb_act
  )
  loss_grad_fn = jax.value_and_grad(
      partial(loss_fn, model), argnums=(0, 1), has_aux=True
  )

  (loss, logits), (dense_grad, emb_grad) = loss_grad_fn(
      train_state.params, emb_act, labels
  )

  updates, new_opt_state = optimizer.update(
      dense_grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)

  emb_grad = jax.tree_util.tree_map(
      lambda x: jnp.reshape(x, (-1, model.embedding_size)), emb_grad
  )

  ##############################################################################
  # Sparse backward pass - embedding update.
  ##############################################################################
  tpu_sparse_dense_matmul_grad = partial(
      embedding.tpu_sparse_dense_matmul_grad,
      feature_specs=feature_specs,
      sharding_strategy='MOD',
  )
  tpu_sparse_dense_matmul_grad = jax.shard_map(
      tpu_sparse_dense_matmul_grad,
      mesh=mesh,
      in_specs=(pd, pd, pe),
      out_specs=pe,
      check_vma=False,
  )
  emb_variables = tpu_sparse_dense_matmul_grad(
      emb_grad,
      preprocessed_inputs,
      emb_variables,
  )

  train_state = train_state.replace(params=new_params, opt_state=new_opt_state)

  metrics_update = TrainMetrics.single_from_model_output(
      loss=loss,
      logits=logits,
      labels=labels,
  )

  return train_state, metrics_update, emb_variables
```

# FDO Configuration

FDO (Feedback Directed Optimization) is an optional feature where we use
batch statistics to update table limits (`max_ids_per_partition`, etc.). The FDO
client is a generic interface and clients can either use provided
implementations or create their own that intergrates with their infrastructure.
When the FDO stats are used to update the limits on the `feature_specs`,
`jax.jit` triggers a recompilation.

In this example, we use a simple file-based implementation. Each host writes its
own stats on `publish()` and stats from all hosts are merged during `load()`.


```python
out_path = os.path.join(FDO_DIR, 'fdo_dump')
os.makedirs(out_path, exist_ok=True)
print(f'FDO storage path: {out_path}')
fdo_client = file_fdo_client.NPZFileFDOClient(out_path)
```

# The Training Loop

The training loop is a simple Python `for`-loop. For each step, we preprocess the input batch data (on the host / CPU). In doing so, we capture the batch statistics which are used by FDO. Then we take a training step using the jitted `train_step_fn`. We finish each step by processing the training metrics and FDO stats.


```python
parameter_overview.log_parameter_overview(train_state.params)
train_metrics = None
step = 0
# Track these separately for testing.
test_loss = None
test_step_count = 0
for features, labels in zip(feature_batches, label_batches):
  step += 1
  test_step_count += 1
  print(f'Step {step}')
  ##############################################################################
  # SC input processing.
  ##############################################################################
  # These are currently global batches so each task needs to offset into
  # the data for it's local slice.
  labels = labels[
      process_id * local_batch_size : (process_id + 1) * local_batch_size
  ]
  labels = jax.make_array_from_process_local_data(global_sharding, labels)

  # Each input preprocessing processes the current process's slice of the
  # global batch.
  features = features[
      process_id * local_batch_size : (process_id + 1) * local_batch_size
  ]
  features = np.reshape(features, (-1, 1))
  feature_weights = np.ones(features.shape, dtype=np.float32)

  # Pack the features into a tree structure.
  feature_structure = jax.tree.structure(feature_specs)
  features = jax.tree_util.tree_unflatten(feature_structure, [features])
  feature_weights = jax.tree_util.tree_unflatten(
      feature_structure, [feature_weights]
  )

  ##############################################################################
  # Preprocess the inputs and build JAX global views of the data.
  ##############################################################################
  make_global_view = lambda x: jax.tree.map(
      lambda y: jax.make_array_from_process_local_data(global_sharding, y),
      x,
  )
  preprocessed_inputs, step_stats = (
      embedding.preprocess_sparse_dense_matmul_input(
          features,
          feature_weights,
          feature_specs,
          local_device_count=global_mesh.local_mesh.size,
          global_device_count=global_mesh.size,
          num_sc_per_device=num_sc_per_device,
          sharding_strategy='MOD',
          batch_number=step,
      )
  )
  preprocessed_inputs = make_global_view(preprocessed_inputs)
  fdo_client.record(step_stats)

  ##############################################################################
  # Combined: SC forward, TC, SC backward
  ##############################################################################
  # We capture the logging here so you can see when XLA compilation is
  # triggered. This should happen following the first FDO update to the limits.
  train_state, metrics_update, emb_variables = train_step_fn(
      global_mesh,
      model,
      optimizer,
      feature_specs,
      train_state,
      preprocessed_inputs,
      emb_variables,
      labels,
  )

  train_metrics = (
      metrics_update
      if train_metrics is None
      else train_metrics.merge(metrics_update)
  )

  if step % LOG_FREQUENCY == 0:
    m = train_metrics.compute()
    test_loss = m['train_loss']
    print(f'Step {step}: Loss = {m['train_loss']}')
    parameter_overview.log_parameter_overview(train_state.params)

  if (step + 1) % LOSS_RESET_FREQUENCY == 0:
    train_metrics = None

  if step % FDO_FREQUENCY == 0:
    print(f'Refreshing FDO stats after step {step}')
    fdo_client.publish()
    loaded_stats = fdo_client.load()
    embedding.update_preprocessing_parameters(
        feature_specs, loaded_stats, num_sc_per_device
    )
```


```python
# Print final loss and step count.
print(f'test_loss = {test_loss:.4f}')
print(f'test_step_count = {test_step_count}')
```

    test_loss = 0.0021
    test_step_count = 1000

