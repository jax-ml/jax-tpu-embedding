# The JAX SC Shakespeare Example - Flax

_An example Shakespeare model that uses the Flax layer interface of the SparseCore embedding API_

Using some text from the Shakespeare corpus, we're going to construct a Flax model that predicts the next word for a given sequence of words. This tutorial is based on the working example in [.../lib/flax/linen/tests/autograd_test.py](https://github.com/jax-ml/jax-tpu-embedding/blob/main/jax_tpu_embedding/sparsecore/lib/flax/linen/tests/autograd_test.py) using the [SparseCoreEmbed](#jax_tpu_embedding.sparsecore.lib.flax.linen.embed.SparseCoreEmbed) class.

In this example we'll walk through the construction and configuration of the model paying particular attention to the embedding aspects.

This example domonstrates the following features of the JAX SC API:
- Input preprocessing
- <project:#TableSpec> and <project:#FeatureSpec> creation
- Model definition, including the embedding Flax layer
- Execution of the model in a simple training loop
- FDO (feedback directed optimization)

# Front matter

We start off with some basic imports and some settings. In a real example, these settings would come from flags or other model configuration.


```python
import functools
import os
from pprint import pformat, pprint
from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tpu_embedding.sparsecore.examples.models.shakespeare import dataset as shakespeare_data
from jax_tpu_embedding.sparsecore.lib.fdo import file_fdo_client
from jax_tpu_embedding.sparsecore.lib.flax.linen import embed
from jax_tpu_embedding.sparsecore.lib.flax.linen import embed_optimizer
from jax_tpu_embedding.sparsecore.lib.nn import embedding
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec
from jax_tpu_embedding.sparsecore.utils import utils
import numpy as np
import optax

np.set_printoptions(threshold=np.inf)
Nested = embedding.Nested
```


```python
VOCAB_SIZE = 2048  # Maximum number of unique words.
GLOBAL_BATCH_SIZE = 256
LEARNING_RATE = 0.005
SEQ_LEN = 16  # Sequence length of context words.
NUM_TABLES = 1  # Number of tables to create.
NUM_STEPS = 100
NUM_EPOCHS = 1
EMBEDDING_SIZE = 16
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

_SHARDING_AXIS = 'device'

# PartitionSpecs for the model and embedding tables.
pd = P(_SHARDING_AXIS)  # Device sharding.

# Create the global mesh and shardings.
global_mesh = Mesh(np.array(global_devices), axis_names=[_SHARDING_AXIS])
data_sharding = NamedSharding(global_mesh, pd)

local_batch_size = GLOBAL_BATCH_SIZE // num_processes
device_batch_size = GLOBAL_BATCH_SIZE // num_global_devices
```


```python
print(
    f'num devices: local = {num_local_devices}, global = {num_global_devices}'
)
print(f'process_id = {process_id}, num_processes = {num_processes}')
print(f'local_devices = {pformat(local_devices)}')
print(f'global_devices = {pformat(global_devices)}')
print(
    f'batch sizes: global={GLOBAL_BATCH_SIZE}, local={local_batch_size},'
    f' device={device_batch_size}'
)
```

# Embedding API: TableSpec and FeatureSpec creation

A <project:#TableSpec> describes a single embedding table which will be used by one or more sparse features in the model. We configure it's size (vocabulary size and embedding dimension), initialization method, and the optimizer to use for the weight update. The `combiner` parameter determines how multivalent features are combined into a single output activation. The `max_ids_per_partition` and `max_unique_ids_per_partition` parameters are described in <project:../parameters.rst>.

A <project:#FeatureSpec> describes the batch input data for a given feature. We configure which table to use for the embedding lookup and the input and output shapes. In this example we're using a sequence of `SEQ_LEN` words for this feature so the input data has `GLOBAL_BATCH_SIZE * SEQ_LEN` tokens.

In general, models may have multiple tables and multiple features. Furthermore, multiple features may share a single embedding table.

More information on embedding specifications can be found in <project:../embedding.rst>.


```python
# TableSpec
table_spec = embedding_spec.TableSpec(
    vocabulary_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_SIZE,
    initializer=jax.nn.initializers.normal(),
    optimizer=embedding_spec.AdamOptimizerSpec(learning_rate=LEARNING_RATE),
    combiner='sum',
    name='shakespeare',
    max_ids_per_partition=64,
    max_unique_ids_per_partition=64,
)
pprint(table_spec)
```

    TableSpec(name='shakespeare',
              vocabulary_size=2048,
              embedding_dim=16,
              initializer=<function normal.<locals>.init at 0x104b9f6368e0>,
              optimizer=AdamOptimizerSpec(),
              combiner='sum',
              max_ids_per_partition=64,
              max_unique_ids_per_partition=64,
              suggested_coo_buffer_size_per_device=None,
              quantization_config=None,
              _stacked_table_spec=None,
              _setting_in_stack=TableSettingInStack(stack_name='shakespeare',
                                                    padded_vocab_size=2048,
                                                    padded_embedding_dim=16,
                                                    row_offset_in_shard=0,
                                                    shard_rotation=0))



```python
# FeatureSpec
feature_spec = embedding_spec.FeatureSpec(
    table_spec=table_spec,
    input_shape=(GLOBAL_BATCH_SIZE * SEQ_LEN, 1),
    output_shape=(
        GLOBAL_BATCH_SIZE * SEQ_LEN,
        EMBEDDING_SIZE,
    ),
    name='shakespeare_feature',
)
feature_specs = nn.FrozenDict({feature_spec.name: feature_spec})

# This call will take care of stacking features and other automatable
# configuration settings.
embedding.prepare_feature_specs_for_training(
    feature_specs,
    global_device_count=num_global_devices,
    num_sc_per_device=num_sc_per_device,
)
feature_specs = nn.FrozenDict(feature_specs)

for fs in feature_specs.values():
  pprint(fs)
```

    FeatureSpec(name='shakespeare_feature',
                table_spec=TableSpec(name='shakespeare',
                                     vocabulary_size=2048,
                                     embedding_dim=16,
                                     initializer=<function normal.<locals>.init at 0x104b9f6368e0>,
                                     optimizer=AdamOptimizerSpec(),
                                     combiner='sum',
                                     max_ids_per_partition=64,
                                     max_unique_ids_per_partition=64,
                                     suggested_coo_buffer_size_per_device=None,
                                     quantization_config=None,
                                     _stacked_table_spec=StackedTableSpec(stack_name='shakespeare',
                                                                          stack_vocab_size=2048,
                                                                          stack_embedding_dim=16,
                                                                          optimizer=AdamOptimizerSpec(),
                                                                          combiner='sum',
                                                                          total_sample_count=np.int64(4096),
                                                                          max_ids_per_partition=64,
                                                                          max_unique_ids_per_partition=64,
                                                                          suggested_coo_buffer_size_per_device=None,
                                                                          quantization_config=None),
                                     _setting_in_stack=TableSettingInStack(stack_name='shakespeare',
                                                                           padded_vocab_size=2048,
                                                                           padded_embedding_dim=16,
                                                                           row_offset_in_shard=0,
                                                                           shard_rotation=0)),
                input_shape=(4096, 1),
                output_shape=(4096, 16),
                _id_transformation=FeatureIdTransformation(row_offset=0,
                                                           col_offset=0,
                                                           col_shift=0))


# The Shakespeare Model

We're basing this on the example  model in [.../examples/models/shakespeare/flax_model.py](https://github.com/jax-ml/jax-tpu-embedding/blob/main/jax_tpu_embedding/sparsecore/examples/models/shakespeare/flax_model.py)

The Shakespeare model consists of an embedding layer and two dense layers. As input, the model takes preprocessed embedding inputs (of type `EmbeddingLookupInput`) which it uses to compute the embedding activations using the generic `SparseCoreEmbed` Flax layer. Since any word may be predicted, the output of the dense model is the same size as the embedding vocabulary.


```python
# Flax Model with SparseCore embedding layer.
class Model(nn.Module):
  """Shakespeare model using embedding layer."""

  feature_specs: Nested[embedding_spec.FeatureSpec]
  global_batch_size: int
  vocab_size: int
  seq_len: int
  embedding_size: int
  feature_name: str = 'shakespeare_feature'
  mesh: jax.sharding.Mesh | None = None
  sharding_axis: str = 'sparsecore_sharding'

  def add_sharding_constraint(self, x: jax.Array, names: tuple[str | None]):
    # Add a sharding constraint to the array.
    #
    # Add a sharding constraint to the array to ensure that the sharding
    # information is not lost during compilation. This may not be necessary but
    # it helps SPMD and ensures that the sharding information is as expected.
    #
    # Args:
    #   x: The array to add the sharding constraint to.
    #   names: The mesh axes for the partition spec.
    #
    # Returns:
    #   The array with the sharding constraint added.
    return jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*names)
        ),
    )

  @nn.compact
  def __call__(self, embedding_lookup_inputs: embedding.PreprocessedInput):
    # Run the embedding layer.
    x = embed.SparseCoreEmbed(
        feature_specs=self.feature_specs,
        mesh=self.mesh,
        sharding_axis=self.sharding_axis,
    )(embedding_lookup_inputs)

    # Unpack the activations.
    x = x[self.feature_name]
    x = jnp.reshape(x, (self.global_batch_size, -1))
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    # Apply the dense portion of the model.
    x = nn.Dense(self.embedding_size)(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))
    x = nn.Dense(self.vocab_size)(x)
    x = self.add_sharding_constraint(x, (self.sharding_axis,))

    return x


model = Model(
    feature_specs=feature_specs,
    global_batch_size=GLOBAL_BATCH_SIZE,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
    embedding_size=EMBEDDING_SIZE,
    mesh=global_mesh,
    sharding_axis=_SHARDING_AXIS,
)
```

# Model Initialization

Here, we initialize the model. We start by creating a zero-array for the embedding activations which are then used as inputs to initialize the dense model. We also initialize the emebdding tables with the initialization functions specified in the `TableSpec`s.


```python
# Process an input batch
def process_inputs(
    feature_specs: Nested[embedding_spec.FeatureSpec],
    batch_number: int,
    feature_batch: embedding.ArrayLike,
    global_mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.NamedSharding,
    num_sc_per_device: int,
) -> tuple[embedding.PreprocessedInput, embedding.SparseDenseMatmulInputStats]:
  """Preprocess a Shakespeare batch into PreprocessedInput and stats.

  Args:
    feature_specs: The feature specs.
    batch_number: The batch number.
    feature_batch: The feature batch.
    global_mesh: The global mesh.
    data_sharding: The NamedSharding for the data.
    num_sc_per_device: The number of sparse cores per device.

  Returns:
    A tuple of PreprocessedInput and SparseDenseMatmulInputStats.
  """
  features = np.reshape(feature_batch, (-1, 1))
  feature_weights = np.ones(features.shape, dtype=np.float32)

  # Pack the features into a tree structure.
  feature_structure = jax.tree.structure(feature_specs)
  features = jax.tree_util.tree_unflatten(feature_structure, [features])
  feature_weights = jax.tree_util.tree_unflatten(
      feature_structure, [feature_weights]
  )
  processed_inputs, stats = embedding.preprocess_sparse_dense_matmul_input(
      features,
      feature_weights,
      feature_specs,
      local_device_count=global_mesh.local_mesh.size,
      global_device_count=global_mesh.size,
      num_sc_per_device=num_sc_per_device,
      sharding_strategy='MOD',
      batch_number=batch_number,
  )
  processed_inputs = jax.tree.map(
      lambda x: jax.make_array_from_process_local_data(data_sharding, x),
      processed_inputs,
  )
  return processed_inputs, stats
```


```python
# Preload the model data
word_ids = shakespeare_data.load_shakespeare(VOCAB_SIZE)
feature_batches, label_batches = shakespeare_data.word_id_batches(
    word_ids,
    NUM_STEPS,
    GLOBAL_BATCH_SIZE,
    SEQ_LEN,
    NUM_TABLES,
)
feature_batches = feature_batches['words_0']

# Note: The input processor expects 2-d lookups, so we scale-up the batch
# size and reshape the results.

print(f'feature_batches len = {len(feature_batches)}')
print(f'feature_batches[0] shape = {feature_batches[0].shape}')
print(f'label_batches len = {len(label_batches)}')
print(f'label_batches[0] shape = {label_batches[0].shape}')
```

    feature_batches len = 100
    feature_batches[0] shape = (256, 16, 1)
    label_batches len = 100
    label_batches[0] shape = (256,)


## Extracting the model sharding

In order to jit the model for initialization and training, we need to provide the proper sharding specification for the model and it's nested layers. In the following, we use `jax.eval_shape()` and `nn.get_sharding()` to programmatically discover this sharding.


```python
# Initialize the model and training state.
first_model_input, unused_stats = process_inputs(
    feature_specs,
    -1,
    feature_batches[0],
    global_mesh,
    data_sharding,
    num_sc_per_device,
)


def init_fn(rng, data, model):
  params = model.init(rng, data)
  return params


# Create an abstract closure to wrap the function before feeding it in
# because `jax.eval_shape` only takes pytrees as arguments.
print('Evaluating model shape')
abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=model),
    jax.random.key(42),
    first_model_input,
)
# This `params_sharding` has the same pytree structure as `params`, the output
# of the `init_fn`.
print('Getting the params sharding')
params_sharding = nn.get_sharding(abstract_variables, global_mesh)
rng_sharding = NamedSharding(global_mesh, P())

print('Jitting init_fn')
jit_init_fn = jax.jit(
    init_fn,
    static_argnums=(2,),
    in_shardings=(
        rng_sharding,
        data_sharding,
    ),
    out_shardings=params_sharding,
)

print('Running the model initialization via init_fn')
params = jit_init_fn(jax.random.key(42), first_model_input, model)

# Create optimizer.
tx = embed_optimizer.create_optimizer_for_sc_model(
    params,
    optax.adam(learning_rate=LEARNING_RATE),
)
opt_state_sharding = NamedSharding(global_mesh, P())
opt_state = tx.init(params)
```

    Evaluating model shape
    Getting the params sharding
    Jitting init_fn
    Running the model initialization via init_fn


# The Training Step Function

The training step is mostly a call to `model.apply()` and a loss calculation. All the details of the embedding lookup and weight update are handled by the `SparseCoreEmbed` Flax layer.

We can use buffer donation on the `params` argument because the values are updated internally and the input `params` aren't used after calling the training step function.



```python
@functools.partial(
    jax.jit,
    in_shardings=(
        params_sharding,
        data_sharding,
        data_sharding,
        opt_state_sharding,
    ),
    out_shardings=(
        params_sharding,
        opt_state_sharding,
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
  train_step_fn = jax.value_and_grad(forward_pass, has_aux=True, allow_int=True)

  (loss_val, unused_logits), grads = train_step_fn(
      params, embedding_lookup_inputs, labels
  )

  updates, opt_state = tx.update(grads, opt_state)
  params = embed_optimizer.apply_updates_for_sc_model(params, updates)

  return params, opt_state, loss_val
```

# FDO Configuration

FDO (Feedback Directed Optimization) is an optional feature where we use
batch statistics to update table limits (`max_ids_per_partition`, etc.). The FDO
client is a generic interface and clients can either use provided
implementations or create their own that intergrates with their infrastructure.
When the FDO stats are used to update the limits on the `feature_specs`,
`jax.jit` triggers a recompilation.

For our flax model, updating the `FeatureSpec`s is as simple as updating the member variable on the `Model` object.

In this example, we use a simple file-based implementation. Each host writes its
own stats on `publish()` and stats from all hosts are merged during `load()`.


```python
out_path = os.path.join(FDO_DIR, 'fdo_dump')
os.makedirs(out_path, exist_ok=True)
print(f'FDO storage path: {out_path}')
fdo_client = file_fdo_client.NPZFileFDOClient(out_path)
```

    FDO storage path: /tmp/fdo_dump


# The Training Loop

The training loop is a simple Python `for`-loop. For each step, we preprocess the input batch data (on the host / CPU). In doing so, we capture the batch statistics which are used by FDO. Then we take a training step using the jitted `train_step_fn`. We finish each step by processing the training metrics and FDO stats.


```python
step = 0
# Track these separately for testing.
test_loss = None
test_step_count = 0
for features, labels in zip(feature_batches, label_batches):
  step += 1
  test_step_count += 1
  ##############################################################################
  # SC input processing.
  ##############################################################################
  # These are currently global batches so each task needs to offset into
  # the data for it's local slice.
  labels = labels[
      process_id * local_batch_size : (process_id + 1) * local_batch_size
  ]
  labels = jax.make_array_from_process_local_data(data_sharding, labels)

  # Each input preprocessing processes the current process's slice of the
  # global batch.
  features = features[
      process_id * local_batch_size : (process_id + 1) * local_batch_size
  ]

  model_inputs, step_stats = process_inputs(
      feature_specs,
      step,
      features,
      global_mesh,
      data_sharding,
      num_sc_per_device,
  )
  fdo_client.record(step_stats)

  ##############################################################################
  # Run the model.
  ##############################################################################
  # We capture the logging here so you can see when XLA compilation is
  # triggered.
  params, opt_state, loss_val = train_step(
      params, model_inputs, labels, opt_state
  )

  if step % LOG_FREQUENCY == 0:
    print(
        f'Step {step}: loss={loss_val}, params is'
        f' {jax.tree.map(jnp.sum, params)}'
    )
  test_loss = loss_val
  if step % FDO_FREQUENCY == 0:
    print(f'Refreshing FDO stats after step {step}')
    fdo_client.publish()
    loaded_stats = fdo_client.load()
    embedding.update_preprocessing_parameters(
        model.feature_specs, loaded_stats, num_sc_per_device
    )
```


```python
# Print the loss and step count for the colab test.
print(f'test_loss = {test_loss:.4f}')
print(f'test_step_count = {test_step_count}')
```

    test_loss = 0.0079
    test_step_count = 100

