# JAX TPU Embedding

[![Unittests](https://github.com/jax-ml/jax-tpu-embedding/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/jax-ml/jax-tpu-embedding/actions/workflows/build_and_test.yml)

JAX TPU Embedding is a library for JAX that provides an efficient, scalable API for large embedding tables on Google Cloud TPUs, leveraging the specialized SparseCore hardware. It is designed to integrate seamlessly into JAX and Flax workflows.

---

## Installation

You can install the library directly from its GitHub repository:

```bash
git clone https://github.com/jax-ml/jax-tpu-embedding.git
cd jax-tpu-embedding
chmod +x .tools/local_build_wheel.sh
.tools/local_build_wheel.sh
pip install ./dist/*.whl
```

### Development

To build and test the library from a local clone, you will need to install Bazel. You can find the required version in the `.bazelversion` file.

```bash
# Clone the repository
git clone [https://github.com/jax-ml/jax-tpu-embedding.git](https://github.com/jax-ml/jax-tpu-embedding.git)
cd jax-tpu-embedding

# Run all tests
bazel test //...
```

---

## Quick Start

Here's a quick example of how to use the high-level Flax API to define a model with an embedding layer.

### 1. Define Embedding Table and Feature Specifications

First, define the structure of your embedding table (`TableSpec`) and how your features map to it (`FeatureSpec`). These specifications tell the library how to allocate memory on the SparseCores and configure the lookup hardware.

```python
import jax
import jax.numpy as jnp
from jax_tpu_embedding.sparsecore.lib.nn import embedding_spec

# Example constants
BATCH_SIZE = 32
SEQ_LEN = 16
VOCAB_SIZE = 1024
EMBEDDING_DIM = 128

# Define the embedding table properties
table_spec = embedding_spec.TableSpec(
    vocabulary_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    name='word_embedding_table',
    optimizer=embedding_spec.SGDOptimizerSpec(learning_rate=0.05)
)

# Define a feature that uses this table
feature_spec = embedding_spec.FeatureSpec(
    table_spec=table_spec,
    input_shape=(BATCH_SIZE, SEQ_LEN), # Shape of the input IDs
    output_shape=(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM), # Desired output activation shape
    name='word_ids'
)

# Feature specs are passed as a PyTree (e.g., a dictionary)
feature_specs = {'word_ids': feature_spec}
```

### 2. Create a Flax Model

Use the `SparseCoreEmbed` layer within a standard Flax `nn.Module`. This layer handles all the communication with the SparseCores.

```python
from flax import linen as nn
from jax_tpu_embedding.sparsecore.lib.flax import embed

class ShakespeareModel(nn.Module):
    feature_specs: embed.Nested[embedding_spec.FeatureSpec]
    
    @nn.compact
    def __call__(self, embedding_inputs):
        # This layer performs the embedding lookup on SparseCores.
        # `embedding_inputs` is a dictionary of integer ID tensors.
        embedding_activations = embed.SparseCoreEmbed(
            feature_specs=self.feature_specs
        )(embedding_inputs)

        # The result is a dictionary of activations, matching the feature spec keys.
        x = embedding_activations['word_ids']

        # Flatten sequence and embedding dimensions for the dense layer
        x = x.reshape((x.shape[0], -1))

        # Add a dense layer (runs on TensorCores)
        x = nn.Dense(VOCAB_SIZE)(x)
        return x
```

### 3. Initialize and Run the Training Step

The `SparseCoreEmbed` layer separates its parameters (the embedding tables, which live on SparseCore HBM) from the rest of the model's parameters (which live on TensorCore HBM). You initialize them separately and pass them to the training step.

```python
# Create a mesh for device layout
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, axis_names=('data',))

# 1. Initialize the embedding tables (SparseCore parameters)
# This needs to be done under the mesh context
with mesh:
    embedding_params = embed.SparseCoreEmbed.create_embedding_variables(
        feature_specs, jax.random.PRNGKey(0)
    )

# 2. Initialize the dense model parts (TensorCore parameters)
model = ShakespeareModel(feature_specs=feature_specs)
dummy_inputs = {'word_ids': jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)}
dense_params = model.init(
    jax.random.PRNGKey(1), dummy_inputs
)['params']

# 3. Define and JIT-compile the training step
@jax.jit
def train_step(dense_params, embedding_params, features, labels):
    def loss_fn(params):
        # The 'embedding' collection is automatically handled by SparseCoreEmbed
        logits = model.apply({'params': params, 'embedding': embedding_params}, features)
        # A real implementation would use a proper loss function like cross-entropy
        return jnp.mean(jnp.square(logits - labels)) # Dummy loss for demonstration
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(dense_params)
    
    # Gradients for `embedding_params` are computed and applied on the
    # SparseCores automatically by the `SparseCoreEmbed` layer, using the
    # optimizer defined in the TableSpec.
    
    # Update dense parameters using your preferred optimizer
    new_dense_params = jax.tree.map(lambda p, g: p - 0.01 * g, dense_params, grads)
    
    return new_dense_params

# --- Example usage with dummy data ---
features = {'word_ids': jax.random.randint(
    jax.random.PRNGKey(2), (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)}
labels = jax.random.normal(jax.random.PRNGKey(3), (BATCH_SIZE, VOCAB_SIZE))

# Run one training step
new_dense_params = train_step(dense_params, embedding_params, features, labels)
```

---

## Key Concepts

-   **`TableSpec`**: Defines the properties of a single embedding table, including its shape (`vocabulary_size`, `embedding_dim`), initializer, and the optimizer (e.g., `SGDOptimizerSpec`) to be run on the SparseCores.
-   **`FeatureSpec`**: Describes a logical feature that maps to a `TableSpec`. It specifies the `input_shape` of the integer ID tensor and the desired `output_shape` of the resulting activation tensors. A single table can be used by multiple features.
-   **`SparseCoreEmbed`**: A Flax Linen layer that acts as the entry point for embedding lookups. It takes a dictionary of feature names to ID tensors and returns a dictionary of feature names to activation tensors. It manages the embedding table parameters separately from the dense model parameters.

---

## Running the Examples

The repository includes a complete Shakespeare next-word-prediction model. You can run it using Bazel.

To run the tests for the example:
```bash
# Make sure you are at the root of the repository
bazel test //jax_tpu_embedding/sparsecore/examples/shakespeare/...
```
This will build and run both the `pmap` and `jit` + `shard_map` versions of the model.

---

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

---

## License

This project is licensed under the terms of the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.
