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
"""Tests for jax2tf export with TPU."""

from __future__ import annotations

import logging
import os

import jax
from jax import config
import jax.numpy as jnp
import tensorflow as tf

from jax.experimental import jax2tf
from tensorflow.core.protobuf import saved_model_pb2

# Set JAX export calling convention (for compatibility with older versions of XLA)
os.environ["JAX_EXPORT_CALLING_CONVENTION_VERSION"] = "9"

# Suppress noisy TF/CUDA/AutoGraph warnings for cleaner output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("AUTOGRAPH_VERBOSITY", "0")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def _inspect_saved_model_xla_versions(saved_model_dir: str) -> list[int]:
  """Parse saved_model.pb and return XlaCallModule version(s) found."""
  pb_path = os.path.join(saved_model_dir, "saved_model.pb")
  if not os.path.exists(pb_path):
    return []

  sm = saved_model_pb2.SavedModel()
  with open(pb_path, "rb") as f:
    sm.ParseFromString(f.read())

  versions = set()
  for mg in sm.meta_graphs:
    for fn in mg.graph_def.library.function:
      for node in fn.node_def:
        if node.op == "XlaCallModule" and "version" in node.attr:
          versions.add(node.attr["version"].i)
  return sorted(versions)


class Jax2TfTpuTest(tf.test.TestCase):

  def test_jax2tf_export_xla_version(self):
    """Tests jax2tf export with XLA version 9."""
    print("tensorflow.__version__", tf.__version__)
    print("jax.__version__", jax.__version__)
    print(
        "JAX export calling convention version:",
        config.jax_export_calling_convention_version,
    )

    model_dir = self.create_tempdir().full_path
    # ── tiny model: y = x @ W + b ──
    W = jnp.ones((4, 2), dtype=jnp.float32)
    b = jnp.zeros((2,), dtype=jnp.float32)

    def predict(x):
      return x @ W + b

    # jax2tf with native serialization — the standard export path for
    # serving JAX models via TFServing.
    tf_predict = jax2tf.convert(
        predict,
        native_serialization_platforms=("tpu",),
        polymorphic_shapes=("batch, 4",),
    )

    class ExportModule(tf.Module):

      @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32)])
      def predict(self, x):
        return tf_predict(x)

    module = ExportModule()
    module.predict(tf.ones([1, 4]))
    tf.saved_model.save(module, model_dir)
    versions = _inspect_saved_model_xla_versions(model_dir)
    self.assertIn(9, versions)

    # Load model and run inference.
    model = tf.saved_model.load(model_dir)
    out = model.predict(tf.ones([1, 4]))
    self.assertAllClose(out, [[4.0, 4.0]])


if __name__ == "__main__":
  tf.test.main()
