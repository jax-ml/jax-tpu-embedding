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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "tpu_py_strict_test")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

tpu_py_strict_test(
    name = "embed_test",
    srcs = ["embed_test.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/flax:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/nn/tests:test_utils",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("einops"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

tpu_py_strict_test(
    name = "autograd_test",
    srcs = [
        "autograd_test.py",
    ],
    env = {
        # "XLA_FLAGS": "--xla_dump_to=sponge",
        "JAX_TRACEBACK_FILTERING": "off",
    },
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:flax_model",
        "//jax_tpu_embedding/sparsecore/lib/flax:embed",
        "//jax_tpu_embedding/sparsecore/lib/flax:embed_optimizer",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
    ],
)

tpu_py_strict_test(
    name = "embed_optimizer_test",
    srcs = ["embed_optimizer_test.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/flax:embed_optimizer",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("optax"),
    ],
)
