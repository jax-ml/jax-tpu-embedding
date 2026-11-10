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
load("@rules_python//python:py_library.bzl", "py_library")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = EXTERNAL_USERS,
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
    ],
    type_checking = True,
    deps = [
        pypi_requirement("absl/logging"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "model",
    srcs = [
        "model.py",
    ],
    type_checking = True,
    deps = [
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("optax"),
    ],
)

py_library(
    name = "flax_model",
    srcs = [
        "flax_model.py",
    ],
    type_checking = True,
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/flax/linen:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
    ],
)

py_library(
    name = "flax_nnx_model",
    srcs = [
        "flax_nnx_model.py",
    ],
    type_checking = True,
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("flax/nnx:v1"),
        pypi_requirement("jax"),
    ],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    type_checking = True,
    deps = [
        ":dataset",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)
