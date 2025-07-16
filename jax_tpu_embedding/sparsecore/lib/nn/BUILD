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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = EXTERNAL_USERS,
)

pytype_strict_library(
    name = "embedding_spec",
    srcs = [
        "embedding_spec.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad_momentum",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adam",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_ftrl",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_laprop",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_sgd",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
    ],
)

pytype_strict_library(
    name = "embedding",
    srcs = ["embedding.py"],
    deps = [
        ":embedding_spec",
        ":table_stacking",
        "//jax_tpu_embedding/sparsecore/lib/core:pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("absl/logging"),
        pypi_requirement("einops"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("numpy"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_library(
    name = "table_stacking",
    srcs = ["table_stacking.py"],
    deps = [
        ":embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        pypi_requirement("absl/logging"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_library(
    name = "nn",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":embedding",  # buildcleaner: keep
        ":embedding_spec",  # buildcleaner: keep
        ":table_stacking",  # buildcleaner: keep
    ],
)
