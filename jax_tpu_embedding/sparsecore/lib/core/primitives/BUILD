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
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:internal"],
)

pytype_strict_library(
    name = "sparse_dense_matmul_csr",
    srcs = [
        "sparse_dense_matmul_csr.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_csr_with_mini_batching",
    srcs = [
        "sparse_dense_matmul_csr_with_mini_batching.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_sgd_with_mini_batching",
    srcs = [
        "sparse_dense_matmul_grad_with_sgd_with_mini_batching.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_adagrad_with_mini_batching",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad_with_mini_batching.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_sgd",
    srcs = [
        "sparse_dense_matmul_grad_with_sgd.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_adagrad",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_laprop",
    srcs = ["sparse_dense_matmul_grad_with_laprop.py"],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_adam",
    srcs = [
        "sparse_dense_matmul_grad_with_adam.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_grad_with_ftrl",
    srcs = [
        "sparse_dense_matmul_grad_with_ftrl.py",
    ],
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "optimizers_computation",
    srcs = [
        "optimizers_computation.py",
    ],
    deps = [
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "sparse_dense_matmul_optimizer_grad",
    srcs = [
        "sparse_dense_matmul_optimizer_grad.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/_src/lib"),
        pypi_requirement("jax/extend"),
        pypi_requirement("numpy"),
    ],
)

# Library target.
pytype_strict_library(
    name = "primitives",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib/core:__pkg__"],
    deps = [
        ":optimizers_computation",  # buildcleaner: keep
        ":sparse_dense_matmul_csr",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_adagrad",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_ftrl",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_laprop",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_sgd",  # buildcleaner: keep
        ":sparse_dense_matmul_optimizer_grad",  # buildcleaner: keep
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [pypi_requirement("numpy")],
)
