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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "CORE_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:internal"],
)

py_library(
    name = "sparse_dense_matmul_csr",
    srcs = [
        "sparse_dense_matmul_csr.py",
    ],
    type_checking = True,
    visibility = CORE_USERS,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_activation_unstack",
    srcs = ["sparse_dense_matmul_activation_unstack.py"],
    type_checking = True,
    deps = [
        ":utils",
        pypi_requirement("jax"),
        pypi_requirement("jax/experimental:layout"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
    ],
)

py_library(
    name = "sparse_dense_matmul_gradient_stack",
    srcs = ["sparse_dense_matmul_gradient_stack.py"],
    type_checking = True,
    deps = [
        ":utils",
        pypi_requirement("jax"),
        pypi_requirement("jax/experimental:layout"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
    ],
)

py_library(
    name = "local_sparse_dense_matmul",
    srcs = ["local_sparse_dense_matmul.py"],
    type_checking = True,
    deps = [
        ":utils",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_sgd",
    srcs = [
        "sparse_dense_matmul_grad_with_sgd.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_adagrad",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_adagrad_momentum",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad_momentum.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_laprop",
    srcs = ["sparse_dense_matmul_grad_with_laprop.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_adam",
    srcs = [
        "sparse_dense_matmul_grad_with_adam.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_ftrl",
    srcs = [
        "sparse_dense_matmul_grad_with_ftrl.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "sparse_dense_matmul_grad_with_f2a",
    srcs = ["sparse_dense_matmul_grad_with_f2a.py"],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "optimizers_computation",
    srcs = [
        "optimizers_computation.py",
    ],
    type_checking = True,
    deps = [
        pypi_requirement("jax"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
    ],
)

py_library(
    name = "sparse_dense_matmul_optimizer_grad",
    srcs = [
        "sparse_dense_matmul_optimizer_grad.py",
    ],
    type_checking = True,
    deps = [
        ":utils",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)

# Library target.
py_library(
    name = "primitives",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = ["//jax_tpu_embedding/sparsecore/lib/core:__pkg__"],
    deps = [
        ":optimizers_computation",  # buildcleaner: keep
        ":sparse_dense_matmul_csr",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_adagrad",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_adagrad_momentum",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_f2a",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_ftrl",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_laprop",  # buildcleaner: keep
        ":sparse_dense_matmul_grad_with_sgd",  # buildcleaner: keep
        ":sparse_dense_matmul_gradient_stack",  # buildcleaner: keep
        ":sparse_dense_matmul_optimizer_grad",  # buildcleaner: keep
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    type_checking = True,
    deps = [
        pypi_requirement("jax"),
        pypi_requirement("jax/extend/mlir:ir"),
        pypi_requirement("jax/extend/mlir/dialects:func_dialect"),
        pypi_requirement("jax/extend/mlir/dialects:stablehlo_dialect"),
        pypi_requirement("numpy"),
    ],
)
