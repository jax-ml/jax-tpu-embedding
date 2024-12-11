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
    default_visibility = [
        "//jax_tpu_embedding/sparsecore:__subpackages__",
        "//smartass/brain:__subpackages__",
    ],
)

pytype_strict_library(
    name = "embedding_spec",
    srcs = [
        "embedding_spec.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_sgd",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
    ],
)

pytype_strict_library(
    name = "embedding_utils",
    srcs = ["embedding_utils.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib/nn:__subpackages__"],
    deps = [
        ":embedding_spec",
        ":table_stacking",
        pypi_requirement("absl/logging"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("numpy"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_library(
    name = "embedding",
    srcs = ["embedding.py"],
    deps = [
        ":embedding_spec",
        ":embedding_utils",
        ":table_stacking",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_cc",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr_with_mini_batching",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        pypi_requirement("absl/logging"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("numpy"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_library(
    name = "embedding_with_mini_batching",
    srcs = ["embedding_with_mini_batching.py"],
    deps = [
        ":embedding",
        ":embedding_spec",
        ":embedding_utils",
        ":table_stacking",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_with_mini_batching_cc",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr_with_mini_batching",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad_with_mini_batching",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_sgd_with_mini_batching",
        pypi_requirement("absl/logging"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("jax/extend"),
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
    name = "embedding_pipelining_utils",
    srcs = [
        "embedding_pipelining_utils.py",
    ],
    deps = [
        ":embedding",
        ":embedding_spec",
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
    ],
)

pytype_strict_library(
    name = "nn",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":embedding",  # buildcleaner: keep
        ":embedding_pipelining_utils",  # buildcleaner: keep
        ":embedding_spec",  # buildcleaner: keep
        ":table_stacking",  # buildcleaner: keep
    ],
)
