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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS", "tpu_py_strict_test")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_binary", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

pytype_strict_contrib_test(
    name = "preprocess_sparse_dense_matmul_input_test",
    srcs = [
        "preprocess_sparse_dense_matmul_input_test.py",
    ],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_binary(
    name = "preprocess_input_benchmarks",
    srcs = ["preprocess_input_benchmarks.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("google_benchmark"),
        pypi_requirement("numpy"),
    ],
)

# Verify that the benchmarks run for small values of (#features, #samples).
# This can also be used with benchy:
# benchy //jax_tpu_embedding/sparsecore/lib/nn/tests:preprocess_input_benchmarks_test
# (Uses default value for num_features and num_samples ignoring the args below.
pytype_strict_contrib_test(
    name = "preprocess_input_benchmarks_test",
    srcs = ["preprocess_input_benchmarks.py"],
    args = [
        "--num_features=5",
        "--num_samples=16",
    ],
    env = {"JAX_PLATFORMS": "cpu"},
    main = "preprocess_input_benchmarks.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("google_benchmark"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.py"],
    visibility = EXTERNAL_USERS,
    deps = [
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "test_utils_test",
    srcs = ["test_utils_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":test_utils",
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "tpu_sparse_dense_matmul_test",
    srcs = [
        "tpu_sparse_dense_matmul_test.py",
    ],
    args = [
        # TPU chip configuration,
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    main = "tpu_sparse_dense_matmul_test.py",
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_contrib_test(
    name = "tpu_sparse_dense_matmul_grad_test",
    srcs = [
        "tpu_sparse_dense_matmul_grad_test.py",
    ],
    args = [
        # TPU chip configuration,
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
        "JAX_TRACEBACK_FILTERING": "off",
    },
    main = "tpu_sparse_dense_matmul_grad_test.py",
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/nn:table_stacking",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),  # buildcleaner: keep
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

tpu_py_strict_test(
    name = "embedding_test",
    srcs = [
        "embedding_test.py",
    ],
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("google/protobuf"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

tpu_py_strict_test(
    name = "table_stacking_test",
    srcs = [
        "table_stacking_test.py",
    ],
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/nn:table_stacking",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("jax"),
    ],
)

pytype_strict_contrib_test(
    name = "embedding_spec_test",
    srcs = ["embedding_spec_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad_momentum",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_ftrl",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_laprop",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("jax"),
        pypi_requirement("optax/schedules"),
    ],
)
