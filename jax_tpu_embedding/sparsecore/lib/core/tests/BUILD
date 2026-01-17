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
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "CORE_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = CORE_USERS,
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [pypi_requirement("numpy")],
)

cc_test(
    name = "input_preprocessing_threads_test",
    srcs = ["input_preprocessing_threads_test.cc"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_threads",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
    ],
)

cc_test(
    name = "input_preprocessing_util_test",
    srcs = ["input_preprocessing_util_test.cc"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:coo_format",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_util",
        "//jax_tpu_embedding/sparsecore/lib/core:partitioned_coo_tensors",
        "//jax_tpu_embedding/sparsecore/lib/core:sort_and_group_coo_tensors_impl",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "input_preprocessing_test_cc",
    srcs = ["input_preprocessing_test.cc"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:abstract_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core:all_reduce_interface",
        "//jax_tpu_embedding/sparsecore/lib/core:coo_format",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_core",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_util",
        "//jax_tpu_embedding/sparsecore/lib/core:minibatching_test_utils",
        "//jax_tpu_embedding/sparsecore/lib/core:ragged_tensor_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc:minibatching_node",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:statusor",
        "@xla//xla:util",
    ],
)

pytype_strict_contrib_test(
    name = "input_preprocessing_test",
    srcs = [
        "input_preprocessing_test.py",
    ],
    env = {
        "JAX_PLATFORMS": "cpu",
    },
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "pybind_input_preprocessing_test",
    srcs = [
        "pybind_input_preprocessing_test.py",
    ],
    env = {
        "JAX_PLATFORMS": "cpu",
    },
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/core",
        "//jax_tpu_embedding/sparsecore/lib/core:constants",
        "//jax_tpu_embedding/sparsecore/lib/core:pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("numpy"),
        pypi_requirement("portpicker"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_coo_input_preprocessing_test",
    srcs = [
        "sparse_coo_input_preprocessing_test.py",
    ],
    shard_count = 10,
    deps = [
        ":test_utils",
        "//jax_tpu_embedding/sparsecore/lib/core:pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("numpy"),
    ],
)

cc_test(
    name = "ragged_tensor_input_batch_test",
    srcs = ["ragged_tensor_input_batch_test.cc"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:coo_format",
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing_util",
        "//jax_tpu_embedding/sparsecore/lib/core:ragged_tensor_input_batch",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "partitioned_coo_tensors_test",
    srcs = ["partitioned_coo_tensors_test.cc"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:coo_format",
        "//jax_tpu_embedding/sparsecore/lib/core:partitioned_coo_tensors",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minibatching_splits_test",
    srcs = ["minibatching_splits_test.cc"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:minibatching_splits_impl",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
