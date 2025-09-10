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
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "CORE_USERS")
load("//third_party/bazel/python:pybind11.bzl", "pybind_extension", "pybind_library")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = CORE_USERS,
)

cc_library(
    name = "input_preprocessing_threads",
    srcs = ["input_preprocessing_threads.cc"],
    hdrs = ["input_preprocessing_threads.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_test(
    name = "input_preprocessing_threads_test",
    srcs = ["input_preprocessing_threads_test.cc"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":input_preprocessing_threads",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "input_preprocessing_util",
    srcs = ["input_preprocessing_util.cc"],
    hdrs = ["input_preprocessing_util.h"],
    deps = [
        ":all_reduce_interface",
        ":coo_format",
        ":partitioned_coo_tensors",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "abstract_input_batch",
    hdrs = ["abstract_input_batch.h"],
    deps = [
        ":input_preprocessing_util",
        "@com_google_absl//absl/base:core_headers",
    ],
)

pybind_library(
    name = "numpy_input_batch",
    srcs = ["numpy_input_batch.cc"],
    hdrs = ["numpy_input_batch.h"],
    deps = [
        ":abstract_input_batch",
        ":input_preprocessing_util",
        ":process_coo_tensors",
        "@com_google_absl//absl/log:check",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

pybind_library(
    name = "sparse_coo_input_batch",
    srcs = ["sparse_coo_input_batch.cc"],
    hdrs = ["sparse_coo_input_batch.h"],
    deps = [
        ":abstract_input_batch",
        ":input_preprocessing_util",
        ":process_coo_tensors",
        ":sparse_csr_input_stream",
        ":unity_weights_stream",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_test(
    name = "input_preprocessing_util_test",
    srcs = ["input_preprocessing_util_test.cc"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":coo_format",
        ":input_preprocessing_util",
        ":partitioned_coo_tensors",
        ":sort_and_group_coo_tensors_impl",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:span",
        "@com_google_fuzztest//fuzztest",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "input_preprocessing_core",
    srcs = ["input_preprocessing.cc"],
    hdrs = ["input_preprocessing.h"],
    deps = [
        ":abstract_input_batch",
        ":coo_format",
        ":input_preprocessing_threads",
        ":input_preprocessing_util",
        ":minibatching_sync_service",
        ":partitioned_coo_tensors",
        ":sort_and_group_coo_tensors_impl",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_test(
    name = "input_preprocessing_test_cc",
    srcs = ["input_preprocessing_test.cc"],
    deps = [
        ":abstract_input_batch",
        ":all_reduce_interface",
        ":coo_format",
        ":input_preprocessing_core",
        ":input_preprocessing_util",
        ":minibatching_test_utils",
        ":ragged_tensor_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc:minibatching_node",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

pybind_extension(
    name = "pybind_input_preprocessing",
    srcs = ["pybind_input_preprocessing.cc"],
    deps = [
        ":abstract_input_batch",
        ":all_reduce_interface",
        ":input_preprocessing_core",
        ":input_preprocessing_util",
        ":numpy_input_batch",
        ":sparse_coo_input_batch",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc:minibatching_node",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@pybind11//:pybind11_eigen",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

pytype_strict_library(
    name = "input_preprocessing",
    srcs = [
        "input_preprocessing.py",
    ],
    deps = [
        ":constants",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = [
        "constants.py",
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
        ":constants",
        ":input_preprocessing",
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
        ":constants",
        ":pybind_input_preprocessing",
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
        ":pybind_input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("numpy"),
    ],
)

# Library target.
pytype_strict_library(
    name = "core",
    srcs = ["__init__.py"],
    # C++ dependencies must go in "data".
    data = [
        ":input_preprocessing_core",  # buildcleaner: keep
        ":input_preprocessing_threads",  # buildcleaner: keep
        ":input_preprocessing_util",  # buildcleaner: keep
    ],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":constants",  # buildcleaner: keep
        ":input_preprocessing",  # buildcleaner: keep
        ":pybind_input_preprocessing",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/core/primitives",  # buildcleaner: keep
    ],
)

cc_library(
    name = "ragged_tensor_input_batch",
    hdrs = ["ragged_tensor_input_batch.h"],
    deps = [
        ":abstract_input_batch",
        ":input_preprocessing_util",
        ":process_coo_tensors",
        ":sparse_csr_input_stream",
        ":unity_weights_stream",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "unity_weights_stream",
    hdrs = ["unity_weights_stream_impl.h"],
)

cc_library(
    name = "sparse_csr_input_stream",
    hdrs = ["sparse_csr_input_stream_impl.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

cc_library(
    name = "process_coo_tensors",
    hdrs = ["process_coo_tensors_impl.h"],
    deps = [
        ":abstract_input_batch",
        ":input_preprocessing_util",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "partitioned_coo_tensors",
    hdrs = ["partitioned_coo_tensors.h"],
    deps = [
        ":coo_format",
        ":minibatching_splits_impl",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "coo_format",
    hdrs = ["coo_format.h"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "minibatching_splits_impl",
    hdrs = ["minibatching_splits_impl.h"],
    deps = [
        ":coo_format",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "sort_and_group_coo_tensors_impl",
    hdrs = ["sort_and_group_coo_tensors_impl.h"],
    deps = [
        ":coo_format",
        ":input_preprocessing_util",
        ":minibatching_splits_impl",
        ":partitioned_coo_tensors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
        "@highway//hwy/contrib/sort:vqsort",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "minibatching_sync_service",
    hdrs = ["minibatching_sync_service.h"],
    deps = [
        ":all_reduce_interface",
        ":input_preprocessing_util",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_test(
    name = "minibatching_sync_service_test",
    srcs = ["minibatching_sync_service_test.cc"],
    deps = [
        ":input_preprocessing_util",
        ":minibatching_sync_service",
        ":minibatching_test_utils",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc:minibatching_node",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "all_reduce_interface",
    hdrs = ["all_reduce_interface.h"],
    deps = ["@com_google_absl//absl/status:statusor"],
)

cc_library(
    name = "minibatching_test_utils",
    testonly = True,
    hdrs = ["minibatching_test_utils.h"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/grpc:minibatching_node",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/platform:test",
    ],
)

cc_test(
    name = "ragged_tensor_input_batch_test",
    srcs = ["ragged_tensor_input_batch_test.cc"],
    deps = [
        ":input_preprocessing_util",
        ":ragged_tensor_input_batch",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "partitioned_coo_tensors_test",
    srcs = ["partitioned_coo_tensors_test.cc"],
    deps = [
        ":coo_format",
        ":partitioned_coo_tensors",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "minibatching_splits_test",
    srcs = ["minibatching_splits_test.cc"],
    deps = [
        ":minibatching_splits_impl",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
