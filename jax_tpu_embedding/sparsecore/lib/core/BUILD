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
load("//third_party/bazel/python:pybind11.bzl", "pybind_extension")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:__subpackages__"],
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@highway//:hwy",
        "@highway//hwy/contrib/sort:vqsort",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_test(
    name = "input_preprocessing_util_test",
    srcs = ["input_preprocessing_util_test.cc"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":input_preprocessing_util",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "input_preprocessing_cc",
    srcs = ["input_preprocessing.cc"],
    deps = [
        ":input_preprocessing_threads",
        ":input_preprocessing_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

pytype_strict_library(
    name = "input_preprocessing",
    srcs = [
        "input_preprocessing.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/logging"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = [
        "constants.py",
    ],
    srcs_version = "PY3",
)

pytype_strict_contrib_test(
    name = "input_preprocessing_test",
    srcs = [
        "input_preprocessing_test.py",
    ],
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_dump_to=sponge",
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
    name = "input_preprocessing_cc_test",
    srcs = [
        "input_preprocessing_cc_test.py",
    ],
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    deps = [
        ":constants",
        ":input_preprocessing_cc",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
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
        ":input_preprocessing_threads",  # buildcleaner: keep
        ":input_preprocessing_util",  # buildcleaner: keep
    ],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":constants",  # buildcleaner: keep
        ":input_preprocessing",  # buildcleaner: keep
        ":input_preprocessing_cc",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/core/primitives",  # buildcleaner: keep
    ],
)
