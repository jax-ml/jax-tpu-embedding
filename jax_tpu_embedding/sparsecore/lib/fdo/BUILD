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
load("@rules_python//python:py_test.bzl", "py_test")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = EXTERNAL_USERS,
)

py_library(
    name = "fdo_client",
    srcs = ["fdo_client.py"],
    type_checking = True,
    deps = ["//jax_tpu_embedding/sparsecore/lib/nn:embedding"],
)

py_library(
    name = "file_fdo_client",
    srcs = ["file_fdo_client.py"],
    type_checking = True,
    deps = [
        ":fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("absl/logging"),
        pypi_requirement("etils/epath"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

py_test(
    name = "file_fdo_client_test",
    srcs = ["file_fdo_client_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    type_checking = True,
    deps = [
        ":file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("etils/epath"),
        pypi_requirement("numpy"),
    ],
)

py_library(
    name = "csv_file_fdo_client",
    srcs = ["csv_file_fdo_client.py"],
    type_checking = True,
    deps = [
        ":file_fdo_client",
        pypi_requirement("absl/logging"),
        pypi_requirement("numpy"),
    ],
)

py_test(
    name = "csv_file_fdo_client_test",
    srcs = ["csv_file_fdo_client_test.py"],
    type_checking = True,
    deps = [
        ":csv_file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("etils/epath"),
        pypi_requirement("numpy"),
    ],
)

# Library target.
py_library(
    name = "fdo",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":csv_file_fdo_client",  # buildcleaner: keep
        ":fdo_client",  # buildcleaner: keep
        ":file_fdo_client",  # buildcleaner: keep
    ],
)
