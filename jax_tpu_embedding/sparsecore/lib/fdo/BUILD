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
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:internal"],
)

pytype_strict_library(
    name = "fdo_client",
    srcs = ["fdo_client.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "file_fdo_client",
    srcs = ["file_fdo_client.py"],
    deps = [
        ":fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("absl/logging"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "file_fdo_client_test",
    srcs = ["file_fdo_client_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_library(
    name = "fdo_utils",
    srcs = ["fdo_utils.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

# Library target.
pytype_strict_library(
    name = "fdo",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        ":fdo_client",  # buildcleaner: keep
        ":fdo_utils",  # buildcleaner:keep
        ":file_fdo_client",  # buildcleaner: keep
    ],
)
