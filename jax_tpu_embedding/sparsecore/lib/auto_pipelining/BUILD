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
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:internal"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    type_checking = True,
    deps = [
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
    ],
)

py_library(
    name = "decompose",
    srcs = ["decompose.py"],
    type_checking = True,
    deps = [
        ":preprocess",
        ":utils",
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
    ],
)

py_library(
    name = "preprocess",
    srcs = ["preprocess.py"],
    type_checking = True,
    deps = [
        ":utils",
        pypi_requirement("jax/extend"),
    ],
)

py_library(
    name = "auto_pipelining",
    srcs = ["auto_pipelining.py"],
    type_checking = True,
    deps = [
        ":decompose",
        pypi_requirement("etils/epath"),
        pypi_requirement("jax"),
        pypi_requirement("jax/extend"),
    ],
)
