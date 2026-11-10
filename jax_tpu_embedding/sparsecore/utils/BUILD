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
    name = "utils",
    srcs = [
        "__init__.py",
        "utils.py",
    ],
    type_checking = True,
    deps = [
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("jax/experimental:layout"),
        pypi_requirement("jax/experimental:pallas_tpu"),
    ],
)

py_test(
    name = "utils_test",
    timeout = "short",
    srcs = ["utils_test.py"],
    type_checking = True,
    deps = [
        ":utils",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)
