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
    default_visibility = ["//jax_tpu_embedding/sparsecore:__subpackages__"],
)

pytype_strict_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        pypi_requirement("jax"),
    ],
)

pytype_strict_contrib_test(
    name = "checkpoint_utils_test",
    srcs = ["checkpoint_utils_test.py"],
    env = {"JAX_PLATFORMS": "cpu"},
    deps = [
        ":checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("jax"),
    ],
)

# Library target.
pytype_strict_library(
    name = "checkpointing",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
    deps = [
        "checkpoint_utils",  # buildcleaner: keep
    ],
)
