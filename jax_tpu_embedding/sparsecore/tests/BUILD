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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "tpu_py_strict_test")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

pytype_strict_contrib_test(
    name = "version_test",
    srcs = ["version_test.py"],
    data = ["//jax_tpu_embedding/sparsecore:version"],
    deps = [
        "//jax_tpu_embedding/sparsecore",
        pypi_requirement("absl/testing:absltest"),
    ],
)

tpu_py_strict_test(
    name = "jax_sc_shakespeare_tests",
    srcs = ["jax_sc_shakespeare_tests.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
    ],
)

tpu_py_strict_test(
    name = "jax_spmd_tc_with_sc_tests",
    srcs = ["jax_spmd_tc_with_sc_tests.py"],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
    ],
)
