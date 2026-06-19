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
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load(
    "//third_party/bazel/python:pytype.bzl",
    "pytype_strict_binary",
    "pytype_strict_library",
    pytype_strict_test = "pytype_strict_contrib_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = EXTERNAL_USERS,
)

pytype_strict_library(
    name = "embed",
    srcs = [
        "embed.py",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("flax/nnx:v1"),
        pypi_requirement("jax"),
        pypi_requirement("optax"),
    ],
)

# Library target.
pytype_strict_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        ":embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/nn:table_stacking",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("absl/logging"),
        pypi_requirement("flax/nnx:v1"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("orbax/checkpoint"),
    ],
)

pytype_strict_library(
    name = "nnx",
    srcs = ["__init__.py"],
    visibility = [
        "//jax_tpu_embedding/sparsecore/lib:__pkg__",
        "//jax_tpu_embedding/sparsecore/lib/flax:__pkg__",
    ],
    deps = [
        ":checkpoint_utils",  # buildcleaner: keep
        ":embed",  # buildcleaner: keep
        ":offline_converter_lib",  # buildcleaner: keep
    ],
)

pytype_strict_library(
    name = "offline_converter_lib",
    srcs = ["offline_converter.py"],
    deps = [
        ":checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/utils",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
    ],
)

pytype_strict_binary(
    name = "offline_converter",
    srcs = ["offline_converter.py"],
    main = "offline_converter.py",
    deps = [
        ":checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
    ],
)

# Note that tests in //jax_tpu_embedding/sparsecore/examples/shakespeare exercise
# checkpoints converted using this binary.
pytype_strict_test(
    name = "offline_converter_test",
    srcs = ["offline_converter.py"],
    args = [
        "--input_checkpoint=jax_tpu_embedding/sparsecore/examples/shakespeare/testdata/checkpoint-2x2x2.tgz",
        "--output_checkpoint=TEST_UNDECLARED_OUTPUTS_DIR/converted_checkpoints",
        "--target_num_devices=4",
        "--target_device_kind=TPU_v5",
    ],
    data = ["//jax_tpu_embedding/sparsecore/examples/shakespeare:checkpoint_2x2x2"],
    main = "offline_converter.py",
    deps = [
        ":checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
    ],
)
