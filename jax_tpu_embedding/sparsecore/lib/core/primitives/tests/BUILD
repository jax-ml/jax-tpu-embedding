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
load("//jax_tpu_embedding/sparsecore:tpu.bzl", "tpu_py_strict_test")
load("//third_party/bazel/python:pypi.bzl", "pypi_requirement")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:__subpackages__"],
)

tpu_py_strict_test(
    name = "sparse_dense_matmul_csr_test",
    srcs = [
        "sparse_dense_matmul_csr_test.py",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_csr_with_mini_batching_test",
    srcs = [
        "sparse_dense_matmul_csr_with_mini_batching_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_csr_with_mini_batching",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_grad_with_sgd_with_mini_batching_test",
    srcs = [
        "sparse_dense_matmul_grad_with_sgd_with_mini_batching_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_sgd_with_mini_batching",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_grad_with_adagrad_with_mini_batching_test",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad_with_mini_batching_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad_with_mini_batching",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_grad_with_sgd_test",
    srcs = [
        "sparse_dense_matmul_grad_with_sgd_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_sgd",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_optimizer_grad_test",
    srcs = [
        "sparse_dense_matmul_optimizer_grad_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:optimizers_computation",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_optimizer_grad",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_grad_with_adagrad_test",
    srcs = [
        "sparse_dense_matmul_grad_with_adagrad_test.py",
    ],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core:input_preprocessing",
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_adagrad",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("einops"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
    ],
)

pytype_strict_contrib_test(
    name = "sparse_dense_matmul_grad_with_laprop_test",
    srcs = ["sparse_dense_matmul_grad_with_laprop_test.py"],
    args = [
        # TPU chip configuration,
        "--logtostderr",
    ],
    tags = [
        "exclusive-if-local",
        "requires-tpu",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core/primitives:sparse_dense_matmul_grad_with_laprop",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("absl/testing:parameterized"),
        pypi_requirement("numpy"),
    ],
)
