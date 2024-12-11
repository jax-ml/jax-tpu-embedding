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
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_binary", "pytype_strict_contrib_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:__subpackages__"],
)

pytype_strict_contrib_test(
    name = "jax_sc_shakespeare_jit_test",
    srcs = ["jax_sc_shakespeare_jit.py"],
    args = [
        "--alsologtostderr",
        # TPU chip configuration,
        "--dump_dir=sponge",
        # "--vmodule=jax_sc_shakespeare_jit=1",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    main = "jax_sc_shakespeare_jit.py",
    tags = [
        "exclusive-if-local",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("clu:metrics"),
        pypi_requirement("clu:parameter_overview"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_binary(
    name = "jax_sc_shakespeare_jit",
    srcs = ["jax_sc_shakespeare_jit.py"],
    args = [
        "--alsologtostderr",
        # TPU chip configuration,
    ],
    main = "jax_sc_shakespeare_jit.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("clu:metrics"),
        pypi_requirement("clu:parameter_overview"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("jax:experimental"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
        pypi_requirement("tree"),
    ],
)

# bazel test -c opt --test_output=all //jax_tpu_embedding/sparsecore/examples/shakespeare:jax_sc_shakespeare_test
pytype_strict_contrib_test(
    name = "jax_sc_shakespeare_pmap_test",
    srcs = ["jax_sc_shakespeare_pmap.py"],
    args = [
        "--alsologtostderr",
        # TPU chip configuration,
        "--dump_dir=sponge",
        # "--vmodule=jax_sc_shakespeare_pmap=1",
    ],
    env = {
        "XLA_FLAGS": "--xla_dump_to=sponge",
    },
    main = "jax_sc_shakespeare_pmap.py",
    tags = [
        "exclusive-if-local",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("clu:metrics"),
        pypi_requirement("clu:parameter_overview"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("tree"),
    ],
)

pytype_strict_binary(
    name = "jax_sc_shakespeare_pmap",
    srcs = ["jax_sc_shakespeare_pmap.py"],
    args = [
        "--alsologtostderr",
        # TPU chip configuration,
    ],
    main = "jax_sc_shakespeare_pmap.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:dataset",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("clu:metrics"),
        pypi_requirement("clu:parameter_overview"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("tree"),
    ],
)
