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
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_binary")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

tpu_py_strict_test(
    name = "jax_sc_shakespeare_jit_test",
    srcs = ["jax_sc_shakespeare_jit.py"],
    args = [
        # "--vmodule=jax_sc_shakespeare_jit=1",
    ],
    main = "jax_sc_shakespeare_jit.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        "//jax_tpu_embedding/sparsecore/utils",
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
        pypi_requirement("orbax/checkpoint"),
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
    tags = ["oss_exclude"],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/lib/proto:embedding_spec_py_pb2",
        "//jax_tpu_embedding/sparsecore/utils",
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
        pypi_requirement("orbax/checkpoint"),
    ],
)

tpu_py_strict_test(
    name = "jax_sc_shakespeare_pipelined_jit_test",
    srcs = ["jax_sc_shakespeare_pipelined_jit_test.py"],
    args = [
        # "--vmodule=jax_sc_shakespeare_pipelined_jit_test=1",
    ],
    main = "jax_sc_shakespeare_pipelined_jit_test.py",
    tags = [
        "requires-net:external",  # For xprof uploading.
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:model",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_pipelining_utils",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding_spec",
        "//jax_tpu_embedding/sparsecore/utils",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("absl/testing:absltest"),
        pypi_requirement("clu:metrics"),
        pypi_requirement("clu:parameter_overview"),
        pypi_requirement("flax:core"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
    ],
)

tpu_py_strict_test(
    name = "jax_sc_shakespeare_jit_flax_nnx_test",
    srcs = ["jax_sc_shakespeare_jit_flax_nnx.py"],
    args = [
        "--checkpoint_dir=TEST_UNDECLARED_OUTPUTS_DIR/checkpoints",
        "--fdo_dir=TEST_UNDECLARED_OUTPUTS_DIR/fdo",
        # "--vmodule=jax_sc_shakespeare_jit_flax_nnx=1",
    ],
    main = "jax_sc_shakespeare_jit_flax_nnx.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:flax_nnx_model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:csv_file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("flax/nnx"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
    ],
)

# Test restoring from a checkpoint that was generated by training on a 2x2x2 TPU topology and
# restoring on a 2x2x1 TPU topology.
tpu_py_strict_test(
    name = "jax_sc_shakespeare_jit_flax_nnx_restore_test",
    srcs = ["jax_sc_shakespeare_jit_flax_nnx.py"],
    args = [
        "--checkpoint_dir=TEST_UNDECLARED_OUTPUTS_DIR/checkpoints",
        "--fdo_dir=TEST_UNDECLARED_OUTPUTS_DIR/fdo",
        # "--vmodule=jax_sc_shakespeare_jit_flax_nnx=1",
        "--cross_topology_checkpoint_restore=jax_tpu_embedding/sparsecore/examples/shakespeare/testdata/checkpoint-2x2x2.tgz",
    ],
    data = ["testdata/checkpoint-2x2x2.tgz"],
    main = "jax_sc_shakespeare_jit_flax_nnx.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:flax_nnx_model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:csv_file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("flax/nnx"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
    ],
)

# Test restarting from a checkpoint that was converted from
# testdata/checkpoint-2x2x2 to a 2x2x1 topology using the offline converter.
tpu_py_strict_test(
    name = "jax_sc_shakespeare_jit_flax_nnx_converted_test",
    srcs = ["jax_sc_shakespeare_jit_flax_nnx.py"],
    args = [
        "--input_checkpoint_dir=jax_tpu_embedding/sparsecore/examples/shakespeare/testdata/converted-2x2x1.tgz",
        "--fdo_dir=TEST_UNDECLARED_OUTPUTS_DIR/fdo",
        "--num_steps=1400",
        "--batch_size=256",
        "--log_frequency=100",
        "--loss_window=100",
        # "--vmodule=jax_sc_shakespeare_jit_flax_nnx=1",
    ],
    data = ["testdata/converted-2x2x1.tgz"],
    main = "jax_sc_shakespeare_jit_flax_nnx.py",
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:flax_nnx_model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:csv_file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("flax/nnx"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
    ],
)

filegroup(
    name = "checkpoint_2x2x2",
    srcs = ["testdata/checkpoint-2x2x2.tgz"],
    visibility = ["//jax_tpu_embedding/sparsecore/lib/flax/nnx:__pkg__"],
)

pytype_strict_binary(
    name = "jax_sc_shakespeare_jit_flax_nnx",
    srcs = ["jax_sc_shakespeare_jit_flax_nnx.py"],
    args = [
        "--alsologtostderr",
        # TPU chip configuration,
    ],
    main = "jax_sc_shakespeare_jit_flax_nnx.py",
    tags = ["oss_exclude"],
    deps = [
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:config",
        "//jax_tpu_embedding/sparsecore/examples/models/shakespeare:flax_nnx_model",
        "//jax_tpu_embedding/sparsecore/lib/fdo:csv_file_fdo_client",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:checkpoint_utils",
        "//jax_tpu_embedding/sparsecore/lib/flax/nnx:embed",
        "//jax_tpu_embedding/sparsecore/lib/nn:embedding",
        "//third_party/jax:tpu_support",
        pypi_requirement("absl:app"),
        pypi_requirement("absl/flags"),
        pypi_requirement("absl/logging"),
        pypi_requirement("flax/nnx"),
        pypi_requirement("jax"),
        pypi_requirement("numpy"),
        pypi_requirement("optax"),
        pypi_requirement("orbax/checkpoint"),
    ],
)
