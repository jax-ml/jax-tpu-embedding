# jax_tpu_embedding (go/jax_tpu_embedding)

load("//tools/build_defs/license:license.bzl", "license")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

package(
    default_applicable_licenses = ["//third_party/py/jax_tpu_embedding:license"],
    default_visibility = [
        "//visibility:public",
    ],
)

license(
    name = "license",
    package_name = "jax_tpu_embedding",
)

licenses(["notice"])

exports_files(["LICENSE"])

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    parse_tests = False,
    deps = ["//third_party/py/jax_tpu_embedding:jax_tpu_embedding.bzl"],
)

pytype_strict_library(
    name = "tpu_embedding",
    srcs = ["tpu_embedding.py"],
    srcs_version = "PY3",
    deps = [
        ":config_utils",
        ":input_utils",
        ":pytype_utils",
        ":tpu_embedding_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental/jax2tf",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core/tpu/kernels:tpu_embedding_configuration_ops",
        "//third_party/tensorflow/python/tpu/ops:gen_tpu_embedding_ops",
    ],
)

pytype_strict_library(
    name = "config_utils",
    srcs = ["config_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pytype_utils",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/python/tpu:tpu_embedding_v2_utils",
    ],
)

pytype_strict_library(
    name = "input_utils",
    srcs = ["input_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":pytype_utils",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/tensorflow",
        "//third_party/py/tree",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_utils",
    srcs = ["tpu_embedding_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core/tpu/kernels:tpu_embedding_enqueue_ops",
        "//third_party/tensorflow/dtensor/python:gen_dtensor_ops",
        "//third_party/tensorflow/dtensor/python:tpu_util",
        "//third_party/tensorflow/python/eager:context",
    ],
)

pytype_strict_library(
    name = "checkpoint",
    srcs = ["checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/etils/epath",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental/gda_serialization:serialization",
        "//third_party/py/numpy",
        "//third_party/py/orbax/checkpoint",
        "//third_party/py/tensorstore",
    ],
)

pytype_strict_library(
    name = "pytype_utils",
    srcs = ["pytype_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core/protobuf/tpu:tpu_embedding_configuration_py_pb2",
    ],
)

copybara_config_test(
    name = "copybara_test",
    config = "copy.bara.sky",
    deps = [
        "//third_party/py/etils:copybara_utils",
    ],
)
