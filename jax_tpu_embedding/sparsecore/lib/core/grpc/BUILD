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
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "CORE_USERS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = CORE_USERS,
)

proto_library(
    name = "all_reduce_proto",
    srcs = ["all_reduce.proto"],
    # has_services = True,
)

cc_proto_library(
    name = "all_reduce_cc_proto",
    deps = [":all_reduce_proto"],
)

cc_grpc_library(
    name = "all_reduce_cc_grpc_proto",
    srcs = [":all_reduce_proto"],
    grpc_only = True,
    deps = [":all_reduce_cc_proto"],
)

cc_library(
    name = "grpc_all_reduce_interface",
    srcs = ["all_reduce_interface.cc"],
    hdrs = ["all_reduce_interface.h"],
    deps = [
        ":all_reduce_cc_grpc_proto",
        ":all_reduce_service_impl",
        ":grpc_credentials",
        "//jax_tpu_embedding/sparsecore/lib/core:all_reduce_interface",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "all_reduce_service_impl",
    srcs = ["all_reduce_service_impl.cc"],
    hdrs = ["all_reduce_service_impl.h"],
    deps = [
        ":all_reduce_cc_grpc_proto",
        ":all_reduce_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "grpc_credentials",
    hdrs = ["grpc_credentials.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "minibatching_node",
    hdrs = ["minibatching_node.h"],
    deps = [
        ":all_reduce_service_impl",
        ":grpc_all_reduce_interface",
        ":grpc_credentials",
        "//jax_tpu_embedding/sparsecore/lib/core:all_reduce_interface",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc/oss:grpc_credentials",  # buildcleaner: keep
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.cc"],
    deps = [
        ":all_reduce_cc_proto",
        ":all_reduce_service_impl",
        ":grpc_all_reduce_interface",
        ":grpc_credentials",
        "//jax_tpu_embedding/sparsecore/lib/core/grpc/oss:grpc_credentials",  # buildcleaner: keep
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_fuzztest//fuzztest",
        "@com_google_fuzztest//fuzztest:googletest_fixture_adapter",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:test",
    ],
)
