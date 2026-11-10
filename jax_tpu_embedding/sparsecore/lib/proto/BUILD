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
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")
load("@rules_python//python:py_library.bzl", "py_library")
load("//jax_tpu_embedding/sparsecore:jax_tpu_embedding.bzl", "EXTERNAL_USERS")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = EXTERNAL_USERS,
)

proto_library(
    name = "embedding_spec_proto",
    srcs = ["embedding_spec.proto"],
)

py_proto_library(
    name = "embedding_spec_py_pb2",
    deps = [":embedding_spec_proto"],
)

py_library(
    name = "proto",
    srcs = ["__init__.py"],
    type_checking = True,
    visibility = ["//jax_tpu_embedding/sparsecore/lib:__pkg__"],
)
