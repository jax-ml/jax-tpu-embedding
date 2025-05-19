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
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//jax_tpu_embedding/sparsecore:internal"],
)

pytype_strict_library(
    name = "lib",
    srcs = ["__init__.py"],
    visibility = ["//jax_tpu_embedding/sparsecore:__pkg__"],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib/core",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/fdo",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/flax",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/nn",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/lib/proto",  # buildcleaner: keep
    ],
)
