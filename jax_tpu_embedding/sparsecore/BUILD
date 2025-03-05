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
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//third_party/bazel/python:pytype.bzl", "pytype_strict_library")
load(":configure_file.bzl", "configure_file")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:public"],
)

# Sets the version suffix for the package, e.g. dev20250321-805775f.
string_flag(
    name = "version_suffix",
    build_setting_default = "",
)

# Sets the git commit for the package, e.g. 805775fcb5f9272e4c52dce751b00cf7f70364f2.
string_flag(
    name = "git_commit",
    build_setting_default = "",
)

configure_file(
    name = "version",
    flag_substitutions = {
        "VERSION_SUFFIX": ":version_suffix",
        "GIT_COMMIT": ":git_commit",
    },
    output = "version.py",
    template = "version.py.in",
)

pytype_strict_library(
    name = "sparsecore",
    srcs = [
        "__init__.py",
        ":version",
    ],
    deps = [
        "//jax_tpu_embedding/sparsecore/lib",  # buildcleaner: keep
        "//jax_tpu_embedding/sparsecore/utils",  # buildcleaner: keep
    ],
)
