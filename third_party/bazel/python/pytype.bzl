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
"""Default (OSS) build versions of Python pytype rules."""

load("@bazel_skylib//lib:collections.bzl", "collections")
load("@rules_python//python:defs.bzl", "py_test")

# Placeholder to use until bazel supports pytype_library.
def pytype_library(name, deps = [], pytype_deps = [], pytype_srcs = [], **kwargs):
    _ = (pytype_deps, pytype_srcs)  # @unused
    native.py_library(name = name, deps = collections.uniq(deps), **kwargs)

# Placeholder to use until bazel supports pytype_strict_binary.
def pytype_strict_binary(name, deps = [], **kwargs):
    native.py_binary(name = name, deps = collections.uniq(deps), **kwargs)

# Placeholder to use until bazel supports pytype_strict_library.
def pytype_strict_library(name, deps = [], **kwargs):
    native.py_library(name = name, deps = collections.uniq(deps), **kwargs)

# Placeholder to use until bazel supports pytype_strict_contrib_test.
def pytype_strict_contrib_test(name, deps = [], **kwargs):
    py_test(name = name, deps = collections.uniq(deps), **kwargs)
