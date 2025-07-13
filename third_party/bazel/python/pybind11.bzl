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
"""Pybind11 Extensions."""

load("@bazel_skylib//lib:collections.bzl", "collections")
load("@xla//xla/tsl:tsl.bzl", "tsl_pybind_extension_opensource")
load("//third_party/bazel_rules/rules_cc/cc:cc_library.bzl", "cc_library")

def pybind_extension(name, deps, **kwargs):
    # Add pybind11 to deps.
    deps = collections.uniq(deps + ["@pybind11"])
    tsl_pybind_extension_opensource(name = name, deps = deps, **kwargs)

def pybind_library(name, deps, **kwargs):
    # Add pybind11 and python headers to deps.
    deps = collections.uniq(deps + ["@pybind11", "@local_config_python//:python_headers"])
    cc_library(name = name, deps = deps, **kwargs)
