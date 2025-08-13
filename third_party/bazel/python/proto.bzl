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
""" Python proto rule. """

load("@com_google_protobuf//bazel:protobuf.bzl", _py_proto_library = "py_proto_library")

def py_proto_library(name, deps = [], extra_deps = [], **kwargs):
    """Generates Python code from proto files.

    Args:
        name: A unique name for this target.
        deps: The list of proto_library rules to generate Python code for.
        extra_deps: The list of py_proto_library rules that correspond to the proto_library rules
            referenced by deps.
        **kwargs: Args passed through to py_proto_library rules.
    """
    srcs = []
    for dep in deps:
        if not dep.startswith(":") or not dep.endswith("_proto"):
            fail("py_proto_library %s's dep %s has an invalid name")
        src = dep[1:-6] + ".proto"
        srcs.append(src)
    _py_proto_library(name = name, srcs = srcs, deps = extra_deps, **kwargs)
