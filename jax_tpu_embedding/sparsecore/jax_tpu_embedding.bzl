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
"""Provides python test rules for jax tpu embedding TPU tests."""

load("//third_party/bazel/python:pytype.bzl", "pytype_strict_contrib_test")

# Visibility rules.
EXTERNAL_USERS = ["//visibility:public"]

# Use jax_tpu_embedding/sparsecore/lib/nn/embedding.py.
CORE_USERS = [
    "//jax_tpu_embedding/sparsecore:__subpackages__",
]

def tpu_py_strict_test(
        name,
        tags = None,
        deps = None,
        args = [],
        **kwargs):
    """Generates unit test for TPU.

    Args:
        name: Name of test. Will be prefixed by accelerator versions.
        tags: BUILD tags to apply to tests.
        deps: Dependencies of the test.
        args: Arguments to apply to tests.
        **kwargs: Additional named arguments to apply to tests.

    """
    tags = tags or []
    deps = deps or []
    kwargs.setdefault("main", "%s.py" % name)
    kwargs.setdefault("python_version", "PY3")

    args = [
        "--logtostderr",
    ] + args

    tags = [
        "requires-tpu",
    ] + tags

    pytype_strict_contrib_test(
        name = name,
        tags = tags,
        deps = deps,
        args = args,
        **kwargs
    )
