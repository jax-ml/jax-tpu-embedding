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
"""Hermetic Python initialization."""

load("@python//:defs.bzl", "interpreter")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS_WITH_LOCAL_WHEELS")
load("@rules_python//python:pip.bzl", "package_annotation", "pip_parse")

def python_init_pip():
    setuptools_annotations = {
        # We require the "Lorem ipsum.txt" file from the following directory,
        # but cannot depend directly on a filename containing spaces.
        "setuptools": package_annotation(
            data = [":site-packages/setuptools/_vendor/jaraco/text"],
        ),
    }

    pip_parse(
        name = "pypi",
        annotations = setuptools_annotations,
        python_interpreter_target = interpreter,
        requirements_lock = REQUIREMENTS_WITH_LOCAL_WHEELS,
    )
