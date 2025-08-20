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
# buildifier: disable=module-docstring
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/xla:revision.bzl", "XLA_COMMIT", "XLA_SHA256")

XLA_ARCHIVE = "https://api.github.com/repos/openxla/xla/tarball/{commit}".format(commit = XLA_COMMIT)

def repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        type = "tar.gz",
        strip_prefix = "openxla-xla-{commit}".format(commit = XLA_COMMIT[:7]),
        urls = [
            # Try TF mirror first.
            "https://storage.googleapis.com/mirror.tensorflow.org/%s" % XLA_ARCHIVE[8:],
            XLA_ARCHIVE,
        ],
    )
