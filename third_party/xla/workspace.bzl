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

# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

XLA_COMMIT = "dae36884b3a38de63bd2601729808f0cf52cc1ac"
XLA_SHA256 = "983d483dc7c5aa448018f3108fc054a15763231bbb9293f52dcd20e18913163e"
XLA_ARCHIVE = "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)

def repo():
    http_archive(
        name = "xla",
        sha256 = XLA_SHA256,
        strip_prefix = "xla-{commit}".format(commit = XLA_COMMIT),
        urls = [
            # Try TF mirror first.
            "https://storage.googleapis.com/mirror.tensorflow.org/%s" % XLA_ARCHIVE[8:],
            XLA_ARCHIVE,
        ],
    )
