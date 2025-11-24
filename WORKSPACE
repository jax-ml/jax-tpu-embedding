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
workspace(name = "jax_tpu_embedding")

###############################################################################
##  XLA Initialization
###############################################################################
# This is adapted from JAX's WORKSPACE file.

# The XLA commit is determined by external/xla/workspace.bzl.
load("//third_party/xla:workspace.bzl", xla_repo = "repo")

xla_repo()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

# Initialize hermetic C++.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_ml_toolchain",
    sha256 = "1a911c79fc734c39538781a7a4672b06aab8354c1ddb985c98e3df78f430bcde",
    strip_prefix = "rules_ml_toolchain-f13852164b6fe240f8a989a744221a51e0d485cd",
    urls = [
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/f13852164b6fe240f8a989a744221a51e0d485cd.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")
register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.10": "//third_party/py:requirements_lock_3_10.txt",
        "3.11": "//third_party/py:requirements_lock_3_11.txt",
        "3.12": "//third_party/py:requirements_lock_3_12.txt",
        "3.13": "//third_party/py:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/bazel/python:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Load all XLA dependencies.
load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

# Even though we don't use CUDA, this is required since it is needed
# by TSL, one of our dependencies.
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)
cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

###############################################################################
##  SparseCore-Specific Dependencies
###############################################################################

HIGHWAY_VERSION = "1.2.0"
HIGHWAY_SHA256 = "7e0be78b8318e8bdbf6fa545d2ecb4c90f947df03f7aadc42c1967f019e63343"
HIGHWAY_ARCHIVE = "https://github.com/google/highway/archive/{version}.tar.gz".format(version = HIGHWAY_VERSION)
http_archive(
    name = "highway",
    sha256 = HIGHWAY_SHA256,
    strip_prefix = "highway-{version}".format(version = HIGHWAY_VERSION),
    urls = [HIGHWAY_ARCHIVE],
)

FUZZTEST_COMMIT = "0f82dad406f431ca5e8607626825be15423ba339"

http_archive(
    name = "com_google_fuzztest",
    strip_prefix = "fuzztest-" + FUZZTEST_COMMIT,
    url = "https://github.com/google/fuzztest/archive/" + FUZZTEST_COMMIT + ".zip",
)
