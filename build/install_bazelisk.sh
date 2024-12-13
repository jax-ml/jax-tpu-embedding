#!/bin/bash
#!/usr/bin/env bash
# ==============================================================================
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
# ==============================================================================
#
# Script to install bazelisk locally to $HOME/bin.
#
# Usage:
#   . build/install_bazelisk.sh

if [ -z "${BAZELISK_VERSION}" ]; then
  BAZELISK_VERSION=v1.15.0
fi

# Downloads bazelisk to ~/bin as `bazel`.
function install_bazelisk {
  case "$(uname -s)" in
    Darwin) local name=bazelisk-darwin-amd64 ;;
    Linux)
      case "$(uname -m)" in
       x86_64) local name=bazelisk-linux-amd64 ;;
       aarch64) local name=bazelisk-linux-arm64 ;;
       *) die "Unknown machine type: $(uname -m)" ;;
      esac ;;
    *) die "Unknown OS: $(uname -s)" ;;
  esac

  mkdir -p "$HOME/bin"
  wget --no-verbose -O "$HOME/bin/bazel" \
      "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/$name" \
      2> /dev/null

  chmod u+x "$HOME/bin/bazel"
  if [[ ! ":$PATH:" =~ :"$HOME"/bin/?: ]]; then
    export PATH="$HOME/bin:$PATH"
  fi
}

install_bazelisk
