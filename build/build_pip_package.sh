#!/bin/bash
#!/usr/bin/env bash
# Tool to build the JAX SparseCore pip package.
#
# Usage:
#   bazel build build:build_pip_package
#   bazel-bin/build/build_pip_package
#
# Arguments:
#   output_dir: An output directory. Defaults to `$PWD`.

set -e  # fail and exit on any command erroring

osname="$(uname -s | tr 'A-Z' 'a-z')"
echo "$osname"

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${osname}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_macos() {
  [[ "${osname}" == "darwin" ]]
}

function abspath() {
  cd "$(dirname "$1")"
  echo "$PWD/$(basename "$1")"
  cd "$OLDPWD"
}

plat_name=""
if is_macos; then
  if [[ x"$(arch)" == x"arm64" ]]; then
    plat_name="--plat-name macosx_11_0_arm64"
  else
    plat_name="--plat-name macosx-10.9-x86_64"
  fi
fi

main() {
  local output_dir
  output_dir="$1"

  if [[ -z "${output_dir}" ]]; then
    output_dir="${PWD}"
  fi
  mkdir -p "${output_dir}"
  output_dir="$(abspath "${output_dir}")"
  echo "=== Destination directory: ${output_dir}"
  echo "=== Current directory: ${PWD}"

  runfiles="$PWD"
  if [[ -d "bazel-bin/jax_tpu_embedding" ]]; then
    # Running from build directory.
    if is_windows; then
      runfiles="bazel-bin/build/build_pip_package.exe.runfiles/jax_tpu_embedding"
    else
      runfiles="bazel-bin/build/build_pip_package.runfiles/jax_tpu_embedding"
    fi
  fi

  local temp_dir
  temp_dir="$(mktemp -d)"
  trap "rm -rf ${temp_dir}" EXIT
  echo "=== Using tmpdir ${temp_dir}"

  cp -LR "${runfiles}/jax_tpu_embedding" "${temp_dir}"
  cp "${runfiles}/LICENSE" "${temp_dir}"
  cp "${runfiles}/pyproject.toml" "${temp_dir}"
  cp "${runfiles}/README.md" "${temp_dir}"

  pushd "${temp_dir}" > /dev/null

  if (which python3) | grep -q "python3"; then
      installed_python="python3"
  elif (which python) | grep -q "python"; then
      installed_python="python"
  fi

  # Build pip package
  # The plat_name variable is intentionally unquoted since it _does_
  # intentionally contain spaces.
  # shellcheck disable=SC2086
  $installed_python -m build

  cp dist/*.whl "${output_dir}"
}

main "$@"
