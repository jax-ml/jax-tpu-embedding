#!/bin/bash
#!/usr/bin/env bash

set -e

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

PIP_FILE_PREFIX="pip_package/"

export PYTHON_VERSION="${PYTHON_VERSION:-3}"
export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"

if [[ -z "${PYTHON_MINOR_VERSION}" ]]; then
  PYTHON="python${PYTHON_VERSION}"
else
  PYTHON="python${PYTHON_VERSION}.${PYTHON_MINOR_VERSION}"
fi

function main() {
  DEST=${1}
  if [[ -z "${DEST}" ]]; then
    echo "No destination directory provided."
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory.
  [ ! -d $DEST ] && mkdir -p "${DEST}"
  
  DEST=$(readlink -f "${DEST}")
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy jax_tpu_embedding files"

  cp ${PIP_FILE_PREFIX}setup.py "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L jax_tpu_embedding "${TMPDIR}"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  ${PYTHON} setup.py bdist_wheel
  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"