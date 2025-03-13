#!/bin/bash
#
# Script to build the python wheel.  Run from the root folder.

# Determine the hermetic python version.
if [ -z "$JTE_HERMETIC_PYTHON_VERSION" ]; then
  if [ -z "$HERMETIC_PYTHON_VERSION" ]; then
    # Use the default hermetic python version.
    JTE_HERMETIC_PYTHON_VERSION=3.12
  else
    JTE_HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION}"
  fi
fi

# Try to determine the git commit ID if not set.
if [ -z "$JTE_GIT_SHA" ]; then
  # Extract git hash from current git folder, if any.
  JTE_GIT_SHA=`git rev-parse HEAD 2> /dev/null || echo ""`
fi

# Determine the appropriate wheel suffix.  If it's not release,
# and if the version suffix is not explicitly set, build a dev version.
if [ -z "$JTE_RELEASE" ] && [ -z "$JTE_VERSION_SUFFIX" ]; then
  # Build suffix as dev${DATE}
  JTE_VERSION_SUFFIX="dev$(date '+%Y%m%d')"
fi

# Output directory for the wheel.
if [ -z "$JTE_WHEEL_OUTDIR" ]; then
  JTE_WHEEL_OUTDIR="$PWD/dist"
fi

echo "JTE_HERMETIC_PYTHON_VERSION: ${JTE_HERMETIC_PYTHON_VERSION}"
echo "JTE_RELEASE: ${JTE_RELEASE}"
echo "JTE_VERSION_SUFFIX: ${JTE_VERSION_SUFFIX}"
echo "JTE_GIT_SHA: ${JTE_GIT_SHA}"
echo "JTE_WHEEL_OUTDIR: ${JTE_WHEEL_OUTDIR}"

bazel run //tools:build_wheel --config=release_linux_x86_64 \
  --repo_env=HERMETIC_PYTHON_VERSION="${JTE_HERMETIC_PYTHON_VERSION}" \
  --//jax_tpu_embedding/sparsecore:version_suffix="${JTE_VERSION_SUFFIX}" \
  --//jax_tpu_embedding/sparsecore:git_commit="${JTE_GIT_SHA}" \
  -- --output_dir="${JTE_WHEEL_OUTDIR}"
