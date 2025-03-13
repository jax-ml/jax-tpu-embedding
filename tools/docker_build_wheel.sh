#!/bin/bash
#
# Script to build python wheels in a docker container.  Assumes linux x86_64.
# Run from the root folder.

if [ -z "$JTE_DOCKER_IMAGE" ]; then
  JTE_DOCKER_IMAGE=us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest
fi

if [ -z "$JTE_DOCKER_WORKDIR" ]; then
  JTE_DOCKER_WORKDIR=/build/jax_tpu_embedding
fi

# Mark the wheel output as relative to the docker build folder.
if [ -z "$JTE_WHEEL_OUTDIR" ]; then
  JTE_WHEEL_OUTDIR="${JTE_DOCKER_WORKDIR}/dist"
fi

# Try to determine the git commit ID locally if not set.
# (The docker container does not contain git).
if [ -z "$JTE_GIT_SHA" ]; then
  # Extract git hash from current git folder, if any.
  JTE_GIT_SHA=`git rev-parse HEAD 2> /dev/null || echo ""`
fi

docker run \
  -v "$PWD":"${JTE_DOCKER_WORKDIR}" \
  -w "${JTE_DOCKER_WORKDIR}" \
  --env HERMETIC_PYTHON_VERSION="${HERMETIC_PYTHON_VERSION}" \
  --env JTE_HERMETIC_PYTHON_VERSION="${JTE_HERMETIC_PYTHON_VERSION}" \
  --env JTE_RELEASE="${JTE_RELEASE}" \
  --env JTE_VERSION_SUFFIX="${JTE_VERSION_SUFFIX}" \
  --env JTE_GIT_SHA="${JTE_GIT_SHA}" \
  --env JTE_WHEEL_OUTDIR="${JTE_WHEEL_OUTDIR}" \
  "${JTE_DOCKER_IMAGE}" \
  bash -c tools/local_build_wheel.sh
