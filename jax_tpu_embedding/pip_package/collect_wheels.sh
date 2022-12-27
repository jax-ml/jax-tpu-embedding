#!/bin/bash

function collect_wheels() {
  release_version=$1
  wheel_version="${release_version}"
  wheel_folder=$2
  if [ "${release_version}" != "nightly" ]; then
    wheel_version=$( echo "${release_version}" | grep -oP '\d+.\d+(.\d+)?' )
  fi

  mkdir /tmp/staging-wheels
  pushd /tmp/staging-wheels
  cp $wheel_folder/*.whl .
  rename -v "s/^jax_tpu_embedding-(.*?)-py3/jax_tpu_embedding-${wheel_version}+$(date -u +%Y%m%d)-py3/" *.whl
  popd
  mv /tmp/staging-wheels/* .
  mv $wheel_folder/*.txt .
}
