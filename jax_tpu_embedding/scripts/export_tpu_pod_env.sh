#!/bin/bash

# This script needs to be run for TPU Pod slices to initialize multihost
# environment variables before using jax embedding api. It needs to be run on
# each host.
# Usage: run `source export_tpu_pod_env.sh` on each host.

retry_delay_seconds=0.5

_get_metadata() {
  local url="$1"
  local num_retrys="$2"
  api_resp=""

  for i in $(seq 1 $num_retrys); do
    api_resp=$(curl -s -H "Metadata-Flavor: Google" "$url")
    if [ $? -eq 0 ] && [ $(echo "$api_resp" | wc -l) -gt 0 ]; then
      echo "$api_resp"
      return 0
    fi
    sleep $retry_delay_seconds
  done

  echo "Getting metadata for $url failed for $num_retrys tries" >&2
  return 1
}

get_metadata() {
  local key="$1"
  local gce_metadata_endpoint="http://$(echo "${GCE_METADATA_IP:-metadata.google.internal}")"
  local url="${gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/${key}"
  _get_metadata "$url"
}

get_host_ip() {
  local gce_metadata_endpoint="http://$(echo "${GCE_METADATA_IP:-metadata.google.internal}")"
  local url="${gce_metadata_endpoint}/computeMetadata/v1/instance/network-interfaces/0/ip"
  _get_metadata "$url"
}

export TPU_HOSTNAME_OVERRIDE=$(get_host_ip)
export DTENSOR_CLIENT_ID=$(get_metadata agent-worker-number)
export DTENSOR_NUM_CLIENTS=$(get_metadata worker-network-endpoints | tr ',' '\n' | wc -l)
export DTENSOR_JOB_NAME=tpu_worker
export DTENSOR_JOBS=$(get_metadata worker-network-endpoints | tr ',' '\n' | awk -F: '{print $3 ":9991"}' | paste -sd ',')


echo "DTENSOR_CLIENT_ID: ${DTENSOR_CLIENT_ID}"
echo "DTENSOR_NUM_CLIENTS: ${DTENSOR_NUM_CLIENTS}"
echo "DTENSOR_JOB_NAME: ${DTENSOR_JOB_NAME}"
echo "TPU_HOSTNAME_OVERRIDE: ${TPU_HOSTNAME_OVERRIDE}"
echo "DTENSOR_JOBS: ${DTENSOR_JOBS}"
