#!/usr/bin/env bash
set -euo pipefail
trial=$1
read -r stage pid <"$trial/remote-launch.txt"
case "$stage" in /mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-kv-pressure-run-20261120.*) ;; *) exit 1 ;; esac
prior="$(dirname "$stage")/w3-qkv-tracing-20261119.s9zZEnqW"
ssh -o BatchMode=yes aws-a100 "
  set -e
  cd '$stage'
  test \$(cat campaign.exit) = 0
  sha256sum -c dispatch.sha256 run-tools.sha256 binary.sha256 result.sha256 >/dev/null
  /usr/bin/python3 -B - '$stage' '$prior'
" <"$trial/verify_kv_pressure.py" >"$trial/verified-summary.tmp"
mv "$trial/verified-summary.tmp" "$trial/verified-summary.json"
