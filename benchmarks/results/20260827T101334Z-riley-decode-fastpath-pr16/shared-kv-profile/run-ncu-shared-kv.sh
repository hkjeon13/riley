#!/usr/bin/env bash
# Profiles the shared-KV candidate's steady decode launches after warmup.
set -euo pipefail

# Reuse the source, environment, and workload bindings without running the
# AB/BA tail from the paired profile script.
source <(sed '/^# B,C,C,B,B,C,C,B,B,C keeps/,${d}' /input/run-native-profile.sh)

/usr/local/cuda/bin/ncu \
  --set basic \
  --launch-skip 500 \
  --launch-count 600 \
  --csv \
  --log-file /evidence/ncu-shared-kv.csv \
  "$profile" \
  --output /evidence/ncu-shared-kv-run.json \
  --role candidate \
  --pair-index 1 \
  --run-id decode-attention.883e682.ncu-candidate \
  --recorded-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --implementation-id riley-decode-bucket-packed-gpu-shared-kv-heads-v3 \
  --runtime-flag-value bucket-packed-gpu \
  "${common[@]}"
