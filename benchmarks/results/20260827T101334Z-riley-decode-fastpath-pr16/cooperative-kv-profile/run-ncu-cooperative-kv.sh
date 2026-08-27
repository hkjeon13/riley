#!/usr/bin/env bash
# Profiles final cooperative shared-KV candidate launches after warmup.
set -euo pipefail

# Reuse the exact source, executable, environment, and workload bindings
# without executing the AB/BA tail from the paired profile script.
source <(sed '/^# B,C,C,B,B,C,C,B,B,C keeps/,${d}' /input/run-native-profile.sh)

/usr/local/cuda/bin/ncu \
  --set basic \
  --launch-skip 500 \
  --launch-count 600 \
  --csv \
  --log-file /evidence/ncu-cooperative-kv.csv \
  "$profile" \
  --output /evidence/ncu-cooperative-kv-run.json \
  --role candidate \
  --pair-index 1 \
  --run-id decode-attention.436dad6.ncu-candidate \
  --recorded-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --implementation-id riley-decode-bucket-packed-gpu-cooperative-shared-kv-heads-v4 \
  --runtime-flag-value bucket-packed-gpu \
  "${common[@]}"
