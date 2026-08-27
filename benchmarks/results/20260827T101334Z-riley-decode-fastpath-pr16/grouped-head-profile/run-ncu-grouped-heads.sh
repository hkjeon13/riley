#!/usr/bin/env bash
# Profiles the candidate's steady decode launches after the fixed warmup span.
set -euo pipefail

# Reuse the provenance and workload bindings without executing its AB/BA tail.
source <(sed '/^# B,C,C,B,B,C,C,B,B,C keeps/,${d}' /input/run-native-profile.sh)

/usr/local/cuda/bin/ncu \
  --set basic \
  --launch-skip 500 \
  --launch-count 600 \
  --csv \
  --log-file /evidence/ncu-grouped-heads.csv \
  "$profile" \
  --output /evidence/ncu-grouped-heads-run.json \
  --role candidate \
  --pair-index 1 \
  --run-id decode-attention.8233056.ncu-candidate \
  --recorded-at-utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --implementation-id riley-decode-bucket-packed-gpu-grouped-heads-v2 \
  --runtime-flag-value bucket-packed-gpu \
  "${common[@]}"
