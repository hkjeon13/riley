#!/bin/bash
set -euo pipefail
P=/mnt/efs/nlp/psyche/riley-phase2-20261023
O=/mnt/efs/nlp/psyche/riley-phase2-resume-20261024
W=/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019
while [ ! -f "$O/diagnostic.finished" ]; do sleep 5; done
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
EXE=$(find "$P/build/target/release/deps" -maxdepth 1 -type f -name 'flashinfer_attention_gpu-*' -executable | head -1)
set +e
python3 "$W/control.py" --lease-dir "$W/leases-v2" --index 7 run -- "$EXE" --ignored --nocapture > "$O/random-before.log" 2>&1
code=$?
printf '%s\n' "$code" > "$O/random-before.exit"
exit "$code"
