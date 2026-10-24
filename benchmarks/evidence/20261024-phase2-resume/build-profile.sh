#!/bin/bash
# Run after qualification so compilation does not overlap timed arms.
set -euo pipefail
RUN=/mnt/efs/nlp/psyche/riley-phase2-20261023/build
OUT=/mnt/efs/nlp/psyche/riley-phase2-resume-20261024
export RUSTUP_HOME=/home/nlp/riley-261013/rustup CARGO_HOME=/home/nlp/riley-261013/cargo RUSTUP_TOOLCHAIN=1.85.0
export PATH="$CARGO_HOME/bin:/usr/local/cuda-12.4/bin:/usr/local/bin:/usr/bin:/bin"
export CMAKE=/usr/local/bin/cmake CARGO_BUILD_JOBS=6 CMAKE_BUILD_PARALLEL_LEVEL=6
export CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4 CUDACXX=/usr/local/cuda-12.4/bin/nvcc RILEY_CUDA_ARCHITECTURES=80 CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 CARGO_TARGET_DIR="$RUN/target" TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
test -f "$OUT/qualification.execution.json"
cd "$RUN/source"
taskset -c 48-55 cargo build --locked --offline --release -p riley-server --features server,owned-hf-eager > "$OUT/build/server-profile.log" 2>&1
cp "$CARGO_TARGET_DIR/release/riley" "$RUN/riley-p2h-profile"
sha256sum "$RUN/riley-p2h-profile" > "$OUT/build/binary-profile.sha256"
for filter in native_profile native_defaults_match owned_decode_graph_requires_exact_iteration_profile; do
    taskset -c 48-55 cargo test --locked --offline --release -p riley-server --features server,owned-hf-eager --bin riley "$filter" > "$OUT/build/cpu-profile-$filter.log" 2>&1
done
taskset -c 48-55 cargo test --locked --offline --release -p riley-cuda --features cuda,cuda-cublas-gemm-probe --test flashinfer_attention_gpu --no-run --message-format=json > "$OUT/build/reference-test-build.jsonl" 2> "$OUT/build/reference-test-build.log"
EXE=$(python3 - "$OUT/build/reference-test-build.jsonl" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row.get('reason')=='compiler-artifact' and row.get('executable') and row['target']['name']=='flashinfer_attention_gpu':
        print(row['executable'])
PY
)
set +e
python3 /mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019/control.py --lease-dir /mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019/leases-v2 --index 7 run -- "$EXE" --ignored --nocapture > "$OUT/build/gpu-fi-reference.log" 2>&1
code=$?
printf '%s\n' "$code" > "$OUT/build/gpu-fi-reference.exit"
set -e
test "$code" = 0
touch "$OUT/build/profile.finished"
