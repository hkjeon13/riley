#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
parent="$evidence/w4-decode-final-retry-20261120.5bTpAh8k"
prior="$evidence/w4-decode-retry-20261119.t5Ij5K7O"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$prior"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        dec13cff8d28408097fc5d2ce6f53b874ba59b377849d3e17447bdac1bb3c70b result.sha256 \
        79afa94d08bc2b41abd72f91b8c88eb211f77e97dfb2648bcd2ddd008ebc698d tools/run_accuracy.py \
        fd91fa83c0761dc9a0095f62c1f592aaf9b4bce708898e2549d4eca04a8d3150 tools/capture_decode_rows.py | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$parent"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        51a0c0367e07a0bc6544df0ecf963b654cd34ae44390a00b19b800b17229cd93 result.sha256 \
        113f97544a1da221250cd3f2672463c4b7cf3b8907a1987e388c4dc494f2d86e source-patched.sha256 \
        c4cbfc36d4653d104810f339995cc8a8161665913f431d60c0d434ffe624a131 tools/capture_final_rows.py \
        39a9979b427a89aae788aadd95e610203d6d0fb3c2c0034873c815e6c995594e tools/replay_final_head.py | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
mkdir src tools
cp -a -- "$parent/src/"{Cargo.toml,Cargo.lock,rust-toolchain.toml,crates,kernels,third_party} src/
cp -- "$prior/holder.py" .
cp -- "$prior/tools/"{run_accuracy.py,long_accuracy.py,compare_runner.py} tools/
cp -- "$prior/tools/capture_decode_rows.py" tools/capture_reference.py
cp -- "$parent/tools/"{capture_final_rows.py,replay_final_head.py} tools/
cp -- replay.py tools/tie_projection.py
cat >tools/capture_decode_rows.py <<'PY'
from tie_projection import capture

capture()
PY
ln -s -- "$prior/fixtures" fixtures
PYTHONPATH="$stage/tools" /usr/bin/python3 -B replay.py select fixtures | cmp - targets.json
(cd src && sha256sum --strict -c "$parent/source-patched.sha256")
/usr/bin/python3 -B replay.py prepare src targets.json
(cd src && sha256sum crates/riley-server/src/engine.rs crates/riley-runtime/src/llama/batch_executor.rs) >patched-source.sha256

export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 CARGO_TARGET_DIR="$stage/target"
export RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 W3_GPU_INDEX=7
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >build.log 2>&1
sha256sum target/release/riley >binary.sha256

"${control[@]}" --index 7 status >gpu7-before.json
printf 'CAPTURE %s\n' "$(date -u +%FT%TZ)"
rc=0
"${control[@]}" --index 7 run -- /usr/bin/python3 -B "$stage/tools/run_accuracy.py" riley || rc=$?
"${control[@]}" --index 7 status >gpu7-after.json
printf 'CAPTURE_END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
PYTHONPATH="$stage/tools" "$python" -B replay.py verify "$stage" "$root/models/Qwen2.5-7B-Instruct" >verified.json
sha256sum verified.json binary.sha256 patched-source.sha256 gpu7-{before,after}.json \
    riley/{client.log,receipt.json,server.log,rows/*.bf16} >result.sha256
