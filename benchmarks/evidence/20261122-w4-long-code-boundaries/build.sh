#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
native="$evidence/w4-native-projection-plans-20261120.yvjrWvOj"
trace="$evidence/w4-decode-layer-trace-20261120.kl10bRNz"
attention="$evidence/w4-o31-serving-20261120.7W3nr5JH"
down="$evidence/w4-fp32-down-serving-20261120.Saj3XXSY"
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-inputs.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/archive-inputs.sha256")
test "$(cat "$attention/campaign.exit")" = 0
test "$(cat "$down/campaign.exit")" = 0
mkdir src tools
cp "$native/tools/"{run_accuracy,long_accuracy,compare_runner}.py tools/
cp "$native/holder.py" .
ln -s "$native/fixtures" fixtures
tar -xf "$base/source.tar" -C src --no-same-owner
sha256sum --strict -c "$down/source-inputs.sha256"
(cd src && sha256sum --strict -c "$trace/source-base.sha256"
    gzip -dc "$trace/decode-layer-trace.patch.gz" | git apply --check -
    gzip -dc "$trace/decode-layer-trace.patch.gz" | git apply -
    sha256sum --strict -c "$trace/source-patched.sha256")
python3 -B prepare.py "$stage" "$attention/prepare_candidate.py" "$down/prepare_candidate.py"
sha256sum src/kernels/src/gemm.cu src/crates/riley-cuda/src/{gemm,ffi}.rs     src/crates/riley-runtime/src/llama/{forward.rs,executor/dispatch.rs} >patched-source.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >dependencies.txt
if grep -F 'not found' dependencies.txt; then exit 1; fi
strings target/release/riley | grep -F -e W4_O31_PLAN -e W4_DOWN_FP32_PLAN -e RILEY_W4_TRACE_DIR >switches.txt
for name in W4_O31_PLAN W4_DOWN_FP32_PLAN RILEY_W4_TRACE_DIR; do grep -qF "$name" switches.txt; done
sha256sum target/release/riley patched-source.sha256 toolchain.txt dependencies.txt switches.txt >build-result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
