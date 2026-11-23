#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w4-schema-ids-20261123.ZsUYnqkm"
model="$root/models/Qwen2.5-Coder-14B-Instruct"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export RUSTC="$toolchain/rustc" CARGO_HOME="$root/tools/cargo"
export CARGO_TARGET_DIR="$stage/target" CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
cargo="$toolchain/cargo"
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
printf '%s  %s\n' \
    7dc88437ba28ea18ca1c0c11b9b3cd85324110da913ce1b10c521da60ce54dcb "$package/source.tar" \
    82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00 "$model/riley-checkpoint.json" |
    sha256sum --strict -c
mkdir src
tar -xf "$package/source.tar" -C src
"$RUSTC" --version --verbose >rustc.txt
"$cargo" --version >cargo.txt
lscpu --json >cpu.json
lscpu --parse=CPU,CORE,SOCKET,ONLINE | /usr/bin/python3 -B -c '
import os, sys
allowed = os.sched_getaffinity(0)
first = None
for line in sys.stdin:
    if line.startswith("#"):
        continue
    number, core, socket, online = line.strip().split(",")
    cpu = int(number)
    if cpu not in allowed or online != "Y":
        continue
    if first is None:
        first = cpu, core, socket
    elif core != first[1] and socket == first[2]:
        print(f"{first[0]},{cpu}")
        break
else:
    raise SystemExit("could not identify two allowed physical cores on the same socket")
' >affinity.txt
build() {
    local name=$1
    printf 'BUILD %s %s\n' "$name" "$(date -u +%FT%TZ)"
    "$cargo" build --offline --locked --release --manifest-path src/tools/riley-checkpoint/Cargo.toml >"build-$name.log" 2>&1
    install -m 700 target/release/riley-checkpoint "$name"
}
(cd src && sha256sum --strict -c "$stage/baseline-inputs.sha256")
build baseline
(cd src
    git apply --check "$stage/overlap.patch"
    git apply "$stage/overlap.patch"
    sha256sum --strict -c "$stage/candidate-inputs.sha256")
build candidate
sha256sum baseline candidate >binary.sha256
test "$(cut -d' ' -f1 binary.sha256 | sort -u | wc -l)" -eq 2
"$cargo" test --offline --locked --manifest-path src/Cargo.toml -p riley-model --test checkpoint_loading >linux-tests.log 2>&1
pin=$(cat affinity.txt)
trial() {
    local name=$1 engine=$2
    printf 'START %s %s %s\n' "$name" "$engine" "$(date -u +%FT%TZ)"
    taskset -c "$pin" /usr/bin/time -f 'wall=%e\nuser=%U\nsystem=%S\nrss_kib=%M\nmajor_faults=%F\nminor_faults=%R\nexit=%x' \
        -o "$name.time" "./$engine" verify --checkpoint "$model" >"$name.stdout" 2>"$name.stderr"
    test ! -s "$name.stderr"
    test "$(cat "$name.stdout")" = "checkpoint verify: verified"
    printf 'END %s %s %s\n' "$name" "$engine" "$(date -u +%FT%TZ)"
}
trial 01-baseline baseline
trial 02-candidate candidate
trial 03-candidate candidate
trial 04-baseline baseline
printf '%s  %s\n' 82a56cdab6e16e31478b7449be768dce15992b448d543fea4b2f2ed163237d00 "$model/riley-checkpoint.json" |
    sha256sum --strict -c
sha256sum *.time *.stdout *.stderr rustc.txt cargo.txt cpu.json affinity.txt binary.sha256 linux-tests.log >result.sha256
