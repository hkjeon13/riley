#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
prior="$(dirname "$stage")/w3-aged-decode-native-20261124.tNZ3fCMo"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-dispatch.sha256
test "$(cat "$prior/build.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
(cd "$prior" &&
    sha256sum --strict -c "$stage/parent.sha256" &&
    sha256sum --strict -c build-result.sha256 result.sha256 >/dev/null)
cp -- "$prior"/{riley-diagnostic,dependencies.txt,toolchain.txt,patched-source.sha256,build-result.sha256,help.txt} .
mkdir src
(cd "$prior/src" &&
    sha256sum --strict -c "$stage/patched-source.sha256" &&
    cp --parents -- crates/riley-scheduler/src/scheduler.rs crates/riley-scheduler/tests/aged_decode_reservation.rs \
        crates/riley-server/src/{engine,main}.rs "$stage/src")
sha256sum --strict -c build-result.sha256
(cd src && sha256sum --strict -c ../patched-source.sha256)
printf 'REUSED_BINARY %s\n' "$(date -u +%FT%TZ)"
