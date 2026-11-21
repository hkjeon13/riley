#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-current-fixed-rag-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact A100 fixed-arrival RAG evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$expected'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
evidence=$(dirname "$1")
package="$evidence/w5-current-allocator-20261121.ZVgY3v8o"
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
reference="$evidence/w5-current-rag-20261121.hvh3VS67"
(cd "$reference" && sha256sum -c dispatch.sha256 result.sha256 >/dev/null)
printf '%s  %s\n' 5e506712f4f1be0b955874179a40f4af490b07847630fd756472ad7e29376a33 "$reference/verify_allocator_image.py" | sha256sum -c >/dev/null
python=$(dirname "$evidence")/tools/vllm-venv/bin/python
export PYTHONPATH="$1:$reference:$stock/harness/ci:$stock/src/ci:$stock/harness/benchmarks/live"
cmp verified.json <("$python" -B verify_replay.py "$package" "$evidence" "$1" "$reference")
cat verified.json
REMOTE
