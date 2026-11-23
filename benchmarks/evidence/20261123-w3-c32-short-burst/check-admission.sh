#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-short-burst-gpu-20261123\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact short-only burst evidence path.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
(cd "$here" && shasum -a 256 -c verifier-inputs.sha256 >/dev/null)
checksum=$(shasum -a 256 "$here/result.sha256" | cut -d' ' -f1)
{
    python3.12 -B "$here/compose-verifier.py" |
        python3.12 -B -c 'import sys; source = sys.stdin.read(); print(source[:source.rindex("\nif __name__ == \"__main__\":\n")])'
    cat <<'PY'

from observation_reference import compare_cohort
stage = Path(sys.argv[1])
schedules = read(stage / "schedules.json")
report = {}
for gpu in ("6", "7"):
    timings = []
    for cap in (0, 8):
        arm = stage / f"private-short-d{cap}-gpu{gpu}"
        sample = read(arm / "workload.json")
        first = check_replay(sample["short"], schedules[gpu])["first_32_indices"]
        require(sorted(first) == list(range(32)), "unexpected request cohorts")
        timings.append(bind_observations((arm / "server.log").read_text(), sample)["short"])
    report[gpu] = {}
    for label, indices in (("all", range(96)), ("first_32", range(32)), ("later_64", range(32, 96))):
        comparison = compare_cohort(*timings, indices)
        report[gpu][label] = {
            "requests": comparison["requests"],
            "phases": {
                phase: {
                    "baseline_p50_ms": values["off_warm_p50"],
                    "candidate_p50_ms": values["private_p50"],
                    "paired_delta_p50_ms": values["paired_delta_p50"],
                }
                for phase, values in comparison["phase_ms"].items()
            },
        }
print(json.dumps({"capture_checksums_sha256": sys.argv[2], "paired_0_vs_8": report}, indent=2, allow_nan=False))
PY
} |
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  result.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c result.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
stock=\"\$evidence/w5-prefix-metrics-package-20261121.nEFyze5I\"
run=\"\$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR\"
warm=\"\$evidence/w3-coder-startup-warmup-20261121.db8Pexwl\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=\"\$1:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$warm:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$evidence/w5-current-rag-20261121.hvh3VS67\"
\"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\" \"\$2\"
' bash '$1' '$checksum'"
