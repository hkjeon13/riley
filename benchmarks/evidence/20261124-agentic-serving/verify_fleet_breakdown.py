"""Check each displayed time budget against archived timestamps and SLO counts."""

import json
import math
import tarfile
from pathlib import Path

from summarize_agent_workflow import summarize

HERE = Path(__file__).resolve().parent


def verify():
    slo = json.loads((HERE / "registration.json").read_bytes())["engineering_slo_ms"]
    submitted = complete = goals = 0
    cache_reports = 0
    error_ms = 0
    with tarfile.open(HERE / "replicas-raw.tar.gz") as archive:
        for arm in "abcdef":
            waves = json.load(archive.extractfile(f"replicas/{arm}/capacity.json"))
            for wave in waves:
                result = summarize(wave["records"], slo)
                counts = result["submitted"], result["complete"], result["met_slo"]
                if counts != (
                    wave["submitted"],
                    wave["correct"],
                    wave["slo_completed"],
                ):
                    raise ValueError("changed workload denominator")
                cache = result["reported_final_prefix_cache"]
                valid = [
                    task["turns"][-1]["usage"] for task in wave["records"] if task["ok"]
                ]
                if cache["cached_tokens"] != sum(
                    x["prompt_tokens_details"]["cached_tokens"] for x in valid
                ):
                    raise ValueError("cache credit did not reconcile")
                if cache["prompt_tokens"] != sum(x["prompt_tokens"] for x in valid):
                    raise ValueError("prompt tokens did not reconcile")
                cache_reports += cache["reported_workflows"]
                for task in wave["records"]:
                    parts = result["cells"][task["workflow_id"]][
                        "final_latency_components_ms"
                    ]
                    if not task["ok"]:
                        if parts is not None:
                            raise ValueError("failed task counted")
                        continue
                    diff = abs(sum(parts.values()) - task["last_final_ms"])
                    if not math.isclose(diff, 0, abs_tol=1e-7):
                        raise ValueError("time was lost or counted twice")
                    error_ms = max(error_ms, diff)
                submitted += counts[0]
                complete += counts[1]
                goals += counts[2]
    with tarfile.open(HERE / "promotion-raw.tar.gz") as archive:
        vllm = json.load(
            archive.extractfile("promotion/pair-1-vllm/workflows-200.json")
        )
    unknown = summarize(vllm, slo)["reported_final_prefix_cache"]
    if (
        unknown["reported_workflows"] != 0
        or unknown["unreported_workflows"] != 5
        or unknown["cached_tokens"] is not None
    ):
        raise ValueError("absent vLLM cache details interpreted as measured")
    incomplete = {"workflow_id": "before-http", "ok": False, "turns": [{}]}
    if summarize([incomplete], slo)["failed"] != 1:
        raise ValueError("pre-dispatch failure omitted")
    return {
        "archived_launches": 6,
        "submitted": submitted,
        "complete": complete,
        "slo_complete": goals,
        "largest_reconciliation_error_ms": error_ms,
        "pre_http_failure_counted": True,
        "verified_workflow_cache_reports": cache_reports,
        "unreported_vllm_final_cache": unknown,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
