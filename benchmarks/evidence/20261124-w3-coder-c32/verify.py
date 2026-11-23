"""Check fixed-arrival C32 timing and locate the contribution of content gaps."""

import argparse
import json
from math import isclose, isfinite
from pathlib import Path
import runpy

RAG = runpy.run_path(str(Path(__file__).resolve().parent.parent / "20261124-w3-coder-arrivals/verify-replay.py"))
BASE = RAG["BASE"]
require, read, digest = (RAG[key] for key in ("require", "read", "digest"))
SCHEDULE_SHA = "e061b5399da890f08d53dd2e322a0d0f706287d4a10932fab99fd0b7b137dde4"


def content_gaps(run):
    combined = []
    for row in run["rows"]:
        gaps = row["content_gaps_s"]
        require(isinstance(gaps, list) and all(isfinite(value) and value >= 0 for value in gaps)
                and isclose(sum(gaps), row["e2e"] - row["ttft"], abs_tol=1e-8),
                "content gaps do not reconstruct the measured decode interval")
        combined.extend(gaps)
    combined.sort()
    require(bool(combined), "no measured inter-content gap")
    quantiles = {f"p{int(q*100)}_ms": combined[int(q * len(combined))] * 1000 for q in (.5, .95, .99)}
    require(quantiles["p99_ms"] == run["reported_itl_p99_ms"], "archived inter-content p99 differs")
    stalls = [value for value in combined if value >= .050]
    return {"samples": len(combined), **quantiles, "gaps_at_least_50ms": len(stalls),
            "share_of_decode_interval_in_50ms_gaps": sum(stalls) / sum(combined)}


def verify(stage):
    require((stage / "campaign.exit").read_text().strip() == "0"
            and (stage / "vllm-version.txt").read_text().strip() == "0.27.1+cu129", "C32 replay did not complete")
    require(digest(stage / "schedules.json") == SCHEDULE_SHA, "C32 schedule differs")
    data = read(stage / "schedules.json")
    BASE["confirm_order"](stage)
    by_gpu = {}
    for index, engines in BASE["ORDER"].items():
        plans = data["schedules"][str(index)]
        require(set(plans) == {"c32-riley-arrivals", "c32-vllm-arrivals"}, "both C32 schedule origins are required")
        holder = BASE["reservations"](stage, index)
        arms = {engine: RAG["arm"](stage, index, engine, plans, holder) for engine in engines}
        comparisons = {}
        for label in plans:
            riley, vllm = arms["riley"][label], arms["vllm"][label]
            result = RAG["pair"](riley, vllm)
            del result["ttft_p95_stage_met"], result["ttft_p95_final_met"]
            result["tpot_p50_goal_met"] = (result["riley_over_vllm"]["tpot_p50_ms"] <= 1.03
                                           if result["valid_matched_comparison"] else None)
            result["content_gaps"] = {"riley": content_gaps(riley), "vllm": content_gaps(vllm)}
            result["write_components"] = {
                engine: RAG["write_components"](cells[label]["rows"]) for engine, cells in arms.items()
            }
            comparisons[label] = result
        by_gpu[index] = comparisons
    return {"image_id": RAG["IMAGE"], "source_revision": RAG["REVISION"], "target_tpot_p50_ratio": 1.03,
            "max_write_error_p95_ms": RAG["MAX_WRITE_ERROR_P95_MS"], "by_gpu": by_gpu,
            "closed_origin_differences": data["paired_origin_differences"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
