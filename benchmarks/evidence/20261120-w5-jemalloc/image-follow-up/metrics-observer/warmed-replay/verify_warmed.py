"""Bind the mirrored RSS comparison and its intact warmup to one live process."""

import json
from pathlib import Path
import sys

import observer_client
import verify_observer
from warmed_client import PHASES, WARMUP
from verify_a100_candidate import digest, require


def verify(package, evidence, stage, reference):
    prior_path = reference / "verified.json"
    require(digest(prior_path) == "6ae0725185f626ae9f20d0c08098e3493b4ad065a2e0b90e4b7a1c40d20aa331",
            "original experiment changes")
    prior = observer_client.read(prior_path)
    observer_client.PHASES = PHASES
    verify_observer.PHASES = (WARMUP, *PHASES)
    result = verify_observer.verify(package, evidence, stage)
    for key in ("image_id", "candidate_binary_sha256", "checkpoint_sha256", "api_checks", "reservations", "same_counts_per_phase"):
        require(result[key] == prior[key], f"original experiment identity or workload differs: {key}")

    out = stage / "packaged-first" / "gpu-evidence"
    warm_file = out / "warmup.json"
    warm_meta = out / "warmup.observer.json"
    warm = observer_client.read(warm_file)
    metadata = observer_client.read(warm_meta)
    require(warm["status"] == "passed" and warm["problem_count"] == 0
            and warm["seconds"] == observer_client.SECONDS and warm["rate"] == observer_client.RATE
            and warm["counts"] == result["same_counts_per_phase"]
            and metadata["extra_metrics"] is False and metadata["samples"] == [],
            "warmup did not use the same workload and baseline observation")
    first = observer_client.read(out / f"{PHASES[0][0]}.observer.json")
    require(metadata["begin_ns"] < metadata["end_ns"] <= first["begin_ns"], "warmup overlapped the scored phases")

    buckets = result["phases"]
    require(all(buckets[name]["extra_metrics_requests"] >= observer_client.SECONDS - 1 if scrape else
                buckets[name]["extra_metrics_requests"] == 0 for name, scrape in PHASES),
            "observation cadence changed")
    result["matched_prior_report_sha256"] = digest(prior_path)
    result["phase_order"] = [name for name, _ in PHASES]
    result["warmup"] = {"rss_growth_kib": metadata["rss_end"]["rss_kib"] - metadata["rss_start"]["rss_kib"],
                        "anonymous_growth_kib": metadata["rss_end"]["anonymous_kib"] - metadata["rss_start"]["anonymous_kib"],
                        "workload_sha256": digest(warm_file), "measurement_sha256": digest(warm_meta)}
    result["paired_rss_delta_kib"] = {
        order: buckets[f"scrape-{order}"]["rss_growth_kib"] - buckets[f"baseline-{order}"]["rss_growth_kib"]
        for order in ("first", "second")
    }
    return result


if __name__ == "__main__":
    print(json.dumps(verify(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
