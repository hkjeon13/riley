"""Recompute targeted BF16 scores and bind the eight rows to the existing W4 audit."""

from argparse import ArgumentParser
from hashlib import file_digest, sha256
import json
from pathlib import Path
import sys

AUDIT = Path("benchmarks/evidence/20261119-w4-layer-trace/shared-prefix-audit")


def sha_file(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verify(root, audit_path):
    sys.path.insert(0, str((root / "tools").resolve()))
    from capture_decode_rows import files, require, score_row

    audit = json.loads(audit_path.read_bytes())
    require((root / "campaign.exit").read_text().strip() == "0", "W4 runner did not complete")
    checksums = {}
    expected_receipts = {"idle-before.json", "idle-after.json", "riley/receipt.json",
                         "riley/client.log", "riley/server.log"}
    for line in (root / "result.sha256").read_text().splitlines():
        digest, name = line.split(None, 1)
        row = Path(name)
        require(name in expected_receipts or (row.parent == Path("riley/rows") and row.suffix == ".bf16"),
                f"unexpected receipt path: {name}")
        require(sha_file(root / name) == digest and name not in checksums, f"checksum mismatch: {name}")
        checksums[name] = digest
    require(expected_receipts <= checksums.keys() and len(checksums) == 13, "W4 receipt set is incomplete")
    before, after = (json.loads((root / f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
    require(all(before[key] == after[key] for key in before if key != "updated_ns")
            and after["phase"] == "idle" and after["physical_index"] == 7 and after["pid"] == 2431237
            and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3
            and not after["other_processes"], "original A100 7 lease was not restored")
    gpu = json.loads((root / "riley/receipt.json").read_bytes())
    binary = (root / "binary.sha256").read_text().split()
    keys = {Path(name).name: digest for name, digest in gpu["sha256"].items()}
    require(gpu["mode"] == "riley" and gpu["status"] == "passed" and gpu["server_exit"] == 0
            and gpu["holder_pid"] == after["pid"] and gpu["holder_start_ticks"] == after["start_ticks"]
            and gpu["gpu_uuid"] == after["gpu"] and gpu["observed_active"] is True and gpu["nvml_samples"] > 0
            and binary == [keys["riley"], "target/release/riley"]
            and keys["riley-checkpoint.json"] == "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f",
            "GPU evidence does not bind the build, checkpoint, holder and normal shutdown")
    earlier = {suite: {row["label"]: row for row in values["cases"]} for suite, values in audit["suites"].items()}
    receipt = json.loads((root / "riley/client.log").read_bytes())
    observed = []
    for original, report in zip(files(root / "fixtures"), receipt["cases"], strict=True):
        suite, case, base, _, offset, hf_id = original
        prompt_len = len(base["prompt_tokens"])
        riley_id = base["tokens"][offset]
        key = f"riley/rows/{prompt_len}-{offset}.bf16"
        row = report["row"]
        require((suite, case.label, prompt_len, offset, riley_id, hf_id) == (
            report["suite"], report["case"], report["prompt_len"], report["offset"],
            report["riley_id"], report["hf_id"]), f"{suite}/{case.label}: wrong request identity")
        base_hash = sha256(json.dumps(base["tokens"]).encode()).hexdigest()
        require(base_hash == report["observed_output_hash"], f"{suite}/{case.label}: served ID digest changed")
        require(key in checksums and score_row((root / key).read_bytes(), row["captured_vocabulary"],
                                              row["addressable_vocabulary"], riley_id, hf_id) == row,
                f"{suite}/{case.label}: captured scores changed")
        previous = earlier[suite][case.label].get("divergence")
        require((previous is None and riley_id == hf_id) or
                (previous is not None and (previous["offset"], previous["riley_id"], previous["hf_sdpa_id"])
                 == (offset, riley_id, hf_id)), f"{suite}/{case.label}: prior first divergence changed")
        observed.append({
            **report, "prior_first_divergence": previous,
            "is_tied_with_hf_id_in_riley_row": hf_id in row["tied_maximum_ids"],
        })
    require(len(observed) == 8, "missing row evidence")
    divergences = [case for case in observed if case["prior_first_divergence"] is not None]
    controls = [case for case in observed if case["prior_first_divergence"] is None]
    require(len(divergences) == 7 and len(controls) == 1, "expected seven divergences and a matching control")
    decode = [case for case in divergences if case["offset"] > 0]
    unique = sum(len(case["row"]["tied_maximum_ids"]) == 1 for case in divergences)
    return {
        "scope": "Actual Riley BF16 logits at seven first divergences and one matching control.",
        "cases": observed, "input_audit_sha256": sha_file(audit_path),
        "decisions": {
            "divergences": len(divergences), "decode_divergences": len(decode),
            "riley_hf_id_ties": sum(case["is_tied_with_hf_id_in_riley_row"] for case in divergences),
            "riley_hf_id_decode_ties": sum(case["is_tied_with_hf_id_in_riley_row"] for case in decode),
            "unique_riley_maxima": unique,
            "largest_tied_id_would_match_hf_at_divergence": sum(
                max(case["row"]["tied_maximum_ids"]) == case["hf_id"] for case in divergences
            ),
            "largest_tied_id_would_change_matching_control": any(
                max(case["row"]["tied_maximum_ids"]) != case["hf_id"] for case in controls
            ),
        },
        "binary_sha256": keys["riley"], "checkpoint_sha256": keys["riley-checkpoint.json"],
        "gpu_server_exit": gpu["server_exit"], "nvml_samples": gpu["nvml_samples"],
        "original_20gib_holder_restored": True, "result_sha256": checksums,
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--audit", default=AUDIT / "result.json", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.audit), indent=2, allow_nan=False))
