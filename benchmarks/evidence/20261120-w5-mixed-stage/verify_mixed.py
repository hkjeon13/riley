"""Tie public mixed-iteration witnesses to the exact distribution and holder receipts."""

import json
from pathlib import Path
import sys

from verify_smoke import BINARY, GPU, IMAGE, digest, read, reservations


def verify(root):
    reservations(root, "before")
    reservations(root, "after")
    assert (root / "exit").read_text().strip() == "0"
    output = root / "gpu-evidence"
    report = read(output / "report.json")
    assert (report["status"], report["image_id"], report["gpu_uuid"], report["riley_binary_sha256"]) == (
        "passed", IMAGE, GPU, BINARY)
    assert report["checkpoint_sha256"] == "655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f"
    assert report["api_checks"] == 40 and report["shutdown_exit"] == 0 and report["tool_checks_enabled"] is True
    assert report["mixed_stage"]["server_arguments"] == [
        "--inference-profile", "native-bf16-varlen-v1", "--max-active-sequences", "2",
        "--max-sequence-tokens", "8192", "--max-output-tokens", "256", "--kv-blocks", "4096",
        "--batch-token-budget", "512", "--prefill-chunk-tokens", "256", "--batch-shape-buckets",
        "1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512",
    ]
    for name, expected in report["artifacts_sha256"].items():
        assert (output / name).parent == output and digest(output / name) == expected
    mixed = read(output / "mixed-stage.json")
    assert mixed["status"] == "passed" and mixed["prefill_chunk_tokens"] == 256
    assert mixed["serial_max_active_rows"]["prefill"] == 256
    assert mixed["baselines"]["prefill"]["prompt_tokens"] >= 2048
    assert mixed["baselines"]["decode"]["generated_ids"] >= 64
    assert len(mixed["trials"]) == 3 and sum(trial["matching_responses"] for trial in mixed["trials"]) == 6
    assert report["mixed_stage"]["matching_responses"] == 6
    last = 0
    for trial in mixed["trials"]:
        before = trial["prelaunch"]
        assert before["iteration"] > last and before["active_rows"] == before["inflight"] == 1
        assert before["waiting"] == 0 and before["gpu_ns"] > 0 and trial["witnesses"]
        last = before["iteration"]
        for witness in trial["witnesses"]:
            assert witness["iteration"] > last
            assert witness["active_rows"] == 257 and witness["inflight"] == 2 and witness["gpu_ns"] > 0
            assert witness["active_rows"] + witness["padding_rows"] == witness["selected_dense_rows"]
            last = witness["iteration"]
    assert report["mixed_stage"]["observed_mixed_iterations"] == sum(len(t["witnesses"]) for t in mixed["trials"])
    assert all(mixed["drained"][field] == 0 for field in ("inflight", "waiting", "kv_blocks"))
    return {"remote_stage": str(root), "image_id": IMAGE, "riley_binary_sha256": BINARY, "gpu_uuid": GPU,
            "checkpoint_sha256": report["checkpoint_sha256"], "api_checks": report["api_checks"],
            "original_report_sha256": digest(output / "report.json"),
            "original_mixed_sha256": digest(output / "mixed-stage.json"),
            "server_arguments": report["mixed_stage"]["server_arguments"],
            "gpu_idle_reservations_gib": {"6": 20, "7": 20}, "mixed": mixed}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
