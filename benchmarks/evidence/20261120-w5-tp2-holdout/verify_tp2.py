"""Verify the two-device packaged run and export only the parity receipt."""

import json
from pathlib import Path
import sys

from verify_smoke import BINARY, GPU, IMAGE, digest, read, reservations


GPU6 = "GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7"
CHECKPOINT = "c535a624b44249e514a8facd9c4cb7ccdfdf4a17f3b556e27c2c981fd16c9365"


def verify(root):
    reservations(root, "before")
    reservations(root, "after")
    assert (root / "exit").read_text().strip() == "0"
    assert (root / "peer-link.txt").read_text().strip() == "NV12"
    output = root / "gpu-evidence"
    report = read(output / "report.json")
    assert (report["status"], report["image_id"], report["gpu_uuid"], report["riley_binary_sha256"]) == (
        "passed", IMAGE, GPU6, BINARY)
    assert report["checkpoint_sha256"] == CHECKPOINT
    assert report["checkpoint_source_model"] == "Qwen/Qwen2.5-32B-Instruct"
    assert report["api_checks"] == 40 and report["shutdown_exit"] == 0 and report["tool_checks_enabled"] is True
    tp = report["tensor_parallel"]
    assert tp["gpu_uuids"] == [GPU6, GPU]
    assert tp["server_arguments"] == [
        "--inference-profile", "native-bf16-varlen-v1", "--tensor-parallel-devices", "0,1",
        "--max-active-sequences", "32", "--max-sequence-tokens", "8192", "--max-output-tokens", "256",
        "--kv-blocks", "2048",
    ]
    for name, expected in report["artifacts_sha256"].items():
        assert (output / name).parent == output and digest(output / name) == expected
    holdout = read(output / "tp-holdout.json")
    assert holdout == tp["holdout"] and holdout["workload_sha256"] == digest(root / "holdout.json.gz")
    assert holdout["status"] == "passed" and holdout["output_token_limit"] == 32
    assert sorted(item["prompt_tokens"] for item in holdout["cases"]) == [
        255,256,257,511,512,513,1023,1024,1025,7836,7917,
    ]
    assert holdout["modes"]["c1"] == {"requests": 11, "observed_peak_active": 1}
    for clients, requests in ((8, 11), (32, 32)):
        mode = holdout["modes"][f"c{clients}"]
        assert mode["requests"] == mode["exact_matches"] == requests and 1 < mode["observed_peak_active"] <= clients
    return {"status": "passed", "remote_stage": str(root), "image_id": IMAGE, "riley_binary_sha256": BINARY,
            "checkpoint_sha256": CHECKPOINT, "checkpoint_source_revision": report["checkpoint_source_revision"],
            "api_checks": report["api_checks"], "tensor_parallel": tp, "gpu_link": "NV12",
            "gpu_idle_reservations_gib": {"6": 20, "7": 20}, "original_report_sha256": digest(output / "report.json")}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
