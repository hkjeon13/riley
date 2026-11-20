"""Verify the MHA smoke with the checkpoint's actual tool-support contract."""

import json
from pathlib import Path
import sys

from verify_smoke import BINARY, GPU, IMAGE, digest, read, reservations


MODEL = "SmolLM2-1.7B-Instruct"
CHECKPOINT = "592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753"


def summarize(root, prior):
    reservations(root, "before")
    reservations(root, "after")
    previous = read(prior / "verified-smoke.json")["results"][MODEL]
    assert previous["status"] == "execution_failed" and previous["api_checks_passed"] == 33
    terminal = int((root / "exit").read_text())
    arm = root / "gpu-evidence"
    summary = {"remote_stage": str(root), "image_id": IMAGE, "binary_sha256": BINARY, "gpu_uuid": GPU,
               "checkpoint_sha256": CHECKPOINT, "model": MODEL, "gpu_idle_reservations_gib": {"6": 20, "7": 20},
               "prior_result_sha256": digest(prior / "verified-smoke.json"), "exit": terminal}
    api = read(arm / "api.json")["results"] if (arm / "api.json").is_file() else []
    summary["api_checks"] = len(api)
    summary["api_checks_passed"] = sum(row["passed"] is True for row in api)
    summary["api_failed_check_names"] = [row["name"] for row in api if row["passed"] is not True]
    path = arm / "report.json"
    if not path.is_file():
        assert terminal != 0
        return {**summary, "status": "execution_failed", "driver_log_sha256": digest(root / "driver.log")}
    report = read(path)
    assert (report["checkpoint_sha256"], report["image_id"], report["gpu_uuid"], report["riley_binary_sha256"]) == (
        CHECKPOINT, IMAGE, GPU, BINARY)
    assert report["api_checks"] == summary["api_checks"] == summary["api_checks_passed"] == 39
    assert report["shutdown_exit"] == 0 and report["tool_checks_enabled"] is False
    for name, expected in report["artifacts_sha256"].items():
        assert (arm / name).parent == arm and digest(arm / name) == expected
    raw = read(arm / "smoke.json")
    smoke = report["serving_smoke"]
    assert smoke["seconds"] == raw["seconds"] == 45 and smoke["rate"] == raw["rate"] == 8
    assert smoke["expect_tools_unsupported"] is raw["expect_tools_unsupported"] is True
    assert smoke["counts"] == raw["counts"] and smoke["checks"] == raw["checks"]
    assert smoke["counter_changes"] == raw["counter_changes"]
    assert smoke["server_arguments"][-4:] == ["--kv-blocks", "4096", "--prefix-cache-blocks", "128"]
    assert report["status"] == smoke["status"] == ("passed" if terminal == 0 else "failed")
    return {**summary, "status": report["status"], "smoke": smoke, "report_sha256": digest(path),
            "smoke_sha256": digest(arm / "smoke.json"), "checkpoint_source_revision": report["checkpoint_source_revision"]}


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, allow_nan=False))
