"""Validate the diagnostic image against the original image and strict W5 workload."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

BASE_REVISION = "d3d8fe6e5b1c528bff51c0008cdc24544c99f862"
BASE_ARCHIVE = "cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5"
PATCH = "28f0ef9ed0e5e2e81b5f82d68f99554977c461ca3cfe91ea929b5becb38fd520"
RILEY = "/opt/riley/bin/riley"
RUNTIME_FIELDS = ("checkpoint_sha256", "gpu_uuid", "soak_server_arguments")


def compare_image(stage, base, digest, read, require):
    candidate_dir = stage / "cpu-evidence"
    base_dir = base / "cpu-evidence"
    receipt, original = (read(folder / "report.json") for folder in (candidate_dir, base_dir))
    archive = digest(stage / "source.tar")
    require(original["source_archive_sha256"] == BASE_ARCHIVE and receipt["source_archive_sha256"] == archive
            and archive != BASE_ARCHIVE and (stage / "source-archive.sha256").read_text().strip() == archive
            and (stage / "image.id").read_text().strip() == receipt["image_id"]
            and receipt["image_id"] != original["image_id"], "wrong diagnostic image archive or ID")
    for field in ("schema_version", "status", "source_revision", "version", "preset_count", "distribution", "checks",
                  "cpu_cli_driver_stub_sha256", "driver_dependency"):
        require(receipt[field] == original[field], f"CPU image evidence disagrees on {field}")
    require(receipt["source_revision"] == BASE_REVISION and receipt["status"] == "passed",
            "expected a successful CPU check of an image based on the approved revision")
    inspect, = read(candidate_dir / "image-inspect.json")
    labels = inspect["Config"]["Labels"]
    require(labels["io.riley.diagnostic-patch-sha256"] == digest(stage / "reserve-validated-body.patch.gz") == PATCH,
            "diagnostic patch is not bound to the built image")
    inventories = [dict(line.split(None, 1)[::-1] for line in (folder / "runtime.sha256").read_text().splitlines())
                   for folder in (base_dir, candidate_dir)]
    control_files, candidate_files = inventories
    require(len(control_files) == 21 and control_files.keys() == candidate_files.keys()
            and {name for name in control_files if control_files[name] != candidate_files[name]} == {RILEY}
            and control_files[RILEY] == original["binaries"][RILEY] and candidate_files[RILEY] == receipt["binaries"][RILEY],
            "runtime changed outside the diagnostic Riley executable")
    return {"base_revision": BASE_REVISION, "patch_sha256": PATCH, "base_source_archive_sha256": BASE_ARCHIVE,
            "candidate_source_archive_sha256": archive, "base_image_id": original["image_id"],
            "candidate_image_id": receipt["image_id"], "candidate_binary_sha256": candidate_files[RILEY],
            "base_binary_sha256": control_files[RILEY], "unchanged_runtime_files": 20,
            "candidate_cpu_report_sha256": digest(candidate_dir / "report.json")}


def compare_mode(stage, mode, image, controls, read_soak, digest, read, require):
    arm = stage / mode
    base_path = stage.parent / controls[mode]
    candidate_path = arm / "gpu-evidence"
    original, base_soak = read_soak(base_path)
    receipt, candidate_soak = read_soak(candidate_path)
    require(all(original[field] == receipt[field] for field in RUNTIME_FIELDS)
            and original["mixed_soak"]["mode"] == receipt["mixed_soak"]["mode"] == mode
            and base_soak["counts"] == candidate_soak["counts"], f"{mode}: workload comparison changed")
    for prefix, record in (("base", original), ("candidate", receipt)):
        require(record["image_id"] == image[f"{prefix}_image_id"]
                and record["riley_binary_sha256"] == image[f"{prefix}_binary_sha256"]
                and record["source_archive_sha256"] == image[f"{prefix}_source_archive_sha256"]
                and record["source_revision"] == BASE_REVISION, f"{mode}: wrong {prefix} image or archive")
    require(receipt["cpu_report_sha256"] == image["candidate_cpu_report_sha256"]
            and int((arm / "exit").read_text()) == int(receipt["status"] == "failed"),
            f"{mode}: CPU report and GPU verdict differ")
    pre, post = (read(arm / f"idle-{point}.json") for point in ("before", "after"))
    require(all(pre[key] == post[key] for key in pre if key != "updated_ns")
            and post["physical_index"] == 6 and post["pid"] == 2431228 and post["phase"] == "idle"
            and post["allocation_bytes"] == post["torch_reserved_bytes"] == 20 * 1024**3
            and not post["other_processes"], "original GPU 6 reservation was not restored")
    return {
        "control_report_sha256": digest(base_path / "report.json"),
        "candidate_report_sha256": digest(candidate_path / "report.json"),
        "candidate_soak_sha256": digest(candidate_path / "soak.json"),
        "identical_outcome_counts": True, "counts": candidate_soak["counts"],
        "control_rss": original["mixed_soak"]["rss"], "candidate_rss": receipt["mixed_soak"]["rss"],
        "cache_mode": mode, "api_checks": receipt["api_checks"], "candidate_failed_checks": receipt["mixed_soak"]["failed_checks"],
        "control_late_p95_ms": {name: base_soak["latency_windows"]["late"][name] for name in ("ttft_ms_p95", "e2e_ms_p95")},
        "candidate_late_p95_ms": {name: candidate_soak["latency_windows"]["late"][name] for name in ("ttft_ms_p95", "e2e_ms_p95")},
        "same_20gib_reservation_restored": True, "candidate_status": receipt["status"],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve().parent / "w5-tcache-only-20261119.G3IuT0Wh"))
    from verify_periodic_trim import CONTROLS, digest, read, read_soak, require

    image = compare_image(args.stage, args.base, digest, read, require)
    result = {"scope": "Source-bound isolated W5 HTTP buffer diagnostic.", "image": image}
    if not args.cpu_only:
        result["modes"] = {mode: compare_mode(args.stage, mode, image, CONTROLS, read_soak, digest, read, require)
                           for mode in ("cache-on", "cache-off") if (args.stage / mode / "exit").is_file()}
    print(json.dumps(result, indent=2, allow_nan=False))
