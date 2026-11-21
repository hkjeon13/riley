"""Qualify the opt-in allocator with the same packed Riley checker and workload."""

import json
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import check_distribution_gpu as checker
from current_binding import PROFILE, process_proof
from verify import MODES, MODEL, RILEY, check_reservations, digest, read, require, soak, verify_accuracy
from verify_allocator_image import LIB, LIB_SHA, image_config, inspect


def bind(root, parent_path):
    parent, variant = read(parent_path), read(root / "verified-image.json")
    require(parent["status"] == variant["status"] == "passed"
            and parent["image_id"] == variant["parent_image_id"]
            and digest(parent_path) == variant["parent_cpu_report_sha256"]
            and all(parent[key] == variant[key] for key in ("source_revision", "source_archive_sha256", "binaries"))
            and variant["added_files_sha256"][LIB] == LIB_SHA, "allocator package differs from the current parent")
    return {**parent, "image_id": variant["image_id"]}, parent["image_id"]


def run(root, parent_path, base, scope):
    spec, parent = bind(root, parent_path)
    adapter = root / "gpu-checker-input.json"
    require(read(adapter) == spec, "synthetic CPU receipt differs from the proven variant")
    original_config, original_ready, original_docker = checker.check_config, checker.wait_ready, checker.docker
    before = root / f"{scope}-allocator-before.json"

    def config(documents, image, revision, archive):
        variant, = documents
        require(variant["Id"] == image == spec["image_id"], "wrong allocator image")
        stock = inspect(parent)
        original_config([stock], parent, revision, archive)
        image_config(stock, variant)

    def ready(container, port):
        result = original_ready(container, port)
        before.write_text(json.dumps(process_proof(container, spec["image_id"])) + "\n")
        return result

    def docker(*args):
        if args[0] == "stop":
            proof = process_proof(args[-1], spec["image_id"])
            require(proof == read(before), "allocator process changed before shutdown")
            (root / f"{scope}-allocator-after.json").write_text(json.dumps(proof) + "\n")
        return original_docker(*args)

    if scope == "accuracy":
        golden = base / "src/benchmarks/evidence"
        holdout = golden / "20261119-w4-holdout/holdout.json.gz"
        options = {"accuracy_baseline": golden / "20261119-w4-long/riley/capture.json.gz",
                   "accuracy_holdout": (holdout, golden / "20261119-w4-holdout/riley/capture.json.gz"),
                   "structured_holdout": holdout}
        gpu = MODES["cache-off"]
    else:
        options = {"mixed_soak": scope, "require_prefix_cache_metrics": True}
        gpu = MODES[scope]
    with (patch.object(checker, "check_config", config), patch.object(checker, "wait_ready", ready),
          patch.object(checker, "docker", docker)):
        return checker.check_gpu(adapter, gpu, base / "model", MODEL, root / scope, True, **options)


def verify(root, parent_path):
    spec, parent = bind(root, parent_path)
    adapter = root / "gpu-checker-input.json"
    require(read(adapter) == spec, "synthetic CPU receipt differs from the proven variant")
    cpu_hash = digest(adapter)
    modes = {label: soak(root, label, gpu, spec, cpu_hash) for label, gpu in MODES.items()}
    accuracy = verify_accuracy(root, spec, cpu_hash)
    reservations = check_reservations(root, "before")
    require(reservations == check_reservations(root, "after-soaks") == check_reservations(root, "after"),
            "allocator tests changed the standing GPU reservations")
    process_samples = {}
    for label in (*MODES, "accuracy"):
        before = root / f"{label}-allocator-before.json"
        proof = read(before)
        require(proof == read(root / f"{label}-allocator-after.json") and proof["image_id"] == spec["image_id"]
                and proof["library_sha256"] == LIB_SHA and proof["malloc_conf"] == PROFILE,
                f"{label} allocator process proof differs")
        if label in modes:
            rss = read(root / label / "soak.json")["rss_source"]
            require(all(proof[key] == rss[key] for key in ("container_id", "host_pid")),
                    f"{label} RSS belongs to a different process")
        process_samples[label] = digest(before)
    soaks = {"source_revision": spec["source_revision"], "image_id": spec["image_id"], "binary_sha256": spec["binaries"][RILEY],
             "checkpoint_sha256": MODEL, "duration_seconds_per_mode": 1800, "requests_per_second_per_mode": 8,
             "release_gate": "passed" if all(row["status"] == "passed" for row in modes.values()) else "failed", "modes": modes}
    return {"status": "passed", "source_revision": spec["source_revision"], "remote_evidence": str(root),
            "image_id": spec["image_id"], "parent_image_id": parent, "checkpoint_sha256": MODEL,
            "riley_binary_sha256": spec["binaries"][RILEY], "parent_cpu_report_sha256": digest(parent_path),
            "allocator_cpu_report_sha256": digest(root / "verified-image.json"),
            "allocator_library_sha256": LIB_SHA, "allocator_settings": PROFILE,
            "allocator_process_proof_sha256": process_samples, "soaks": soaks,
            "accuracy": accuracy, "reservations": reservations}


if __name__ == "__main__":
    action, folder, cpu, *args = sys.argv[1:]
    root, parent_path = Path(folder), Path(cpu)
    if action == "run":
        def interrupted(_signal, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        raise SystemExit(run(root, parent_path, Path(args[0]), args[1]))
    report = bind(root, parent_path)[0] if action == "bind" else verify(root, parent_path)
    print(json.dumps(report, indent=2, allow_nan=False))
