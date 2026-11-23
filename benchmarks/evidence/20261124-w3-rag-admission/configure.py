"""Select the existing tail cells and capture their server observation IDs."""

import json
from pathlib import Path


LABELS = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals")


def edit(path, replacements):
    source = path.read_text()
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"expected exactly one pinned anchor in {path}: {before!r}")
        source = source.replace(before, after)
    path.write_text(source)


def configure(stage):
    selected = stage / "schedules.json"
    schedules = json.loads(selected.read_bytes())["schedules"]["7"]
    chosen = {label: schedules[label] for label in LABELS}
    selected.write_text(json.dumps({"schedules": {"7": chosen}}, indent=2) + "\n")

    edit(stage / "arm.py", [
        ('    binary = cpu["binaries"][RILEY]', '    binary = digest(STAGE / "riley-diagnostic")'),
        ('        "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}",',
         '''        "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}",
        "--env", "RILEY_W3_EXPORT_OBSERVATIONS=1",
        "--mount", f"type=bind,source={STAGE / 'riley-diagnostic'},target={RILEY},readonly",'''),
    ])
    edit(stage / "tools/load.py", [
        ('            chunk = json.loads(payload)\n', '''            chunk = json.loads(payload)
            if capture_hashes and "response_id" not in record and isinstance(chunk.get("id"), str):
                record["response_id"] = chunk["id"]
'''),
    ])
    edit(stage / "replay.py", [
        ('"start", "sent_at", "ttft", "e2e")', '"start", "sent_at", "ttft", "e2e", "response_id")'),
        ('"client_timing": {"cpu_affinity":',
         '"warmup_response_ids": [row["response_id"] for row in warmup],\n            "client_timing": {"cpu_affinity":'),
    ])


if __name__ == "__main__":
    configure(Path.cwd())
