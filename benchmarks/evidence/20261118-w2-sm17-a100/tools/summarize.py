"""Summarize the W2 baseline and paired event-publication screen."""
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import sys

root = Path(sys.argv[1])
arms = ("off1", "host", "nsys2", "off2", "paired-off1", "event1", "event2", "paired-off2")
entries = {}
for arm in arms:
    directory = root / arm
    if not (directory / "receipt.json").exists():
        continue
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt["status"] != "passed":
        entries[arm] = {"status": receipt["status"], "error": receipt.get("error")}
        continue
    raw = (directory / "load.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["load_sha256"]:
        raise ValueError(f"{arm} load summary hash mismatch")
    load = json.loads(raw)
    idle = json.loads((directory / "idle-after.json").read_text())
    if (idle["phase"] != "idle" or idle["allocation_bytes"] != 20 * 1024**3
            or idle["pid"] != receipt["holder_pid"] or idle["start_ticks"] != receipt["holder_start_ticks"]):
        raise ValueError(f"{arm} holder restoration mismatch")
    entries[arm] = {**{k: load[k] for k in (
        "requests", "ok", "errors", "rejected_429", "warmup_requests", "warmup_ok",
        "warmup_errors", "completion_tokens_mean", "output_tok_s", "ttft_p50_ms", "ttft_p95_ms",
        "tpot_p50_ms", "tpot_p95_ms", "e2e_p95_ms", "itl_p99_ms")},
        "status": "passed", "nvml_samples": receipt["nvml_samples"], "peak_process_memory_mib": receipt["max_process_memory_mib"]}

trace = []
for line in (root / "host/server.log").read_text().splitlines():
    if not line.startswith("HOSTSTEP us/step ") or "captured=0 eager=0" not in line:
        continue
    row = {k: float(v) for k, v in re.findall(r"([a-z+]+)=(\d+(?:\.\d+)?)", line)}
    trace.append(row)
fuller = [r for r in trace if r["wait"] >= 5500]
serial_keys = ("cancel+commands", "plan", "prepare", "pack", "upload", "download",
               "sample", "commit", "events", "publish")
serial = [sum(r[k] for k in serial_keys) for r in fuller]
doc = {"arms": entries, "host_trace": {
    "windows": len(trace), "selected_windows": len(fuller),
    "selection": "128-step windows with zero capture/eager counts and mean WAIT >= 5500 us, a proxy for sustained load. Window aggregates do not bind to 32-row geometry.",
    "serial_segments_excluding_submit_wait": serial_keys,
    "serial_us_median": st.median(serial), "serial_us_range": [min(serial), max(serial)],
    "wait_us_median": st.median(r["wait"] for r in fuller),
    "median_us_per_segment": {k: st.median(r[k] for r in fuller) for k in fuller[0]},
}}
if all(entries.get(a, {}).get("status") == "passed" for a in ("off1", "off2", "host")):
    for metric in ("output_tok_s", "tpot_p50_ms"):
        control = st.mean(entries[a][metric] for a in ("off1", "off2"))
        doc.setdefault("host_trace_vs_off_percent", {})[metric] = 100 * (entries["host"][metric] / control - 1)
if all(entries.get(a, {}).get("status") == "passed" for a in ("paired-off1", "paired-off2", "event1", "event2")):
    for metric in ("output_tok_s", "tpot_p50_ms", "tpot_p95_ms", "ttft_p50_ms", "ttft_p95_ms", "e2e_p95_ms"):
        baseline = st.mean(entries[a][metric] for a in ("paired-off1", "paired-off2"))
        optin = st.mean(entries[a][metric] for a in ("event1", "event2"))
        doc.setdefault("after_submit_vs_immediate", {})[metric] = {
            "baseline": baseline, "opt_in": optin, "percent_change": 100 * (optin / baseline - 1)}
print(json.dumps(doc, indent=2))
