"""Verify paired timing modes while reusing the archived short-burst checks."""
from pathlib import Path
import subprocess
import sys


here = Path(__file__).resolve().parent
parent = here.parent / "20261123-w3-c32-short-burst"
composed = subprocess.check_output([sys.executable, "-B", str(parent / "compose-verifier.py")], text=True)
header, marker, source = composed.partition(
    '\n"""Verify the bounded burst comparisons against the shipped output and 256-row control."""\n'
)
if not marker:
    raise ValueError("expected the archived verifier after its reference modules")
for old, new in (
    ('BASE = "short-d0"\nCANDIDATES = ("short-d8", "short-d16")\nORDER = (BASE, *CANDIDATES)',
     '''BASE = "short-d0-control"
ORDER = (BASE, "short-d8-control", "short-d8-admission", "short-d0-admission")
PAIRS = ((BASE, ("short-d8-control", "short-d0-admission")),
         ("short-d8-control", ("short-d8-admission",)),
         ("short-d0-admission", ("short-d8-admission",)))'''),
    ('''    require(digest(build / "target/release/riley") == build_checks["artifacts"]["target/release/riley"]
            and digest(build / "scheduler-test.log") == build_checks["artifacts"]["scheduler-test.log"],
            "measured binary or CPU proof changed")
    require("1 passed; 0 failed" in (build / "scheduler-test.log").read_text(),
            "short-only scheduler proof did not pass")''',
     '''    require(digest(build / "target/release/riley") == build_checks["artifacts"]["target/release/riley"],
            "measured binary changed")'''),
    ("    runs, reservations, differences = {}, {}, {}",
     "    runs, reservations, differences, phase_changes, ordinary_changes = {}, {}, {}, {}, {}"),
    ("        raw, arms = {}, {}", "        raw, arms, traces, ordinary = {}, {}, {}, {}"),
    ('            short_limit = int(layout.removeprefix("short-d"))',
     '''            label, mode = layout.rsplit("-", 1)
            short_limit = int(label.removeprefix("short-d"))'''),
    ('                    and receipt["stage_trace"] is False and receipt["short_burst_decode_limit"] == short_limit,',
     '''                    and receipt["stage_trace"] is (mode == "admission")
                    and receipt["trace_mode"] == mode and receipt["short_burst_decode_limit"] == short_limit,'''),
    ('            canonical_control = layout == BASE', '            canonical_control = short_limit == 0'),
    ('''                    and not any(line.startswith(("W3-ACTIVATION ", "W3-PREFILL-PHASE ", "W3-DROPPED "))
                                for line in log.splitlines()), "attention or trace path changed")
            timings = bind_observations(log, sample)["short"]''',
     '''                    and not any(line.startswith(("W3-ACTIVATION ", "W3-PREFILL-PHASE "))
                                for line in log.splitlines()), "full trace or attention path changed")
            observations = bind_observations(log, sample)
            timings = observations["short"]
            ordinary[layout] = timings
            if mode == "admission":
                traces[layout], arm["engine_trace"] = engine_trace(
                    log, sample, observations, arm["startup_requests_in_log"])
            else:
                require(not any(line.startswith(("W3-REQUEST-PHASE ", "W3-ENGINE-SPAN ", "W3-DROPPED "))
                                for line in log.splitlines()), "control unexpectedly recorded timing")'''),
    ('                "short_burst_decode_limit": short_limit,',
     '                "short_burst_decode_limit": short_limit, "trace_mode": mode,'),
    ('        for before, others in ((BASE, CANDIDATES), (CANDIDATES[0], CANDIDATES[1:])):',
     '        for before, others in PAIRS:'),
    ('    return {"status": "passed", "remote_evidence": str(stage),',
     '''        phase_changes[gpu] = phase_pairs(traces["short-d0-admission"], traces["short-d8-admission"])
        ordinary_changes[gpu] = {
            mode: ordinary_phase_pairs(ordinary[f"short-d0-{mode}"], ordinary[f"short-d8-{mode}"])
            for mode in ("control", "admission")
        }
    return {"status": "passed", "remote_evidence": str(stage),'''),
    ('"reservations": reservations, "order": timeline, "runs": runs, "comparisons": differences}',
     '''"reservations": reservations, "order": timeline, "runs": runs, "comparisons": differences,
            "paired_trace_phases": phase_changes, "paired_ordinary_phases": ordinary_changes}'''),
    ('\nif __name__ == "__main__":\n', '\n' + (here / "trace-phases.py").read_text() + '\n\nif __name__ == "__main__":\n'),
):
    if source.count(old) != 1:
        raise ValueError(f"expected one archived verifier anchor: {old}")
    source = source.replace(old, new, 1)
result = header + marker + source
compile(result, "short-burst-admission-verifier", "exec")
print(result, end="")
