"""Run the preserved six Coder workloads with the priority switch as the only variable."""
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
for old, new in (
    ('MODES = ("riley-first", "qkv-first", "qkv-second", "riley-second")',
     'MODES = ("control-first", "priority-first", "priority-second", "control-second")'),
    ('    overrides = {}',
     '    env.pop("RILEY_W3_EARLY_PREFILL_PRIORITY", None)\n'
     '    overrides = {"RILEY_W3_QKV1024": "1", "RILEY_W3_REQUEST_TRACE": "1"}'),
    ('        if mode.startswith("qkv-"):\n            overrides["RILEY_W3_QKV1024"] = "1"',
     '        if mode.startswith("priority-"):\n            overrides["RILEY_W3_EARLY_PREFILL_PRIORITY"] = "1"'),
    ('                server_binding = (pid, ticks)',
     '''                if server_binding is None:
                    observed = {}
                    for item in (proc / "environ").read_bytes().split(b"\\0"):
                        key, _, value = item.partition(b"=")
                        if key in {b"RILEY_W3_QKV1024", b"RILEY_W3_REQUEST_TRACE", b"RILEY_W3_EARLY_PREFILL_PRIORITY"}:
                            observed[key.decode()] = value.decode()
                    if observed != overrides:
                        raise RuntimeError("live diagnostic environment differs from plan")
                    receipt["live_diagnostic_environment"] = observed
                server_binding = (pid, ticks)'''),
):
    if source.count(old) != 1:
        raise ValueError(f"original driver changed: {old}")
    source = source.replace(old, new)
compile(source, str(path), "exec")
path.write_text(source)
