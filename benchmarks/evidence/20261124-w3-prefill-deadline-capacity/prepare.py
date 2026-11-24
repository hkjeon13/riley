"""Keep frozen replay inputs and substitute just the source build and opt-in switch."""

import ast
from pathlib import Path


arm = Path("arm.py")
source = arm.read_text()
for old, new in (
    ("sha256:01b5c9d4bedfc93284af6dfc9fbe747002327d2818f7164f8d4e77565eaa1a8c",
     "sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24"),
    ("evidence/w4-uri-package-20261124.73IfvPre/cpu-evidence/report.json",
     "evidence/w4-pointer-syntax-20261124.HgwYVqCf/cpu-evidence/report.json"),
    ("8a7c618c1882effcacd49107a0b41ea3fc29d22d", "b7ca110fc36ff30a606c254eb87d3a898faa79ea"),
    ('    if os.environ["W3_PREFILL_ORDER"] == "reserve":\n        server.append("--prefill-aging-reserve-decode")\n',
     '    candidate = {"control": 0, "candidate": 1}[os.environ["W3_PREFILL_ORDER"]]\n'),
    ('        "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}",\n',
     '        "--env", f"RILEY_SHUTDOWN_METRICS_PATH={SHUTDOWN}",\n'
     '        "--env", f"RILEY_EXPERIMENT_W3_ADMISSION_FIFO={candidate}",\n'),
):
    if source.count(old) != 1:
        raise ValueError(f"expected replay input differs at {old!r}")
    source = source.replace(old, new)
ast.parse(source)
arm.write_text(source)
