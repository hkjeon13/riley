"""Adapt the verified Coder-14B replayer to matched 4K row budgets."""

import ast
from pathlib import Path

path = Path("arm.py")
source = path.read_text()
changes = (
    ("sha256:052056955ecb930c93d938f79a80272583e1e6ec5fe7eadc60bc2885a61689c6",
     "sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24"),
    ("evidence/w4-branch-anchors-20261124.fpN03ecM/cpu-evidence/report.json",
     "evidence/w4-pointer-syntax-20261124.HgwYVqCf/cpu-evidence/report.json"),
    ("2b58655f358ff80a885a4c97f9b6f5a0afd058b0", "b7ca110fc36ff30a606c254eb87d3a898faa79ea"),
    ("    server = [", """    mode = os.environ["W3_PREFILL_ORDER"]
    rows = {"control": "1024", "candidate": "2048"}[mode]
    buckets = run_profile.BUCKETS if mode == "control" else run_profile.BUCKETS + ",2048"
    server = ["""),
    ('"--max-sequence-tokens", "8192"', '"--max-sequence-tokens", "4096"'),
    ('"--batch-token-budget", "1024", "--prefill-chunk-tokens", "1024"',
     '"--batch-token-budget", rows, "--prefill-chunk-tokens", rows'),
    ('"--batch-shape-buckets", run_profile.BUCKETS', '"--batch-shape-buckets", buckets'),
    ('        "--env", f"RILEY_W3_SHORT_PREFILL_CONTINUATION={int(os.environ[\'W3_PREFILL_ORDER\'] == \'candidate\')}",\n',
     ''),
)
for old, new in changes:
    if source.count(old) != 1:
        raise ValueError(f"expected replay input differs at {old!r}")
    source = source.replace(old, new)
ast.parse(source)
path.write_text(source)
