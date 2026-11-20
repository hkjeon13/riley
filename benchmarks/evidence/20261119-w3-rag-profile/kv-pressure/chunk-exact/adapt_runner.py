"""Adapt the verified KV runner to probe the pinned W4 prompts."""
from pathlib import Path
import sys


def replace(path, pairs):
    text = path.read_text()
    for before, after in pairs:
        if text.count(before) != 1:
            raise ValueError(f"unexpected runner source in {path.name}: {before!r}")
        text = text.replace(before, after)
    path.write_text(text)


tools = Path(sys.argv[1])
replace(tools / "run_profile_retry.py", [
    ('from profile_workload import save', 'from profile_workload import save\nfrom exact_batch import ARMS'),
    ('MODES = ("riley-first", "riley-second")', 'MODES = tuple(ARMS)'),
    ('    if mode == "riley-second":\n        expected_blocks = 768\n        blocks = ["--kv-blocks", str(expected_blocks)]',
     '    chunk, expected_blocks = ARMS[mode]\n    blocks = ["--kv-blocks", str(expected_blocks)]'),
    ('"--prefill-chunk-tokens", "1024"', '"--prefill-chunk-tokens", str(chunk)'),
    ('"check_token_ids": False, "cells": cells(model_key, mode)',
     '"check_token_ids": True, "cells": {}'),
    ('        receipt["results"] = results',
     '        results["token-id-probe"] = file_sha256(out / "token-id-probe.json")\n'
     '        receipt["results"] = results'),
])
replace(tools / "profile_workload.py", [
    ('import json\n', 'import json\nfrom exact_batch import probe\n'),
    ('token_id_probe(plan["address"])', 'probe(plan["address"])'),
])
