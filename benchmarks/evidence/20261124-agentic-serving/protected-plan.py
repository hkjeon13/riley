"""Select complete archived Coder-14B RAG and short C32 request schedules."""

import hashlib
import json
import sys
from pathlib import Path

archive, output, index = sys.argv[1:]
raw = (Path(archive) / "schedules.json").read_bytes()
if (
    hashlib.sha256(raw).hexdigest()
    != "7921e63069ddd4b58736558af62590b5f340083ee129d25aa54eeefbdf78c116"
):
    raise ValueError("protected request schedules changed")
schedule = json.loads(raw)["schedules"][index]
cells = {key: schedule[key] for key in ("rag-c8-s1-vllm-arrivals", "c32-vllm-arrivals")}
Path(output).write_text(json.dumps({"cells": cells}, indent=2) + "\n")
