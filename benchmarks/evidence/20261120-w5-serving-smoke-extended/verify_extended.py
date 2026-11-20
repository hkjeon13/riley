"""Bind the larger-model smoke results to their preset KV pools and model geometry."""

import json
from pathlib import Path
import sys

from verify_smoke import BINARY, GPU, IMAGE, check, read


def summarize(root):
    results = {}
    for line in (root / "models.tsv").read_text().splitlines():
        model, checkpoint, blocks = line.split()
        result = check(root / model, checkpoint)
        config = read(root / "model-roots" / model / "config.json")
        if "smoke" in result:
            args = result["smoke"]["server_arguments"]
            assert args[-4:] == ["--kv-blocks", blocks, "--prefix-cache-blocks", "128"]
        results[model] = {
            **result,
            "kv_blocks_requested": int(blocks),
            "attention_heads": config["num_attention_heads"],
            "kv_heads": config["num_key_value_heads"],
        }
    return {"remote_stage": str(root), "image_id": IMAGE, "binary_sha256": BINARY, "gpu_uuid": GPU,
            "idle_reservations_gib": {"6": 20, "7": 20}, "results": results}


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, allow_nan=False))
