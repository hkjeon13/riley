"""Apply the archived candidate signatures to all native-profile row buckets."""
import json
from pathlib import Path
import subprocess
import sys


stage, source = map(Path, sys.argv[1:])
plans = json.loads((source / "verified.json").read_bytes())["operators"]
choices = (
    ("attention-31", "o_proj", "heuristic-0", 3584),
    ("down-13-unsplit", "down_proj", "heuristic-4", 18944),
    ("down-13-split64", "down_proj", "heuristic-6", 18944),
    ("fp32-reduction", "down_proj", "fixed-reduction", 18944),
)
for name, operator, label, k in choices:
    config = plans[operator]["plans"][label]
    fields = ("id", "tile", "stages", "split_k", "reduction", "cta_swizzling", "custom_option", "flags")
    with (stage / f"{name}.log").open("xb") as log:
        subprocess.run([str(stage / "bucket_probe"), str(source / operator), "3584", str(k),
                        *(str(config[field]) for field in fields), label], check=True, stdout=log, timeout=120)
