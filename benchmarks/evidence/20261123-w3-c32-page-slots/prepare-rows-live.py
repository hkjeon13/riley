"""Reduce the prepared page row capacity while retaining the 8192-token context."""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
anchor = '    argv[argv.index("--max-active-sequences") + 1] = slots'
change = '''
    for option in ("--batch-token-budget", "--prefill-chunk-tokens"):
        argv[argv.index(option) + 1] = "192"
    bucket_arg = argv.index("--batch-shape-buckets") + 1
    buckets = [int(row) for row in argv[bucket_arg].split(",") if int(row) <= 192]
    require(buckets[-1] == 192, "expected a 192-row bucket")
    argv[bucket_arg] = ",".join(map(str, buckets))
    argv += ["--prefill-plateau-rows", "128"]'''
if source.count(anchor) != 1:
    raise ValueError("native runner argument anchor changed")
source = source.replace(anchor, anchor + change)
compile(source, str(path), "exec")
path.write_text(source)
