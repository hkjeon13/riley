"""Keep measured prompts intact while bounding the page allocation to 2304 tokens."""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
anchor = '    argv[argv.index("--max-active-sequences") + 1] = slots'
if source.count(anchor) != 1:
    raise ValueError("native runner argument anchor changed")
source = source.replace(anchor, anchor + '\n    argv[argv.index("--max-sequence-tokens") + 1] = "2304"')
compile(source, str(path), "exec")
path.write_text(source)
