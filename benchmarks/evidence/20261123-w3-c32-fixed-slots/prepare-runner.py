"""Use the pinned 1,024-row replay with both fixed-tile slot limits."""
from hashlib import sha256
from pathlib import Path
import sys


path = Path(sys.argv[1])
previous = path.read_bytes()
if sha256(previous).hexdigest() != "671341db32a5cf3513fb65eb6b341d00506997d1c9c493c45db0ef3f59b70aa9":
    raise ValueError("the original 1,024-row runner changed")
source = previous.decode()
old = '''    require((budget, attention, slots) in {("128", "fixed", "32"), ("256", "fixed", "32"),
                                            ("128", "page", "32"), ("128", "page", "48")}
            and workers == "64" and trace == "control", "unexpected attention comparison")'''
new = '''    require(case == "private" and budget in ("128", "256") and attention == "fixed"
            and slots in ("32", "48") and workers == "64" and trace == "control",
            "unexpected fixed-tile comparison")'''
if source.count(old) != 1:
    raise ValueError("the runner case list changed")
source = source.replace(old, new, 1)
compile(source, str(path), "exec")
path.write_text(source)
