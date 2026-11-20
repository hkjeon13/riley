"""Enable the existing Nsight path for the two fixed request-stage arms."""
from pathlib import Path
import sys

script = Path(sys.argv[1])
text = script.read_text()
stripped = 'MODES = ("riley-first", "qkv-first", "qkv-second", "riley-second")'
assert text.count(stripped) == 1
text = text.replace(stripped, 'MODES = ("riley-first", "qkv-first")', 1)
assert text.count('if mode == "profile":') == 3
text = text.replace('if mode == "profile":', 'if mode in MODES:')
assert text.count('"profile": False, "check_token_ids": False') == 1
text = text.replace('"profile": False, "check_token_ids": False',
                    '"profile": True, "check_token_ids": False')
compile(text, str(script), "exec")
script.write_text(text)
