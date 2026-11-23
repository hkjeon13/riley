"""Raise only the tested fixed-tile admission ceiling in the diagnostic build."""
from hashlib import sha256
from pathlib import Path
import sys


source = Path(sys.argv[1]) / "crates/riley-server/src/main.rs"
if sha256(source.read_bytes()).hexdigest() != "d0d88d91e06cc1a3fdc7ba221451bdca6217afe5dee999d52f11c39e647a52da":
    raise ValueError("unexpected fixed tile parent source")
text = source.read_text()
start = text.index("    if native_attention_mode == Some(NativeAttentionMode::FixedTileUnsplit)\n")
end = text.index("    if d128_global_merge", start)
guard = text[start:end]
if guard.count("max_active_sequences > 32") != 1 or guard.count("at most 32 active sequences") != 1:
    raise ValueError("expected one fixed-tile admission guard")
guard = guard.replace("max_active_sequences > 32", "max_active_sequences > 48")
guard = guard.replace("at most 32 active sequences", "at most 48 active sequences")
source.write_text(text[:start] + guard + text[end:])
