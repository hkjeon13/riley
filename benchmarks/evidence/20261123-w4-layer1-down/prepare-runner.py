"""Reuse the verified KV campaign driver with layer-1 projection rows and controls."""
from hashlib import sha256
from pathlib import Path
import sys

from prepare import replace_once


def generate(source_path, module_path):
    raw = source_path.read_bytes()
    expected = Path(__file__).with_name("parent-inputs.sha256").read_text().split()[0]
    if sha256(raw).hexdigest() != expected:
        raise ValueError("the KV history trial differs from the corrected parent")
    source = raw.decode()
    source = replace_once(source, "    expected[LOGITS] = 304128", """    expected[LOGITS] = 304128
    expected.update({(row, "1"): ROW_BYTES for row in ("mlp_input", "down_native", "down_effective", "layer2_input")})
    expected["mlp_gated", "1"] = 18944 * 2""")
    source = replace_once(source, "def respond(out, name, address, case, ids, donor=None, swap=False):",
                          "def respond(out, name, address, case, ids, donor=None, swap=False, down=False):")
    source = replace_once(source,
                          '            (pending / "past-kv.bf16").write_bytes(b"".join(donor[key, "1"] for key in HISTORY))',
                          '''            (pending / "past-kv.bf16").write_bytes(b"".join(donor[key, "1"] for key in HISTORY))
        if down:
            (pending / "layer1-down.bf16").write_bytes(donor["down_native", "1"])''')
    source = replace_once(source, '''        respond(out, "self", address, case, ids, own, swap=True)
        respond(out, "boundary", address, case, ids, donor)
        respond(out, "history", address, case, ids, donor, swap=True)''',
                          '''        respond(out, "self", address, case, ids, own, swap=True, down=True)
        respond(out, "history", address, case, ids, donor, swap=True)
        respond(out, "down", address, case, ids, donor, swap=True, down=True)''')
    # Keep the startup, response, and trace readers. This stage has different controls.
    verify_start = source.index("\ndef verify_role(")
    source = source[:verify_start] + '''

def stop(_sig, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stage, build, role = sys.argv[1:]
    run(Path(stage), Path(build), role)
'''
    module_path.write_text(source)


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
