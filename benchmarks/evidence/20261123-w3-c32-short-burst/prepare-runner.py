"""Select the short-only budget policy without changing the archived server arguments."""
from hashlib import sha256
from pathlib import Path
import sys


path = Path(sys.argv[1])
raw = path.read_bytes()
if sha256(raw).hexdigest() != "7fea0d955360bff811013718e4b236e258706ed45f1df1a1b19ecac357eb3371":
    raise ValueError("unexpected steady-256 replay driver")
source = raw.decode()
for old, new in (
    ("workers, trace, decode_cap, prompt_floor):", "workers, trace, short_limit):"),
    ('''    require((budget, decode_cap, prompt_floor) in {("256", "32", "512"),
                                                       ("256", "8", "64"), ("256", "16", "64")}
            and workers == "64" and trace == "control", "unexpected burst comparison")
    arm = f"{case}-s{budget}-d{decode_cap}-p{prompt_floor}-gpu{index}"''',
     '''    require(case == "private" and budget == "256" and workers == "64"
            and trace == "control" and short_limit in ("0", "8", "16"), "unexpected short-only burst comparison")
    short_env = f"RILEY_W3_SHORT_BURST_DECODE_LIMIT={short_limit}"
    arm = f"{case}-short-d{short_limit}-gpu{index}"'''),
    ('''    argv[argv.index("--prefill-burst-max-decode-rows") + 1] = decode_cap
    argv[argv.index("--prefill-burst-min-prompt-tokens") + 1] = prompt_floor''',
     '''    require(argv[argv.index("--prefill-burst-max-decode-rows") + 1] == "32"
            and argv[argv.index("--prefill-burst-min-prompt-tokens") + 1] == "512",
            "the existing long-prefill policy changed")'''),
    ('"--env", "RILEY_W3_EXPORT_OBSERVATIONS=1", "--env", trace_env,',
     '"--env", "RILEY_W3_EXPORT_OBSERVATIONS=1", "--env", trace_env, "--env", short_env,'),
    ('f"RILEY_W3_HTTP_WORKERS={workers}", trace_env}', 'f"RILEY_W3_HTTP_WORKERS={workers}", trace_env, short_env}'),
    ('"burst_decode_cap": int(decode_cap), "burst_prompt_floor": int(prompt_floor),',
     '"burst_decode_cap": 32, "burst_prompt_floor": 512, "short_burst_decode_limit": int(short_limit),'),
    ("*sys.argv[4:11]", "*sys.argv[4:10]"),
):
    if source.count(old) != 1:
        raise ValueError(f"unexpected driver anchor: {old}")
    source = source.replace(old, new, 1)
compile(source, str(path), "exec")
path.write_text(source)
