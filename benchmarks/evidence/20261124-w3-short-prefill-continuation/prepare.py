"""Reuse the frozen RAG and C32 client with the verified current image."""
from pathlib import Path
import ast

path = Path("arm.py")
source = path.read_text()
changes = (
    ('out = OUT / HISTORY / os.environ["W3_PREFILL_ORDER"]',
     'out = OUT / HISTORY / os.environ["W3_REPEAT_PASS"] / os.environ["W3_PREFILL_ORDER"]'),
    ('sha256:01b5c9d4bedfc93284af6dfc9fbe747002327d2818f7164f8d4e77565eaa1a8c',
     'sha256:052056955ecb930c93d938f79a80272583e1e6ec5fe7eadc60bc2885a61689c6'),
    ('evidence/w4-uri-package-20261124.73IfvPre/cpu-evidence/report.json',
     'evidence/w4-branch-anchors-20261124.fpN03ecM/cpu-evidence/report.json'),
    ('8a7c618c1882effcacd49107a0b41ea3fc29d22d','2b58655f358ff80a885a4c97f9b6f5a0afd058b0'),
    ('    if os.environ["W3_PREFILL_ORDER"] == "reserve":\n        server.append("--prefill-aging-reserve-decode")\n',''),
    ('        "--env", "RILEY_W3_EXPORT_OBSERVATIONS=1",\n',
     '        "--env", f"RILEY_W3_SHORT_PREFILL_CONTINUATION={int(os.environ[\'W3_PREFILL_ORDER\'] == \'candidate\')}",\n'),
)
for original, amended in changes:
    if source.count(original) != 1:
        raise ValueError(f"expected frozen runner input differs at {original!r}")
    source = source.replace(original, amended)
ast.parse(source)
path.write_text(source)
