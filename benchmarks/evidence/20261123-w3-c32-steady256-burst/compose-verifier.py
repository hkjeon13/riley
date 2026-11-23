"""Reuse the prior burst verifier with a single 256-row steady budget."""
from hashlib import sha256
from pathlib import Path


here = Path(__file__).resolve().parent
parent = here.parent / "20261123-w3-c32-burst-thresholds"


def pinned(name, expected):
    raw = (parent / name).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"prior burst verifier input changed: {name}")
    return raw.decode()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"prior burst verifier anchor changed: {old}")
    return text.replace(old, new, 1)


old_fetch = pinned("fetch-summary.sh", "4ca68d0963756b0a3b1c4b887992a10b56b10c78a860cd6c67fbaf15bbc5276f")
loader = old_fetch.split("<<'PY' |\n", 1)[1].split("\nPY\n", 1)[0]
loader = replace_once(loader, 'print((here / "verify.py").read_text())', "")
verifier = pinned("verify.py", "5fd7cd4396211e507adba2033d4afc27097c692348e46cf60e2ed6632e8390b0")

for old, new in (
    ('LOW_STEADY = "s128-d32-p512"\n', ""),
    ('CANDIDATES = ("s128-d8-p64", "s128-d16-p64")', 'CANDIDATES = ("s256-d8-p64", "s256-d16-p64")'),
    ("ORDER = (BASE, LOW_STEADY, *CANDIDATES)", "ORDER = (BASE, *CANDIDATES)"),
    ("len(timeline) == 16", "len(timeline) == 2 * sum(map(len, ORDERS.values()))"),
    ("canonical_control = layout in (BASE, LOW_STEADY)", "canonical_control = layout == BASE"),
    ("    schedules, workloads, canonical = read(stage / ", '    require(digest(stage / "schedules.json") == "a368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0", "schedule drift")\n'
                                                          "    schedules, workloads, canonical = read(stage / "),
    ('''        for before, others in ((BASE, (LOW_STEADY, *CANDIDATES)), (LOW_STEADY, CANDIDATES),
                               (CANDIDATES[0], CANDIDATES[1:])):''',
     '''        for before, others in ((BASE, CANDIDATES), (CANDIDATES[0], CANDIDATES[1:])):'''),
):
    verifier = replace_once(verifier, old, new)
compile(verifier, "steady256-burst-verifier", "exec")
# The archived loader composes the registered reference modules, without its
# old final print. Read them from the same evidence tree as the prior run.
namespace = {"__name__": "__main__"}
exec(compile(loader.replace("Path(sys.argv[1])", f"Path({str(parent)!r})"), str(parent / "fetch-summary.sh"), "exec"), namespace)
print(verifier)
