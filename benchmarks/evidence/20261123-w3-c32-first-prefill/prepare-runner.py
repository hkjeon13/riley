"""Run the existing W3 workload with the additional trace switch in its receipt."""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
for before, after in (
    ("case, budget, workers):", "case, budget, workers, trace):"),
    ('    arm = f"{case}-{budget}-w{workers}-gpu{index}"',
     '    require(trace in ("control", "trace"), "unknown trace setting")\n'
     '    trace_env = f"RILEY_W3_STAGE_TRACE={int(trace == \'trace\')}"\n'
     '    arm = f"{case}-{budget}-w{workers}-{trace}-gpu{index}"'),
    ('"--env", "RILEY_W3_EXPORT_OBSERVATIONS=1",',
     '"--env", "RILEY_W3_EXPORT_OBSERVATIONS=1", "--env", trace_env,'),
    ('f"RILEY_W3_HTTP_WORKERS={workers}"} <=', 'f"RILEY_W3_HTTP_WORKERS={workers}", trace_env} <='),
    ('"steady_budget": int(budget),', '"stage_trace": trace == "trace", "steady_budget": int(budget),'),
    ("*sys.argv[4:8]", "*sys.argv[4:9]"),
):
    if source.count(before) != 1:
        raise ValueError(f"unexpected replay runner anchor: {before}")
    source = source.replace(before, after)
compile(source, str(path), "exec")
path.write_text(source)
