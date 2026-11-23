"""Apply the archived W3 workload and proof checks to the fixed-tile slot experiment."""
from pathlib import Path

here = Path(__file__).resolve().parent
modules = {
    "comparison_base": "20261121-w3-coder-rag-cache/verify.py",
    "verify_probe": "20261121-w3-prefix-cache-min-prompt/verify_probe.py",
    "warm_reference": "20261121-w3-coder-short-warmup/verify_warm.py",
    "private_reference": "20261121-w3-private-startup-warmup/verify.py",
    "replay_verifier": "20261121-w3-c32-arrival-replay/verify.py",
    "observation_reference": "20261122-w3-c32-admission-observations/verify.py",
    "budget_reference": "20261122-w3-c32-steady-budgets/verify.py",
    "http_reference": "20261122-w3-c32-http-workers/verify.py",
    "page_reference": "20261123-w3-c32-page-slots/verify.py",
}
print("import sys, types")
for name, relative in modules.items():
    path = here.parent / relative
    source = path.read_text()
    if name == "private_reference":
        for anchor, changed in (
            ("stock, parent, build_checks, original, workloads):",
             "stock, parent, build_checks, original, workloads, *, label, budget, attention_mode, active_slots, context, max_batch_rows, plateau_override, require_stock):"),
            ('    folder = root / f"{case}-gpu{index}"', '    folder = root / label'),
            ('    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]',
             '    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]\n'
             '    argv[argv.index("--prefill-burst-steady-budget") + 1] = str(budget)\n'
             '    argv[argv.index("--max-active-sequences") + 1] = str(active_slots)\n'
             '    argv[argv.index("--max-sequence-tokens") + 1] = str(context)\n'
             '    for option in ("--batch-token-budget", "--prefill-chunk-tokens"):\n'
             '        argv[argv.index(option) + 1] = str(max_batch_rows)\n'
             '    bucket_arg = argv.index("--batch-shape-buckets") + 1\n'
             '    argv[bucket_arg] = ",".join(row for row in argv[bucket_arg].split(",") if int(row) <= max_batch_rows)\n'
             '    if plateau_override:\n'
             '        argv += ["--prefill-plateau-rows", "128"]\n'
             '    argv += ["--native-attention-mode", attention_mode]'),
            ('    require(exact == REQUEST_COUNTS and', '    require((not require_stock or exact == REQUEST_COUNTS) and'),
        ):
            if source.count(anchor) != 1:
                raise ValueError(f"missing verifier anchor: {anchor}")
            source = source.replace(anchor, changed)
    print(f'module = types.ModuleType({name!r})')
    print("sys.modules[module.__name__] = module")
    print(f'exec(compile({source!r}, {relative!r}, "exec"), module.__dict__)')
print((here / "verify.py").read_text())
