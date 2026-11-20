"""Prepare the pinned baseline-candidate runner and its exact streamed probe."""

from pathlib import Path


def rewrite(path, old, new):
    original = path.read_text()
    if original.count(old) != 1:
        raise ValueError(f"unexpected instrumentation point in {path.name}")
    changed = original.replace(old, new)
    compile(changed, str(path), "exec")
    path.write_text(changed)


if __name__ == "__main__":
    rewrite(
        Path("tools/run_profile_retry.py"),
        'MODES = ("base-first", "ordered-first", "ordered-second", "base-second", "base-trace", "ordered-trace")',
        'MODES = ("base-first", "candidate-first", "candidate-second", "base-second")',
    )
    rewrite(
        Path("tools/profile_workload.py"),
        'save(out / "token-id-probe.json", token_id_probe(plan["address"]))',
        'save(out / "token-id-probe.json", token_id_probe(plan["address"]))\n'
        '                from logprob_probe import save_probe\n'
        '                save_probe(plan["address"], out / "logprob-probe.json")',
    )
