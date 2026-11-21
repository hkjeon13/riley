"""Compare observed pre-decode plans with the isolated priority rule enabled."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay(binary, mode, case):
    fixture = case["requests_index_send_admit_length"] + case["step_windows_ns"]
    encoded = "".join(" ".join(map(str, row)) + "\n" for row in fixture)
    command = [str(binary)] + ([] if mode is None else [mode])
    if mode is None:
        records = [[admit, index, prompt] for index, _, admit, prompt in fixture[:8]] + fixture[8:]
        encoded = "".join(" ".join(map(str, row)) + "\n" for row in records)
    result = subprocess.check_output(command, text=True, input=encoded)
    return [list(map(int, row.split())) for row in result.splitlines()]


def compare(dataset, stock, diagnostic):
    report = {}
    for name, arm in dataset["arms"].items():
        report[name] = {}
        for label, case in arm["cells"].items():
            expected = case.get("observed_plans", case.get("observed_prefill"))
            original = replay(stock, None, case)
            control = replay(diagnostic, "off", case)
            enabled = replay(diagnostic, "on", case)
            if len(expected) != 9 or original != control or original != expected:
                raise ValueError(f"{name}/{label}: stock or diagnostic-off selection differs")
            changed = [idx for idx, (before, after) in enumerate(zip(control, enabled, strict=True), 1)
                       if before != after]
            target = name.startswith("qkv-") and label == "wave-nominal"
            if changed != ([8, 9] if target else []):
                raise ValueError(f"{name}/{label}: priority change escaped its observed boundary")
            report[name][label] = {"changed_steps": changed, "before": control, "after": enabled}
    for order in ("first", "second"):
        baseline, candidate = (report[f"{role}-{order}"]["wave-nominal"] for role in ("riley", "qkv"))
        if baseline["before"] == candidate["before"] or baseline["after"] != candidate["after"]:
            raise ValueError(f"{order}: nominal schedule did not converge")
    return {name: {label: {"changed_steps": item["changed_steps"], "plans": len(item["before"]),
                           "before_tail": [plan[0] for plan in item["before"][-2:]],
                           "after_tail": [plan[0] for plan in item["after"][-2:]]}
                   for label, item in cases.items()} for name, cases in report.items()}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("coder_input", "qwen_input", "confirmation_input", "stock", "diagnostic", "patch"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    sources = {"coder14": args.coder_input, "qwen14": args.qwen_input}
    result = {model: compare(json.loads(source.read_bytes()), args.stock, args.diagnostic)
              for model, source in sources.items()}
    confirmation = json.loads(args.confirmation_input.read_bytes())
    result.update({model: compare(data, args.stock, args.diagnostic) for model, data in confirmation.items()})
    sources["confirmation"] = args.confirmation_input
    cells = [cell for model in result.values() for arm in model.values() for cell in arm.values()]
    print(json.dumps({
        "scope": "Recorded first nine prefill-only plans with fixed admission and completion times.",
        "input_sha256": {model: digest(path) for model, path in sources.items()},
        "diagnostic_patch_sha256": digest(args.patch),
        "binary_sha256": {name: digest(path) for name, path in (("stock", args.stock), ("diagnostic", args.diagnostic))},
        "verified_control_plans": sum(cell["plans"] for cell in cells),
        "changed_plans": sum(len(cell["changed_steps"]) for cell in cells),
        "models": result,
    }, indent=2, allow_nan=False))
