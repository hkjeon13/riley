"""Bound additional prefix reuse from each long job's own two completed tool turns."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from audit_agent_prefix import audit


def read(path):
    return json.loads(path.read_text())


def verify(raw, tokenizer, executable):
    digests = {
        name: sha256((tokenizer / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    assert digests == {
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "tokenizer_config.json": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    }
    assert not (tokenizer / "chat_template.jinja").exists()
    prefix = "a" if (raw / "engine-slots-registration.json").exists() else "w"

    def records(gpu, n):
        return [
            job["work"]
            for wave in read(raw / f"gpu{gpu}" / f"{prefix}{n}" / "waves.json")
            if wave["size"] == 16
            for job in wave["jobs"]
            if job["role"] == "target"
        ]

    result = {}
    for gpu in (6, 7):
        for n in (8, 16):
            works = records(gpu, n)
            assert len({r["workflow_id"] for r in works}) == len(works) == 24
            match = {
                r["workflow"]: r
                for r in audit(works, "m", executable, tokenizer, same_workflow=True)
                if r["turn"] == 2
            }
            assert len(match) == 24
            sample = []
            for work in works:
                assert all(
                    a["done_ns"] <= b["sent_ns"]
                    for a, b in zip(work["turns"], work["turns"][1:])
                )
                final = work["turns"][-1]["usage"]
                receipt = match[work["workflow_id"]]
                assert final["prompt_tokens"] == receipt["total_tokens"]
                hit = final.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                eligible = receipt["prior_completed_eligible_tokens"]
                sample.append(
                    {
                        "seed": work["seed"],
                        "prompt_tokens": final["prompt_tokens"],
                        "eligible": eligible,
                        "hit": hit,
                        "additional_if_retained": max(0, eligible - hit),
                    }
                )
            sums = {
                k: sum(r[k] for r in sample)
                for k in ("prompt_tokens", "eligible", "hit", "additional_if_retained")
            }
            sums["fraction_of_executed_prefill"] = sums["additional_if_retained"] / (
                sums["prompt_tokens"] - sums["hit"]
            )
            result[f"gpu{gpu}/{prefix}{n}"] = {"sum": sums, "rows": sample}
    return {"tokenizer_sha256": digests, "runs": result}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "tokenizer", "executable", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    a.output.write_text(
        json.dumps(verify(a.raw, a.tokenizer, a.executable), indent=2) + "\n"
    )
