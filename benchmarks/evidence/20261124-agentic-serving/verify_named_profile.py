"""Compare profile-only deployment with the same explicitly configured server."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root, baseline):
    arm = root / "native"
    args = (arm / "server-arguments.txt").read_text().splitlines()
    if args != [
        "serve",
        "--model",
        "/models",
        "--model-id",
        "m",
        "--inference-profile",
        "native-bf16-prefix-32k-v1",
    ]:
        raise ValueError("explicit server tuning flags masked profile defaults")
    if (arm / "binary.sha256").read_text().split()[0] != (
        root / "binary.sha256"
    ).read_text().split()[0]:
        raise ValueError("unverified server executable")
    ref = {r["case"]: r for r in read(baseline)["warm2048"]}
    observed = read(arm / "continuations.json")
    if [r["case"] for r in observed] != read(root / "named-prefix32-registration.json")[
        "cases"
    ]:
        raise ValueError("cohort differs")
    rows = []
    manifest = {
        r["index"]: r for r in read(root / "named-prefix32-shapes.json")["rows"]
    }
    for row in observed:
        idx = row["case"]
        prior, final = row["prior"], row["continuation"]
        value = ref[idx]
        usage = final["usage"]
        if (
            not all(
                r["status"] == 200 and r["complete"] and r["done_ns"]
                for r in (prior, final)
            )
            or prior["input_sha256"] != manifest[idx]["prefix_sha256"]
            or final["input_sha256"] != manifest[idx]["request_sha256"]
        ):
            raise ValueError("incomplete or changed prompt")
        rows.append(
            {
                "case": idx,
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "cached": usage["prompt_tokens_details"]["cached_tokens"],
                "first_ms": (final["content_ns"][0] - final["sent_ns"]) / 1e6,
                "matches_observed_cache_and_answer": usage["prompt_tokens"]
                == manifest[idx]["encoded_prompt_tokens"]
                and usage["completion_tokens"] == value["requested_output"]
                and usage["prompt_tokens_details"]["cached_tokens"]
                == value["observed_cache"]
                and final["answer_sha256"] == value["completion_digest"],
            }
        )
    state = read(arm / "shutdown.json")
    return {
        "server_args": args,
        "binary_sha256": (root / "binary.sha256").read_text().split()[0],
        "rows": rows,
        "all_checks_passed": all(r["matches_observed_cache_and_answer"] for r in rows)
        and read(arm / "ready.json")["ready"]
        and read(arm / "lease-after.json")["phase"] == "idle"
        and not any(state["allocation"].values())
        and not any(
            state[k]
            for k in ("waiting_requests", "active_requests", "kv_allocated_blocks")
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("baseline", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.root, args.baseline), indent=2))
