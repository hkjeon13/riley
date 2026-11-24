"""Rescore independent token sequences and report the score margin at the first fork."""

import argparse
import json
from pathlib import Path

from model_decode_reference import lcp


def read(path):
    return json.loads(path.read_text())


def score(root):
    result = {}
    reg = read(root / "model-decode-registration.json")
    for gpu, model in reg["models"].items():
        d = root / f"gpu{gpu}"
        reference = read(d / "transformers.json")
        lease = read(d / "reference-lease.json")
        assert lease["phase"] == "idle" and lease["physical_index"] == int(gpu)
        assert (
            reference["gpu"] == lease["gpu"] and reference["dtype"] == "torch.bfloat16"
        )
        assert reference["model"] == model and reference["attention"] == "sdpa"
        assert len(reference["cases"]) == 4
        arms = {}
        for i, engine in enumerate(reg["order"][gpu][:2]):
            arms[engine] = {
                (m["history"], m["result"]["name"]): m["result"]
                for m in read(d / f"{i}-{engine}" / "fixed.json")
                if m["phase"] == "identity"
            }
        cases = []
        for r in reference["cases"]:
            key = r["history"], r["name"]
            a, b = [arms[engine][key] for engine in ("riley", "vllm")]
            assert (
                a["prompt_ids_sha256"] == b["prompt_ids_sha256"] == r["prompt_sha256"]
            )
            assert (
                a["usage"]["prompt_tokens"]
                == b["usage"]["prompt_tokens"]
                == r["prompt_tokens"]
            )
            ai, bi, hi = a["generated_ids"], b["generated_ids"], r["ids"]
            i = lcp(ai, bi)
            assert len(ai) == len(bi) == 64 and i == r["common_riley_vllm"]
            assert (lcp(ai, hi), lcp(bi, hi)) == (
                r["common_riley_hf"],
                r["common_vllm_hf"],
            )
            fork = r["full_forward_at_pair_fork"]
            leaders = None
            if i == 64:
                assert fork is None
            else:
                assert fork["index"] == i and (fork["riley_id"], fork["vllm_id"]) == (
                    ai[i],
                    bi[i],
                )
                best = fork["top_logits"][0]
                leaders = [e for e in ("riley", "vllm") if fork[e + "_logit"] == best]
            cases.append(
                {
                    "history": key[0],
                    "stream": key[1],
                    "riley_vllm_common_tokens": i,
                    "riley_hf_common_tokens": lcp(ai, hi),
                    "vllm_hf_common_tokens": lcp(bi, hi),
                    "riley_matches_full_hf": ai == hi,
                    "vllm_matches_full_hf": bi == hi,
                    "reference_length": len(hi),
                    "full_forward_highest_scoring_arms_at_fork": leaders,
                    "full_forward_riley_minus_vllm_logit": (
                        fork["riley_logit"] - fork["vllm_logit"]
                    )
                    if fork
                    else None,
                }
            )
        result[model] = cases
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(score(p.parse_args().root), indent=2))
