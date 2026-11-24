"""Test explicit page mode against both earlier service pairs and the independent reference."""

import argparse
import json
from pathlib import Path

from model_decode_reference import lcp
from verify_decode import compare
from verify_model_decode import check_frozen_requests, natural, normal, read


def identity(folder):
    return {
        (r["history"], r["result"]["name"]): r["result"]
        for r in read(folder / "fixed.json")
        if r["phase"] == "identity"
    }


def verify(root):
    output = {}
    reg = read(root / "model-decode-registration.json")
    for gpu, model in reg["models"].items():
        d = root / f"gpu{gpu}"
        page = d / "source-page"
        assert (
            read(page / "lease-after.json")["phase"] == "idle"
            and read(page / "fixed-exit.json") == 0
        )
        assert all(
            read(page / "shutdown.json")[k] == 0
            for k in ("kv_allocated_blocks", "active_requests", "waiting_requests")
        )
        assert (
            "RILEY_NATIVE_ATTENTION mode=page-v1" in (page / "server.log").read_text()
        )
        arms = {
            e: [d / f"{i}-{e}" for i, name in enumerate(reg["order"][gpu]) if name == e]
            for e in ("riley", "vllm")
        }
        check_frozen_requests(root, page)
        ident = identity(page)
        orig = [identity(p) for p in arms["riley"]]
        ref = [identity(p) for p in arms["vllm"]]
        for path in arms["riley"]:
            assert (
                "RILEY_NATIVE_ATTENTION mode=fixed-tile-unsplit-v1"
                in (path / "server.log").read_text()
            )
            assert (path / "binary.sha256").read_text() == (
                page / "binary.sha256"
            ).read_text()
            page_cli = (page / "server-arguments.txt").read_text().splitlines()
            pos = page_cli.index("--native-attention-mode")
            assert page_cli[pos : pos + 2] == ["--native-attention-mode", "page-v1"]
            assert (
                page_cli[:pos] + page_cli[pos + 2 :]
                == (path / "server-arguments.txt").read_text().splitlines()
            )
        cases = []
        for row in read(d / "transformers.json")["cases"]:
            key = row["history"], row["name"]
            r = ident[key]
            assert all(
                a[key]["prompt_ids_sha256"] == r["prompt_ids_sha256"]
                for a in orig + ref
            )
            assert row["prompt_sha256"] == r["prompt_ids_sha256"]
            tokens = r["generated_ids"]
            assert len(tokens) == 64
            cases.append(
                {
                    "history": key[0],
                    "name": key[1],
                    "matches_both_tile": [
                        tokens == a[key]["generated_ids"] for a in orig
                    ],
                    "matches_vllm": [tokens == a[key]["generated_ids"] for a in ref],
                    "matches_hf": tokens == row["ids"],
                    "hf_common_tokens": lcp(tokens, row["ids"]),
                    "tile_common_tokens": lcp(tokens, orig[0][key]["generated_ids"]),
                }
            )
        screens = []
        for ref_dir in arms["vllm"]:
            comp = compare(page, ref_dir)
            va, vb = normal(page), normal(ref_dir)
            for c in comp["cells"]:
                k = (c["history"], c["stream"])
                c["same_reported_warm_cache"] = va[k]["cache"] == vb[k][
                    "cache"
                ] and all(type(n) is int and n > 0 for n in va[k]["cache"])
            screens.append(comp)
        chains = read(page / "natural.json")
        assert len(chains) == 2
        output[model] = {
            "identity": cases,
            "historical_reference_comparisons": screens,
            "natural": [
                natural(w, *spec)
                for w, spec in zip(chains, [(761, 2, 0), (762, 96, 100)], strict=True)
            ],
        }
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(verify(p.parse_args().root), indent=2))
