"""Recompute layer-zero causal attention from the actual native query and ordered KV pages."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from verify_register_layers import compare, read, tensors


def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def vectors(path, pages, shape):
    raw = np.memmap(path, dtype="<u2", mode="r").reshape(
        -1, shape[1], shape[0], shape[2]
    )
    if len(pages) != len(raw):
        raise ValueError("unexpected KV length")
    used = [row[:, : int(n), :] for row, (_, n) in zip(raw, pages, strict=True)]
    bits = np.concatenate(used, axis=1)
    return (bits.astype("uint32") << 16).view("float32"), sha256(
        bits.tobytes()
    ).hexdigest()


def attend(q, k, v, staged):
    heads, width = q.shape
    chunk = heads // k.shape[0]
    out = []
    for i in range(heads):
        dot = k[i // chunk] @ q[i]
        scores = (
            bf16(bf16(dot) * np.float32(width**-0.5))
            if staged
            else dot * np.float32(width**-0.5)
        )
        w = np.exp((scores - scores.max()).astype("float64"))
        w /= w.sum()
        out.append(w @ v[i // chunk])
    return bf16(np.stack(out))


def verify(root, layer_root, oracle_dir=None):
    if oracle_dir:
        oracle_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    prior = read(layer_root / "register-layer-registration.json")["cases"]
    for name in prior:
        for phase in ("prefill", "decode"):
            point = f"{name}-{phase}"
            recs = {}
            for mode in ("page", "tile"):
                d = root / mode / "trace" / point
                m, meta = tensors(d)
                if (d / "rows.bf16").read_bytes() != (
                    layer_root / mode / "trace" / point / "rows.bf16"
                ).read_bytes():
                    raise ValueError("KV download changed the captured tensors")
                lines = (d / "pages.tsv").read_text().splitlines()
                shape = tuple(map(int, lines[0][1:].split()))
                pages = [tuple(map(int, line.split())) for line in lines[1:]]
                if (
                    shape != (16, 4, 128)
                    or len({p for p, _ in pages}) != len(pages)
                    or sum(n for _, n in pages) != meta[0] + 1
                    or any(n != 16 for _, n in pages[:-1])
                    or not 0 < pages[-1][1] <= 16
                ):
                    raise ValueError("bad request page table")
                k, kh = vectors(d / "key.bf16", pages, shape)
                v, vh = vectors(d / "value.bf16", pages, shape)
                q = m["query_rope", 0].reshape(28, 128)
                observed = m["attention_context", 0].reshape(q.shape)
                canonical = attend(q, k, v, False)
                staged = attend(q, k, v, True)
                if mode == "page" and oracle_dir:
                    np.save(oracle_dir / f"{point}.npy", canonical, allow_pickle=False)
                recs[mode] = {
                    "position": meta[0],
                    "key_sha256": kh,
                    "value_sha256": vh,
                    "query_sha256": sha256(q.tobytes()).hexdigest(),
                    "standard": compare(observed.ravel(), canonical.ravel()),
                    "staged": compare(observed.ravel(), staged.ravel()),
                    "per_head_relative_l2_standard": [
                        compare(x, y)["relative_l2"]
                        for x, y in zip(observed, canonical, strict=True)
                    ],
                    "per_head_relative_l2_staged": [
                        compare(x, y)["relative_l2"]
                        for x, y in zip(observed, staged, strict=True)
                    ],
                }
            same = all(
                recs["page"][k] == recs["tile"][k]
                for k in ("query_sha256", "key_sha256", "value_sha256")
            )
            results[point] = {"identical_native_inputs": same, "modes": recs}
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("layer_root", type=Path)
    p.add_argument("--oracle-dir", type=Path)
    args = p.parse_args()
    print(json.dumps(verify(args.root, args.layer_root, args.oracle_dir), indent=2))
