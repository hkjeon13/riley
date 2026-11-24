"""Join like-for-like native and HF operator rows for the exact divergent output."""

import argparse
import json
import tarfile
from hashlib import sha256
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGES = (
    "embedding",
    "input_norm",
    "query_rope",
    "attention_context",
    "attention_projection",
    "layer_residual",
    "final_norm",
    "logits",
)


def read(p):
    return json.loads(p.read_bytes())


def tensors(d):
    raw = (d / "rows.bf16").read_bytes()
    data = {}
    meta = None
    offset = 0
    for line in (d / "index.tsv").read_text().splitlines():
        if line.startswith("#"):
            meta = list(map(int, line[1:].split()))
            continue
        tag, layer, at, n = line.split()
        at, n = int(at), int(n)
        if at != offset or n % 2 or at + n > len(raw):
            raise ValueError("bad vector span")
        k = tag, int(layer) if layer != "-" else -1
        if k in data:
            raise ValueError("duplicate trace point")
        data[k] = (
            np.frombuffer(raw[at : at + n], dtype="<u2").astype("uint32") << 16
        ).view("float32")
        if not np.isfinite(data[k]).all():
            raise ValueError("non-finite tensor")
        offset += n
    if offset != len(raw):
        raise ValueError("unindexed bytes")
    return data, meta


def compare(a, b):
    if a.shape != b.shape:
        raise ValueError("different tensor shape")
    delta = a.astype("float64") - b
    return {
        "equal": bool(np.array_equal(a, b)),
        "relative_l2": float(
            np.linalg.norm(delta) / max(np.linalg.norm(b.astype("float64")), 1e-30)
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }


def verify(root):
    if (root / "exit-status").read_text().strip() != "0" or not (
        root / "released"
    ).exists():
        raise ValueError("incomplete")
    reg = read(root / "register-layer-registration.json")
    if reg != read(HERE / "register-layer-registration.json"):
        raise ValueError("different selector")
    for line in (root / "inputs.sha256").read_text().splitlines():
        h, p = line.split(maxsplit=1)
        if sha256((root / p).read_bytes()).hexdigest() != h:
            raise ValueError("changed harness")
    binary = (root / "binary.sha256").read_text().split()[0]
    ref = read(root / "hf/report.json")
    output = {}
    for mode, gpu, marker in [
        ("page", 6, "page-v1"),
        ("tile", 7, "fixed-tile-unsplit-v1"),
    ]:
        d = root / mode
        lease = read(d / "lease-after.json")
        shutdown = read(d / "shutdown.json")
        if (
            lease["physical_index"] != gpu
            or lease["phase"] != "idle"
            or lease["other_processes"]
            or any(
                shutdown[k]
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            or any(shutdown["allocation"].values())
            or (d / "binary.sha256").read_text().split()[0] != binary
            or f"RILEY_NATIVE_ATTENTION mode={marker}"
            not in (d / "server.log").read_text()
        ):
            raise ValueError("bad serving receipt")
    if (
        read(root / "hf-lease.json")["phase"] != "idle"
        or ref["gpu"] != read(root / "tile/lease-after.json")["gpu"]
    ):
        raise ValueError("bad HF GPU receipt")
    reference = read(root / "prior-hf.json")
    with tarfile.open(HERE / "register-reference-raw.tar.gz") as archive:
        if reference != json.load(archive.extractfile("./hf.json")):
            raise ValueError("HF reference changed")
        for name, spec in reg["cases"].items():
            h = ref["cases"][name]
            ids = {}
            for mode in ("page", "tile"):
                old = json.load(archive.extractfile(f"./{mode}/witness.json"))[name]
                row = read(root / mode / "witness.json")[name]
                m = row["measurement"]
                if (
                    row["answer"] != old["answer"]
                    or m["generated_ids"] != old["measurement"]["generated_ids"]
                    or m["prompt_ids_sha256"] != h["prompt_ids_sha256"]
                    or not m["done_ns"]
                    or m["status"] != 200
                ):
                    raise ValueError("instrumentation changed result")
                ids[mode] = m["generated_ids"]
            n = spec["difference_index"]
            if (
                ids["page"][:n] != ids["tile"][:n]
                or h["prefix"] != ids["page"][:n]
                or ids["page"][n] == ids["tile"][n]
            ):
                raise ValueError("wrong token prefix")
            for phase in ("prefill", "decode"):
                key = f"{name}-{phase}"
                maps = {}
                index = 0 if phase == "prefill" else n
                for mode in ("page", "tile", "hf"):
                    d = root / mode / ("trace" if mode != "hf" else "") / key
                    vectors, meta = tensors(d)
                    if mode != "hf":
                        pos = (
                            h["tokens"] - 1
                            if phase == "prefill"
                            else spec["decode_position"]
                        )
                        token = (
                            h["prefill_token"]
                            if phase == "prefill"
                            else h["prefix"][-1]
                        )
                        if meta[:2] != [pos, token] or (
                            phase == "decode" and meta[2:4] != [0, 1]
                        ):
                            raise ValueError(f"wrong row {meta}")
                    selected = int(vectors["logits", -1].argmax())
                    expected = (
                        reference["cases"][name]["generated_ids"]
                        if mode == "hf"
                        else ids[mode]
                    )[index]
                    if selected != expected:
                        raise ValueError(
                            f"{key}/{mode} traced logits differ: {selected} vs {expected}"
                        )
                    maps[mode] = vectors
                groups = {}
                for a, b in (("page", "tile"), ("page", "hf"), ("tile", "hf")):
                    common = maps[a].keys() & maps[b].keys()
                    expected = {
                        (s, i)
                        for s in (
                            "input_norm",
                            "attention_context",
                            "attention_projection",
                            "layer_residual",
                        )
                        for i in range(h["layers"])
                    }
                    if (
                        not expected <= common
                        or not {("embedding", -1), ("logits", -1), ("final_norm", -1)}
                        <= common
                    ):
                        raise ValueError("missing decoder stages")
                    if (
                        a == "page"
                        and b == "tile"
                        and any(
                            ("query_rope", i) not in common for i in range(h["layers"])
                        )
                    ):
                        raise ValueError("missing RoPE query")
                    stages = {
                        s: [
                            compare(maps[a][k], maps[b][k])
                            for k in sorted(common)
                            if k[0] == s
                        ]
                        for s in STAGES
                        if any(k[0] == s for k in common)
                    }
                    thresholds = {}
                    for s in ("attention_context", "layer_residual"):
                        v = stages[s]
                        thresholds[s] = {
                            str(t): next(
                                (i for i, d in enumerate(v) if d["relative_l2"] >= t),
                                None,
                            )
                            for t in (0.01, 0.1)
                        }
                    groups[f"{a}_{b}"] = {"stages": stages, "first_layer": thresholds}
                output[key] = groups
    return {"binary": binary, "observations": output}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(verify(p.parse_args().root), indent=2))
