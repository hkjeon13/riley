"""Attribute 1,024-row projection launches in a pinned Coder-14B Nsight trace."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from statistics import median
import sys


TRACE_SHA256 = "bf1869c1e9cd809657369736f0bb1d93bec8173e53484780e7d4321a23afe81a"
TRACE_BYTES = 348205056
SERVER_PID = 3217625
LAYERS = 48

# Signatures are checked together with the data-dependent launch order:
# rms_norm -> QKV -> KV write -> attention -> O -> rms_norm -> gate/up -> SiLU -> down.
SIGNATURES = {
    "qkv": ("128x64_64x6_tn_align8", (128, 7, 1), (5120, 7168)),
    "o": ("128x128_32x5_tn_align8", (64, 5, 2), (5120, 5120)),
    "gate_up": ("128x256_32x3_tn_align8", (32, 27, 1), (5120, 27648)),
    "down": ("128x256_32x3_tn_align8", (32, 5, 2), (13824, 5120)),
}


def step_rows(db):
    symbols = dict(db.execute("SELECT id,value FROM StringIds"))
    rows = db.execute(
        """SELECT globalPid,streamId,start,end,gridX,gridY,gridZ,demangledName
           FROM CUPTI_ACTIVITY_KIND_KERNEL
           WHERE end > start AND (globalPid = ? OR ((globalPid >> 24) & 16777215) = ?)
           ORDER BY start,end""",
        (SERVER_PID, SERVER_PID),
    )
    owners, counts = set(), Counter()
    current = []
    steps = []
    symbol_markers = {
        "rms_norm_initial": "vector_bf16_rms_norm_kernel<(bool)0>",
        "rms_norm": "vector_bf16_rms_norm_kernel<(bool)1>",
        "kv_write": "qkv_split_bias_rope_kv_write_x2_kernel(",
        "attention_plan": "::plan_kernel(",
        "attention": "::strided_paged_kernel<",
        "attention_merge": "::merge_parts_kernel<",
        "silu": "silu_gated_multiply_merged_rows_bf16x8_kernel(",
    }
    labels = {}
    for symbol_id, name in symbols.items():
        if name.startswith("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm"):
            for projection, (fragment, grid, _) in SIGNATURES.items():
                if fragment in name:
                    labels[symbol_id, grid] = projection
        for marker, fragment in symbol_markers.items():
            if fragment in name:
                labels[symbol_id, None] = marker

    expected = ["qkv", "kv_write", "attention", "attention_merge",
                "o", "rms_norm", "gate_up", "silu", "down"]
    for owner, stream, start, end, x, y, z, symbol_id in rows:
        owners.add(owner)
        grid = (x, y, z)
        label = labels.get((symbol_id, grid))
        if label in SIGNATURES:
            counts[label] += 1
        else:
            marker = labels.get((symbol_id, None))
            if marker in ("rms_norm_initial", "rms_norm") and grid == (1024, 1, 1):
                label = marker
            elif (marker, grid) in (("kv_write", (8192, 1, 1)), ("silu", (6912, 1, 1)),
                                    ("attention_plan", (1, 1, 1)), ("attention_merge", (2560, 1, 1))):
                label = marker
            elif marker == "attention" and y == 1 and z == 8:
                label = marker
        if label == "rms_norm_initial":
            if current:
                raise ValueError("an incomplete 1,024-row step meets a new step")
            current.append((label, start, end, stream))
        elif current:
            current.append((label, start, end, stream))
            if len(current) == (len(expected) + 1) * LAYERS + 1:
                steps.append(summarize_step(current, expected))
                current = []
    if current or len(owners) != 1 or not steps:
        raise ValueError("missing, unclosed, or ambiguously owned 1,024-row steps")
    # Other shapes can share a launch signature. Every 1,024-row SiLU must
    # belong to a complete validated sequence, independent of the GEMM grid.
    silu_count, = db.execute(
        """SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
           WHERE s.value LIKE '%::silu_gated_multiply_merged_rows_bf16x8_kernel(%'
             AND gridX=6912 AND gridY=1 AND gridZ=1 AND end > start
             AND (globalPid = ? OR ((globalPid >> 24) & 16777215) = ?)""",
        (SERVER_PID, SERVER_PID),
    ).fetchone()
    if silu_count != len(steps) * LAYERS:
        raise ValueError("not every wide SiLU was accounted for")
    return steps, counts


def summarize_step(rows, layer_labels):
    result = {"start_ns": rows[0][1], "span_ns": max(row[2] for row in rows) - rows[0][1]}
    if len({stream for _, _, _, stream in rows}) != 1:
        raise ValueError("wide step spans multiple streams")
    kernel_union = 0
    prior_end = rows[0][1]
    for _, begin, end, _ in rows:
        kernel_union += max(0, end - max(begin, prior_end))
        prior_end = max(prior_end, end)
    result["kernel_union_ns"] = kernel_union
    start = 0
    for layer in range(LAYERS):
        norm = "rms_norm_initial" if layer == 0 else "rms_norm"
        expected = [norm, *layer_labels]
        if layer == 0:
            expected.insert(3, "attention_plan")
        full = rows[start : start + len(expected)]
        actual = [label for label, _, _, _ in full]
        if actual != expected:
            raise ValueError(f"unexpected kernel sequence at {rows[0][1]}, layer {layer}: {actual}")
        start += len(expected)
        for label, begin, end, _ in full:
            result[label] = result.get(label, 0) + end - begin
    return result


def summarize(path):
    with path.open("rb") as file:
        digest = sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != TRACE_BYTES or digest.hexdigest() != TRACE_SHA256:
        raise ValueError("trace size or SHA-256 differs from the recorded profile")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as db:
        steps, signature_counts = step_rows(db)
        clocks = dict(db.execute("""SELECT coalesce(n.text,s.value),n.start FROM NVTX_EVENTS n
                                   LEFT JOIN StringIds s ON s.id=n.textId
                                   WHERE coalesce(n.text,s.value) LIKE 'RILEY_W3_CLOCK_%'"""))
    return {
        "trace_sha256": TRACE_SHA256,
        "trace_bytes": TRACE_BYTES,
        "server_pid": SERVER_PID,
        "layers_per_step": LAYERS,
        "shapes_kn": {name: shape for name, (_, _, shape) in SIGNATURES.items()},
        "all_launch_signature_counts": signature_counts,
        "clock_marks_ns": clocks,
        "median_span_ms": median(step["span_ns"] for step in steps) / 1e6,
        "median_projection_ms_per_step": {
            name: median(step[name] for step in steps) / 1e6 for name in SIGNATURES
        },
        "steps": steps,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), separators=(",", ":"), allow_nan=False))
