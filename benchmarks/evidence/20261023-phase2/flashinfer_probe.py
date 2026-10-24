"""P1 probe: FlashInfer paged prefill with a fixed split size on Qwen3-4B
geometry. (1) Batch invariance: each request's output bytes across batch
order, composition and isolation, with fixed_split_size and with the default
heuristic. (2) Timing for the P0 step shapes, next to vLLM's FA2 varlen.
Diagnostic only; the step shapes match crates/riley-cuda/tests/attention_bench_gpu.rs.
"""
import json
import sys

import torch

QH, KVH, D, PAGE = 32, 8, 128, 16
SEQUENCES, PAGES_PER_SEQUENCE = 32, 64
SCALE = D ** -0.5
dev = torch.device("cuda")
torch.manual_seed(0)


def natural_context(s):
    if s % 3 == 0:
        return 16 + (s * 7) % 32
    if s % 3 == 1:
        return 128 + (s * 13) % 64
    return 398 + (s * 17) % 128


# Requests: (sequence, query length, kv length). The mixed step has a
# 224-token prompt chunk of sequence 0 (positions 0..223) and 31 decodes.
def step(mixed):
    requests = []
    if mixed:
        requests.append((0, 224, 224))
    for s in range(1 if mixed else 0, SEQUENCES):
        requests.append((s, 1, natural_context(s)))
    return requests


pages = SEQUENCES * PAGES_PER_SEQUENCE
k_cache = (torch.randn(pages, PAGE, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
v_cache = (torch.randn(pages, PAGE, KVH, D, device=dev) * 0.5).to(torch.bfloat16)
block_ids = [(b * 29 + 5) % pages for b in range(pages)]
# One query per (sequence, position), so a request's queries are the same
# in every batch it appears in.
query_bank = (torch.randn(SEQUENCES, 1024, QH, D, device=dev) * 0.5).to(torch.bfloat16)


def request_query(s, qlen, kvlen):
    return query_bank[s, kvlen - qlen:kvlen]


def flashinfer_run(requests, fixed_split, wrapper):
    qo_indptr, kv_indptr, kv_indices, last_len = [0], [0], [], []
    for s, qlen, kvlen in requests:
        n = (kvlen + PAGE - 1) // PAGE
        kv_indices += block_ids[s * PAGES_PER_SEQUENCE: s * PAGES_PER_SEQUENCE + n]
        kv_indptr.append(kv_indptr[-1] + n)
        qo_indptr.append(qo_indptr[-1] + qlen)
        last_len.append(kvlen - (n - 1) * PAGE)
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    kwargs = dict(causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    if fixed_split is not None:
        kwargs["fixed_split_size"] = fixed_split
    wrapper.plan(t(qo_indptr), t(kv_indptr), t(kv_indices), t(last_len), QH, KVH, D, PAGE, **kwargs)
    q = torch.cat([request_query(*r) for r in requests])
    out = wrapper.run(q, (k_cache, v_cache))
    return out, qo_indptr


def split_outputs(out, qo_indptr, requests):
    return {r[0]: out[qo_indptr[i]:qo_indptr[i + 1]].clone() for i, r in enumerate(requests)}


def invariance(fixed_split, wrapper):
    base = step(True)
    reference = split_outputs(*flashinfer_run(base, fixed_split, wrapper)[:2], base)
    variants = {
        "reversed": list(reversed(base)),
        "decode_only": [r for r in base if r[1] == 1],
        "half": base[:12],
        "single_each": None,
    }
    mismatched = {}
    for name, requests in variants.items():
        if requests is None:
            bad = 0
            for r in base[:8]:
                out, indptr = flashinfer_run([r], fixed_split, wrapper)
                bad += int(not torch.equal(out, reference[r[0]]))
            mismatched[name] = bad
            continue
        out, indptr = flashinfer_run(requests, fixed_split, wrapper)
        got = split_outputs(out, indptr, requests)
        mismatched[name] = sum(int(not torch.equal(got[s], reference[s])) for s in got)
    return mismatched


def time_call(fn, iterations=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    import flashinfer
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    result = {"flashinfer": flashinfer.__version__, "invariance": {}, "timing_us": {}}
    for fixed in [None, 128, 256, 512]:
        result["invariance"][str(fixed)] = invariance(fixed, wrapper)
        for shape, mixed in [("decode32", False), ("mixed31+224", True)]:
            requests = step(mixed)
            flashinfer_run(requests, fixed, wrapper)
            q = torch.cat([request_query(*r) for r in requests])
            result["timing_us"][f"flashinfer split={fixed} {shape}"] = round(
                time_call(lambda: wrapper.run(q, (k_cache, v_cache))), 1)
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        for shape, mixed in [("decode32", False), ("mixed31+224", True)]:
            requests = step(mixed)
            q = torch.cat([request_query(*r) for r in requests])
            cu_q = torch.tensor([0] + list(torch.tensor([r[1] for r in requests]).cumsum(0).tolist()), dtype=torch.int32, device=dev)
            seqused = torch.tensor([r[2] for r in requests], dtype=torch.int32, device=dev)
            table = torch.tensor([block_ids[r[0] * PAGES_PER_SEQUENCE:(r[0] + 1) * PAGES_PER_SEQUENCE] for r in requests], dtype=torch.int32, device=dev)
            call = lambda: flash_attn_varlen_func(q, k_cache, v_cache, max_seqlen_q=max(r[1] for r in requests), cu_seqlens_q=cu_q,
                                                  max_seqlen_k=max(r[2] for r in requests), seqused_k=seqused, causal=True,
                                                  block_table=table, softmax_scale=SCALE, fa_version=2)
            result["timing_us"][f"fa2 {shape}"] = round(time_call(call), 1)
    except Exception as error:  # reported, not fatal
        result["fa2_error"] = repr(error)
    print(json.dumps(result, indent=1))
    json.dump(result, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
