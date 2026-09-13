# FlashInfer native decode adapter — component evidence

## Status and scope

2026-09-13, integration branch based on `600925ac34bd13cb4024c71206959ea9e34959d1`.
This is a standalone native component, **not a production-registered backend**.
No full-model numerical acceptance, serving benchmark, speedup, or default change is claimed.
The latest serving baseline remains frozen V56; these correctness probes do not replace it.

Source: [adapter](../../../kernels/optional/flashinfer_decode.cu),
[build tool](../../analysis/build_flashinfer_adapter.py),
[functional probe](../../analysis/flashinfer_native_probe.py).
Local and remote final sources match [SHA-256 records](sources.json).

## Implementation and caller contract

- BF16 Q9/KV3/head64/page16; 32 captured rows, active count 1–32, context 1–4096.
- Existing HND `[page, head, token, dim]` K/V is referenced directly. GPU preparation translates row metadata into compact page indices and lengths; it does not repack K/V.
- Metadata workspace is 33,456 bytes, prepared once per iteration and reusable across layers. Attention uses unsplit CUDA-core FlashInfer with no temporary split reduction buffer.
- The caller must retain all device allocations, use the correct CUDA context and ordered stream, supply aligned workspace and sufficient Q/K/V/output/shape extents, initialize status, and prevent concurrent workspace reuse until stream completion. The raw probe C ABI does not establish Riley owner/lease safety.
- Shape positions/page IDs are checked on GPU. Invalid rows are masked; invalid active count masks all rows. Status bits accumulate via OR (2 active, 4 position, 8 page); undersized workspace is rejected on host.
- Model integration needs a dedicated retained metadata allocation: existing layer scratch is overwritten by QKV/projection/FFN. No alias with a still-live result allocation may be assumed.

## Dependency and build

FlashInfer **0.6.16.post3**, torch **2.13.0+cu130**, Python 3.13.
CUDA compiler **13.0.88** is isolated in the task directory; host driver/toolchain is unchanged.
Header tree digest: `2a7d8ab8f81f6cb7fd259270b63bd71e152bff77a105b3c0ba875108b5567e0c`.
The installed include, libcudacxx/include, cub and thrust trees are hashed by the build tool.
The compiler invocation, source digest and output digest are in the build manifests.

| Artifact | Build | Runtime evidence |
| --- | --- | --- |
| `flashinfer_decode.so`, SM89 | Pass | RTX 4090 component probes and native-only memcheck |
| `flashinfer_decode_portable.so`, SM89/90a/100a | Pass | SM89 native-only probe pass; SM90a/SM100a runtime **SKIP: no hardware** |

AOT build success does not establish Hopper/Blackwell runtime correctness or performance.
Multi-GPU behavior is not implemented or tested by this single-device adapter.

## Correctness and numerical observations

`native-adapter-probe.log` records four cases with 1/4/16/32 active rows and lengths drawn from 1/16/398/4096, randomized non-contiguous physical pages, eager/captured equivalence, changed Q, and changed active rows/page IDs/lengths without recapture. Inactive outputs retain sentinels. Invalid page/position/active-count and undersized workspace are rejected/masked.

The native result is bitwise equal to an independently planned **unsplit** Python FlashInfer reference. FlashInfer 0.6.16.post3 CUDA-core planning ignores `disable_split_kv`: the initial smaller-batch comparison failed because the oracle chose split KV. The corrected oracle duplicates requests to 4096 rows and checks the actual pinned-version `_plan_info` unsplit flag; it does not relax tolerance. This larger oracle is for arithmetic verification only and is not a comparable performance workload. A different dependency version must review the private plan schema.

Observed max absolute error against FP32 reference: 0 at length 1 in the single-row case, and approximately 0.00193–0.00195 in the multi-row cases. These observations are **not a preapproved tolerance or a quality gate**. Q is scaled between cases, so the errors do not compare batch sizes. The existing Riley exact profile remains unchanged; full-model logits, greedy token behavior and batch invariance are still pending.

## Sanitizer results, including failures

| Run | Result | Meaning |
| --- | --- | --- |
| `native-adapter-memcheck.log` | **FAIL, exit 86, 34 errors** | Full Python FlashInfer harness reports cuGetProcAddress_v2 CUDA_ERROR_INVALID_VALUE |
| `import-only-memcheck.log` | **FAIL, exit 86, 34 errors** | Same errors with only torch/FlashInfer import and CUDA initialization; no adapter loaded |
| `native-only-memcheck.log` | **PASS, exit 0, 0 errors** | Native adapter, graph and invalid-metadata checks; Python FlashInfer oracle omitted explicitly |

The import-only control supports an environment/import origin for the full harness errors. It does not erase that failure or establish that the full harness is sanitizer-clean. No API-error suppression or driver change was used. Native-only records `exact_flashinfer_unsplit: null`, not true. The ordinary non-sanitized Python oracle run passed separately.

## Reproduction

On the measured host, use the existing task-local environment (paths are historical run locations, not portable installation defaults):

```sh
export ROOT=/tmp/riley-opt-260912
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$ROOT/flashinfer-compatibility/toolchain130/nvidia/cu13"
export PATH="/data/riley-vllm-interim.CfrT9T/venv/bin:$CUDA_HOME/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$ROOT/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:/data/riley-g04-cuda13/lib"
python benchmarks/analysis/build_flashinfer_adapter.py \
  --nvcc "$CUDA_HOME/bin/nvcc" \
  --flashinfer-data /data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/flashinfer/data \
  --architecture 89 --architecture 90a --architecture 100a \
  --source kernels/optional/flashinfer_decode.cu --output /tmp/flashinfer_decode.so
python benchmarks/analysis/flashinfer_native_probe.py --library /tmp/flashinfer_decode.so
/data/cuda-12.8.1/bin/compute-sanitizer --tool memcheck --error-exitcode 86 \
  python benchmarks/analysis/flashinfer_native_probe.py --native-only --library /tmp/flashinfer_decode.so
/data/cuda-12.8.1/bin/compute-sanitizer --tool memcheck --error-exitcode 86 \
  python -c 'import torch, flashinfer; torch.cuda.init(); print("IMPORT_ONLY_CONTROL")'
```

To reproduce the failed full harness, remove `--native-only` from the sanitizer probe.
Run from a checkout containing the three sources. The historical full harness failure predates addition of the native-only CLI switch; its log remains unmodified. Final source hashes identify the subsequent successful normal/native-only probes and build source, rather than claiming an unavailable exact hash for that earlier script revision.

## Next gate

Complete PR03 as a coherent integration batch: pinned optional build and explicit non-exact profile, Rust stream/context/extent/owner checks and recorder lifetime, full-model numerical checks and exact fallback, then matched baseline/vLLM serving measurements. Do not promote from primitive equivalence alone. Follow the plan's hardware skip rules and keep unavailable runtime validation separate from build success.
