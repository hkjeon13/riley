# FA3 dynamic model metadata — 2026-09-14

Implemented a GPU metadata planner and external-workspace FA3 calls for the existing retained model graph. **The model recorder does not call them yet.** Serving selection and performance remain unchanged.

## Implemented batch

1. A single 128-thread CTA translates Riley mixed packets into FA3's rectangular page table, packed Q offsets and true KV lengths. It validates the complete header and page list before publishing query offsets/lengths. An invalid suffix leaves all query/sequence lengths zero; only in-range physical page IDs may be stored. Existing failure status also suppresses publication. Empty batches publish zero work.
2. A fixed 71,424-byte external workspace holds pages, query offsets, lengths, FA3 scheduling vectors/semaphore and LSE. The future model graph reservation can retain and account for this as an ordinary device buffer, avoiding the primitive plan's hidden internal allocation. No Q/K/V repacking, host metadata readback or per-layer allocation is added by this interface.
3. `riley_fa3_model_schedule` prepares FA3 varlen scheduling once for a batch. Each sequential layer's `riley_fa3_model_attention` reuses those read-only scheduling vectors and resets only the mutable semaphore. Cold preparation sets kernel attributes and validates Hopper capabilities; normal calls use the precomputed schedule. These are code-level changes, **not measured Hopper speedups**.

Both prefill and single-query decode use causal BF16 Q9/KV3/head64 attention with GQA packing, HND/page16 KV and KV non-TMA loads. Causal suffix alignment makes decode's single query attend through the final KV token. Numerical equivalence to Riley remains unverified; previous FlashInfer quality failures still preclude assuming an exact replacement.

## Validation

| Scope | Result |
|---|---|
| RTX 4090 metadata graph replay, changing packet/page values | 34 comparisons passed |
| Shapes | Empty, decode, Q15/16/17, mixed, Q1024, 32 requests, context4096 |
| Failure cases | Invalid suffix page, context overflow/wraparound, packed offset, total, active count, decode shape, inherited failure |
| Inactive/padding table and all failed query lengths | Checked against host expectations |
| Final SM89 metadata memcheck | 0 errors |
| Final SM89 metadata racecheck | 0 hazards, 0 errors/warnings |
| SM90a FA3 external-workspace entry points + scheduler build/link | Pass, final native v2 receipt |
| Actual 4090 model-schedule capability rejection | `cudaErrorNotSupported`, native contract probe passed |
| Cargo optional build and Rust unsupported-device lease rollback | Pass, `fa3-model-build-v1.log` |
| Hopper FA3 scheduling reuse, attention outputs, graph replay and model quality | Not run: no Hopper |

The metadata planner runs independently on SM89. Its graph/sanitizer results **do not test FA3 attention or prove that an all-zero failed batch is handled correctly by the Hopper scheduler/kernel**. That empty-work behavior is a required Hopper gate before model integration is promoted. v2 keeps the metadata kernel definition out of the SM90a adapter translation unit; the final native build compiles the exact layout and external entry points.

The 90-second sanitizer limit is per small metadata-only process. No whole-model racecheck was run. `receipt.json` records pinned upstream commits, compiler argv, source/header hashes and the linked native probe's hash. `metadata-probe.txt` records the SM89 probe and explicitly reports `fa3_attention_executed:false`.

## Recorder integration contract

With stable retained Q/K/V/O/workspace addresses and fixed capacity/context:

1. Outside capture, prepare the attention kernel on the supported Hopper device.
2. In the iteration graph, update the request packet, run metadata prepare, then prepare FA3 scheduling once.
3. Run the existing per-layer Q/K/V and KV-write operations before each FA3 attention call. Layers sharing the workspace must remain serialized on the bound stream.
4. Retain every allocation and graph lease until completion and graph destruction; propagate final status to the caller. The raw ABI assumes recorder-side context, extent, alias and lifetime checks.

Capacity is 1–1024 query rows, maximum 32 requests, physical pages 1–4096 and context 1–4096. Metadata is dynamic within those bounds; addresses and captured kernel geometry remain fixed. There is no concurrent-layer or cross-stream workspace sharing contract.

Next connect these functions to the actual model recorder and an explicit experimental numerical profile, then build the complete path. Hopper numerical, invalid/empty scheduling, repeated-layer, graph and full serving gates remain outstanding. A new vLLM table is deferred until serving execution changes and can be measured under matched conditions.

## Reproduce metadata validation

From the repository root with CUDA 13.0.88:

```sh
nvcc -std=c++17 -O3 -arch=sm_89 -Ikernels/optional \
  benchmarks/analysis/fa3_model_metadata_probe.cu kernels/optional/fa3_model_metadata.cu \
  -o /tmp/fa3-model-metadata-probe
/tmp/fa3-model-metadata-probe
timeout 90 compute-sanitizer --tool memcheck --error-exitcode 9 /tmp/fa3-model-metadata-probe
timeout 90 compute-sanitizer --tool racecheck --error-exitcode 9 /tmp/fa3-model-metadata-probe
```

The existing `build_fa3_native_adapter.py` builds the SM90a native adapter and checks the unsupported-device contract. Python is only an offline build tool; runtime remains native.
