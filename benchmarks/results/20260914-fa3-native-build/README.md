# FA3 native build boundary — 2026-09-14

This is native dependency/build evidence, not a serving performance result or a completed backend. Production dispatch and the current exact backend remain unchanged.

## Scope and source identity

- FlashAttention repository: `98eb7998a0eba4047c7a30375522569d4b8efb20`, specifically its **FA3 `hopper/` C++/CUDA implementation**. This does not use the repository's newer Python CuTeDSL implementation.
- CUTLASS: `7127592069c2fe01b041e174ba4345ef9b279671`, matching that revision's `csrc/cutlass` gitlink.
- Both dependencies carry BSD 3-Clause licenses. The build export retains upstream license files and FA3 AUTHORS; distribution must retain applicable notices.
- Native instantiations: BF16/head64 dense causal attention; paged ragged causal prefill and noncausal decode with PackGQA. SplitKV, FP8, KV append and softcap are disabled.
- Paged variants explicitly set `PagedKVNonTMA=true`. This avoids assuming page16 meets TMA KV-layout requirements. Compile success does not validate Riley's HND strides or page table conversion.

The builder exports immutable git objects to a fresh directory rather than trusting a mutable dependency checkout. Python performs offline source preparation and invokes nvcc; the linked executable has no Python or Torch runtime path. The upstream variable-length scheduler preparation is compiled and linked as CUDA C++.

## Error handling

Upstream `hopper/cuda_check.h` calls `exit(1)` for CUDA and CUTLASS failures. A separate build export replaces those two exits with typed C++ exceptions, preserving original headers and licenses in the dependency checkout. The receipt records both header hashes. Injected CUDA invalid-value and CUTLASS invalid-problem errors must be caught without terminating the probe.

This is not yet a production C ABI: a later `noexcept` adapter must catch all exceptions and translate errors, including asynchronous completion failures. The host injection test does not prove GPU fault recovery.

## Verification limits

Final v2 build/link and both injected host error checks passed (exit 0). The executable contains all three native probe symbols; `link-evidence.txt` records symbols and dynamic dependencies.

| Compiled variant | Registers | Stack bytes | Spill stores / loads bytes |
|---|---:|---:|---:|
| Dense causal | 128 | 0 | 0 / 0 |
| Paged ragged causal prefill | 128 | 64 | 100 / 136 |
| Paged ragged noncausal decode | 128 | 64 | 96 / 132 |

These are ptxas compile statistics, not measured bandwidth or latency. Both paged variants spill, and all three report C7510. The warnings remain unresolved pending compiler/library evaluation and Hopper execution.

CUDA 13.0.88 compiles actual SM90a TMA/GMMA code. The initial dense-only build passed with 128 registers and zero spill loads/stores; ptxas also reported C7510 (potential WGMMA serialization across a function boundary). Preserve that warning for Hopper profiling; do not infer achieved overlap or speed from instruction names.

Available device: RTX 4090, compute capability 8.9. Hopper attention execution, numerical comparison, sanitizer, graph replay and serving performance are **not run: no Hopper device**. This probe intentionally launches no attention even if run on a Hopper host. No Blackwell compatibility is claimed from an SM90a build.

## Reproduce

After fetching the pinned FA3 and CUTLASS commits:

```sh
python3 benchmarks/analysis/build_fa3_native_probe.py FLASH_ATTENTION_CHECKOUT NVCC FRESH_OUTPUT
```

The output preserves exported sources/licenses, exact argv, nvcc version, hashes, build log, host contract output and binary. Existing output directories are rejected. `v1-receipt.json` records the initial dense-only build; final v2 evidence covers all three instantiations.

## Next implementation boundary

Implement a bounded native plan/launch ABI with explicit dense versus HND/page16 layouts, shape and extent checks, device capabilities, workspace ownership, stream binding and cold preparation before graph capture. Connect Rust owner/fallback only behind an explicit experimental numerical profile. Hopper numerical and serving gates must precede promotion; previous FlashInfer numerical failures are evidence against assuming interchangeable output.

No new vLLM table is produced because no serving implementation changed. The last serving baseline remains the [FFN integration comparison](../20260913-prefill-ffn-model-serving/README.md); its numbers are not FA3 results.

Sources: [FA3 paper](https://arxiv.org/abs/2407.08608), [pinned native launch implementation](https://github.com/Dao-AILab/flash-attention/blob/98eb7998a0eba4047c7a30375522569d4b8efb20/hopper/flash_fwd_launch_template.h), [pinned error handling](https://github.com/Dao-AILab/flash-attention/blob/98eb7998a0eba4047c7a30375522569d4b8efb20/hopper/cuda_check.h).
