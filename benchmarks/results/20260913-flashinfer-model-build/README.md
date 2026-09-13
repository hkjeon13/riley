# Optional FlashInfer native model build — 2026-09-13

Base commit: `0631f52e031d518e671f921eacaf9df3963f78ef`.
This advances PR03's build and native model integration. **Rust recorder selection, numerical profile acceptance, FlashInfer full-model execution and serving comparisons remain incomplete.** No new serving performance claim.

## Changes

1. `RILEY_FLASHINFER_DATA` enables optional build support from an absolute, already installed data directory. A build-time Python verifier rejects any header tree other than the recorded FlashInfer 0.6.16.post3 + CCCL digest. No downloads, runtime Python calls, or serving selection occur. Cargo tracks the directory and always writes the CMake cache entry, including empty when disabled. CMake also tracks header modifications and directory membership.
2. A separate CUDA object target contains FlashInfer host exceptions within the adapter's noexcept boundary. The remaining native archive retains its existing no-exceptions compilation. A shared internal C declaration prevents signature drift.
3. An explicitly named internal `enqueue_compiled_v7_flashinfer_shared_model` receives a separate workspace. It prepares metadata once after status reset, then calls FlashInfer after each layer's RoPE/KV update while retaining the existing QKV, projection, FFN and output behavior. Existing exact model entry points pass no workspace and never select this backend. Prefill/mixed paths remain unchanged.

The raw model entry assumes the existing validated row/page identities and parent extents. Standalone adapter metadata rejection is **not** full-model validation: QKV/RoPE also accesses pages before attention, so the future recorder must enforce those contracts before launch. Workspace must be separately retained, aligned and non-aliasing through completion; Rust owner/lease enforcement is pending.

## Validation

- Default-backend actual model GPU regression: `loaded_v7_partial` passed, 129 mixed iterations and 224 logits checked; pending-close abort and zero-allocation conditions passed. This is existing-backend regression evidence, not execution of the new FlashInfer model entry.
- Local `cargo check -p riley-cuda --no-default-features`: pass (CPU build only).
- Remote CUDA 13.0.88, SM89: optional-enabled **entire native archive** build pass. Symbol inspection finds both adapter and new internal model entry.
- Same CMake build directory toggled from enabled to disabled: complete rebuild pass. Symbol inspection confirms no FlashInfer definitions or references, while existing GQA model entry remains.
- Missing dependency directory and unreviewed header content are rejected; the real pinned dependency passed CMake verification.
- Initial configure failed because the task-local CUDA prefix lacked `include/cccl`. The failed log is preserved. Task-local links to existing cuBLAS headers/libraries and CCCL resolved it; host installation/driver were not changed. FlashInfer uses its separately pinned bundled header paths.
- This turn's model entry is compile-validated only. Prior standalone SM89/90a/100a build and 4090 functional/memcheck evidence belongs to the prior source snapshot, not an unexecuted full model. Hopper/Blackwell model runtime is untested due to absent devices.

## Reproduction

Use `kernels` as CMake source, CUDA 13.0.88 compiler and toolkit prefix, `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=89`, and `RILEY_FLASHINFER_DATA` pointing to the pinned installed data directory. Build `riley_cuda_native` with `-j4`. Reconfigure the same build directory with `-DRILEY_FLASHINFER_DATA=` and build again to test removal. Exact historical paths and resolved compiler details are in configure logs.

The remote build directory was `/tmp/riley-opt-260912/flashinfer-model-build`; source overlay was `/tmp/riley-opt-260912/hardware-validation-source/kernels`. [Source hashes](sources.json) identify this iteration. The previous standalone probe artifacts remain immutable in [the component evidence folder](../20260913-flashinfer-native-adapter/README.md).

## Remaining PR03 gate

Add the explicit numerical profile and owner-checked Rust/native recorder, with a retained 33,456-byte metadata parent. Then evaluate full logits, greedy outputs and batch invariance before allowing comparative serving runs. Preserve exact fallback. No production profile or server flag should silently enable an unaccepted arithmetic contract.
