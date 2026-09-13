# Native prefill build integration

This batch connects the previously GPU-verified paged-prefill primitive to the optional native archive. It does not select it in the model recorder or serving path and makes no new performance claim.

## Build contract

- `RILEY_FLASHINFER_DATA` still requires the original FlashInfer 0.6.16.post3 full header-tree digest `2a7d8ab8f81f6cb7fd259270b63bd71e152bff77a105b3c0ba875108b5567e0c`.
- CMake creates a separate build-directory header overlay. Only the prefill object uses it; decode continues using original headers. Installed dependency files are unchanged.
- The overlay transformation is now owned by `kernels/optional/prepare_flashinfer_prefill_overlay.py`. The previous benchmark entry point delegates to it. Python runs only during build, with no runtime interpreter/FFI path.
- The patched prefill header must match GPU-verified SHA256 `996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975`; the generated whole tree is `d34d25c9005e6f7fdf3278720851bc279a5fd378404c683e11a9cdb15c00fdb7`. Reconfiguration checks the whole tree and receipt before reuse. Generated header/receipt changes trigger CMake reconfiguration.
- The internal C ABI declares the three prefill entry points; the implementation includes this header so signature disagreement fails compilation. Cargo tracks both the new CUDA source and overlay helper as native build inputs.

## Observed validation

RTX 4090 / SM89, nvcc 13.0.88, CMake 3.31.12. `check.py` records the exact commands and fail-closed assertions; `checks.json` records outcomes.

- Cargo release build with `--features cuda` and the existing ABI link suite passes (2 tests); this is build/link evidence, not a new Rust prefill-owner test. Local/remote hashes agree for all seven changed build/source files.
- Entire native archive builds with optional support enabled; all prefill/decode raw entry points are present.
- Reconfiguration is idempotent. Deliberately changing the generated prefill header causes automatic build-time reconfiguration to fail. Altering the receipt also fails configuration. These failures are expected negative checks, not skipped tests.
- The unchanged native probe linked directly against the CMake archive passes graph replay, ragged/noncontiguous pages, invalid suffix, 32-request/1,024-query boundaries. All 868,032 BF16 outputs are bitwise identical to the previously verified patched shared library (SHA256 `a54af5adead8ee65e9f934afd0cb95bbe39f2d98b21b362840807d7eb3c2524d`).
- Archive-linked native memcheck: 0 errors. Racecheck: 0 hazards, 0 errors, 0 warnings.
- In the same build directory, disabling the option rebuilds the archive without any `riley_flashinfer_` symbols. Re-enabling it builds successfully. The original installed full-tree lock is rechecked unchanged afterward.
- The initial validation harness assertion did not normalize CMake's line-wrapped error text. The build correctly rejected header drift; the assertion was corrected and the complete sequence rerun. Final logs correspond to that completed run.

The SM90a/100a AOT evidence remains the [prior primitive build](../20260913-flashinfer-prefill-native/README.md); this CMake archive run is SM89 only. Hopper/Blackwell runtime tests remain skipped for unavailable hardware.

## Remaining work

Retained Rust workspace ownership, full-model graph identity, and prefill-only routing within mixed batches are not connected. In particular, feeding the current raw prefill adapter an entire mixed batch would change decode arithmetic; do not do that. Full-model quality gates and a matched existing Riley / candidate / vLLM serving comparison are required before acceptance. No new serving table is warranted for this build-only milestone.

Rollback: leave `RILEY_FLASHINFER_DATA` empty to omit both optional objects. No default numerical profile or serving backend changes in this batch.
