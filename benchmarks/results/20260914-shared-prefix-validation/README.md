# PR10 shared-prefix batch validation

Status: **host batch/wire and native structural gates passed; captured-model and serving activation remain incomplete.** This is correctness evidence, not a serving benchmark.

The batch metadata owner now has an explicit shared-prefix configuration. It accepts position-identical committed pages across sequences while rejecting aliased append ranges. Three preallocated page arrays account for 24 bytes per physical page; the default exclusive configuration allocates none of this additional storage. A pool-backed test imports two consumers from the same committed prefix, reserves independent append tails, packs six logical entries over four physical pages, then reclaims all pages.

The V7 wire validator checks each reader against the retained owner/page ledger and rejects writes to a shared page even when the other owner is absent from this batch. Duplicate owner pairs and shifted logical positions remain invalid. The ledger must be complete and originate from actual pool ownership; packet bytes cannot establish authority or model identity. Older wire formats remain exclusive.

Native V7 validation has an explicit capability argument whose default is false. Both specializations preserve canonical packet checks. `graph_resources.cu` still uses the default exclusive mode: this change does not enable shared-prefix serving. A future model-owner integration must bind capability and complete ownership to retained resources before activation.

## Executed checks

| Check | Result |
|---|---|
| Final runtime library, optimized test profile | 347 passed, 0 failed, 1 ignored |
| Final V7/variable wire suite | 18 passed |
| Batch metadata suite | 11 passed |
| Shared-prefix focused tests | 4 passed |
| Scheduler regression suite | 48 passed |
| Linux g++ AddressSanitizer + UndefinedBehaviorSanitizer | Exit 0; 3 valid mode/case combinations, 9 rejection checks |
| Native byte mutation sweep | Every one of 61,568 bytes flipped independently; both capability modes invoked without sanitizer errors |

The single ignored runtime test is the existing `scan_timing_diagnostic`, not a failed correctness test or a hardware skip. Native mutation traversal establishes bounded parser execution, not rejection of every mutated identity: Rust retained-authority checks provide that binding. The native test refuses compilation with `NDEBUG` so its assertions cannot silently disappear.

Linux test binary SHA256: `6d279972e305996f6a74d92e78a758f7f44ad37f2d0da6b8b4b979ae44ec019f`. Source and packet hashes are in [receipt.json](receipt.json); native output is in [linux-native.log](evidence/linux-native.log). The packet contains synthetic test identities and tokens only.

## Reproduce

From the repository root, with a fresh output directory:

```sh
mkdir -p /tmp/riley-shared-prefix-check
RILEY_SHARED_PREFIX_PACKET=/tmp/riley-shared-prefix-check/shared.bin CARGO_PROFILE_TEST_OPT_LEVEL=2 cargo test -p riley-runtime --lib shared_prefix --quiet
CARGO_PROFILE_TEST_OPT_LEVEL=2 cargo test -p riley-runtime --lib --quiet
c++ -std=c++17 -O2 -fsanitize=address,undefined -fno-omit-frame-pointer -Ikernels/src kernels/tests/shared_prefix_packet_test.cpp -o /tmp/riley-shared-prefix-check/check
/tmp/riley-shared-prefix-check/check /tmp/riley-shared-prefix-check/shared.bin
```

No CUDA kernel or model inference ran in this gate. Rust/C ABI/CUDA remains the intended execution path, with no Python runtime bridge. Remaining work: owner-authorized captured-model transfer, model/token identity binding, scheduler cache lookup/publication/eviction, full-model parity and cache-hit/cache-miss serving comparison. The [last vLLM comparison](../20260914-adaptive-decode-serving/README.md) remains unchanged; a new comparison table is due when this serving integration is complete, not after each validation change.
