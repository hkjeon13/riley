# PR 10 exact paged KV evidence — run001

`b0f45eb0ac6b3fd73d198f09eef58375a85f2349`의 block-16 paged KV manager와
exact paged decode를 RTX 4090/sm89에서 correctness, boundary, lifecycle, OOM,
sanitizer와 launch-structure 기준으로 검증했다. 모든 CUDA·모델 실행은
`server-4096`의 network-disabled container에서 수행했다.

## 결론

기본 Llama decode cache는 separate K/V BF16 physical-block pool과 V1 block table을
사용한다. V1 device payload는 U32 physical block ID, U16 valid-token count와 U32 logical
length만 전달한다. Pool cookie와 U64 generation을 포함한 `BlockId`는 host ownership
검증용이며, generation이 device ABI에 포함된다고 주장하지 않는다.

Pool은 deterministic free list를 사용한다. Prefill/decode는 `reserve → block-table upload
→ model execution → commit` 순서이고, allocation/OOM은 mutation 전에 실패한다. Reservation
뒤 device/model 오류는 tentative block을 회수하면서 owner를 poison하는 fail-closed 경로다.
Reset과 Drop은 block 및 CUDA allocation accounting을 회수한다. Optional sidecar는 V1 주소
table과 분리되어 있고 block generation에 묶이며 reuse 때 invalidate된다.

## Correctness

Low-level GPU test는 logical length `1/15/16/17/31/32/33/128/129`를 통과했다. 모든
shape에서 shuffled physical IDs를 사용했고 contiguous materialized reference가 exact였으며,
실제 paged producer workspace를 역순으로 다시 reduce한 결과도 정방향 결과와 일치했다.

Pinned SmolLM2-135M의 explicit contiguous cache와 default paged cache를 같은 teacher token으로
비교했다.

| 실행 | rows | top-1 mismatch | prefill | worst cosine | max abs | max mean abs |
|---|---:|---:|---|---:|---:|---:|
| 32 decode | 33 | 0 | byte-exact | 0.999664329560 | 0.312500000 | 0.136817740 |
| 128 decode | 129 | 0 | byte-exact | 0.998526788174 | 1.625000000 | 0.501330264 |

Paged attention은 전체 유효 page를 읽는 exact algorithm이지만 contiguous path와 reduction
순서가 다르므로 model logits bit-exact를 주장하지 않는다. Numeric 값은 diagnostic이고
merge gate는 prefill byte parity와 모든 row의 greedy top-1 exact parity다. PR09 contiguous
GPU regression 3개도 모두 통과했다.

## Boundary, lifecycle, OOM

- 한 physical block을 reset 뒤 세 번 재사용했으며 generation은 `1 → 2 → 3`이었다.
- 동일 prompt replay와 다른 prompt의 fresh owner 비교가 각각 byte-exact였고 contamination은
  없었다.
- invalid prompt는 reservation/device upload 전에 실패했고 owner는 poison되지 않았다.
- 1-block pool은 logical length 16에서 다음 block 요청을 preflight OOM으로 거절했다. Table,
  logits, pool/CUDA accounting은 변하지 않았고 reset 뒤 byte-exact replay가 가능했다.
- 8,064-token prefill과 128 decode로 512 blocks/8,192 tokens에 정확히 도달했다. 다음 decode는
  table/logits/pool mutation 전에 실패했다.
- 명시적 close와 implicit paged Drop 뒤 CUDA allocation accounting은 0이었다.
- Lifecycle run의 physical lease는 3회, high-water mark는 1이었다. 성공한 free-list
  pop/generation/owner-bind 구간만 잰 합계는 1,533 ns, 최대는 599 ns였다. Clock-controlled
  allocator benchmark가 아니므로 일반적인 latency 주장에 사용하지 않는다.

SmolLM2 shape의 K+V payload는 token당 23,040 bytes, block당 368,640 bytes다. Near-limit
pool의 `usable_kv_bytes`는 현재 사용량이 아니라 전체 512-block preallocated capacity
188,743,680 bytes다. 최대 V1 device table은 3,072 bytes(U32 IDs 2,048 + U16 valid counts
1,024), pinned table staging은 2,048 bytes다. Static unused-capacity와 현재 committed tail의
dynamic internal fragmentation은 별도 metric이다.

`sidecar_device_bytes`는 opaque view에 선언된 byte length의 합이다. Descriptor가 같은
allocation을 공유하거나 겹치면 unique physical memory가 아니며, 이 실행의 sidecar count와
declared bytes는 0이다.

## Sanitizer와 launch evidence

- Low-level paged memcheck: 0 errors, 0 leaked bytes
- Low-level paged racecheck: 0 hazards, 0 errors, 0 warnings
- Model lifecycle memcheck: 0 errors, 0 leaked bytes

Nsight Compute 2025.1.1은 debug test binary의 matching invocation 한 개를 `--set launch`로
수집했다.

| 대상 | block | grid | grid blocks | threads | waves/SM |
|---|---:|---:|---:|---:|---:|
| shape 129 producer | `(32,1,1)` | `(9,9,1)` | 81 | 2,592 | 0.03 |
| shape 129 reducer | `(256,1,1)` | `(1,1,1)` | 1 | 256 | 0.00 |
| logical 8,065 producer | `(32,1,1)` | `(505,9,1)` | 4,545 | 145,440 | 1.48 |
| logical 8,065 reducer | `(256,1,1)` | `(1,1,1)` | 1 | 256 | 0.00 |

Producer grid가 logical block 수 9/505와 query head 9에 맞게 확장되고 reducer는 하나의
ordered finalization launch라는 구조 증거다. GPU clock을 고정하지 않았으므로 성능
benchmark나 before/after 개선으로 해석하지 않는다.

## 검증 환경과 명령

- Host: `server-4096` / Intel Core i7-13700K / 24 logical CPUs / 67,185,598,464 bytes RAM
- Host kernel: Linux 6.8.0-138-generic; container OS: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- Driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- Image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- Container network disabled; source read-only; target와 사전 복사한 Cargo registry만 writable

주요 command는 모두 `--locked --offline` 조건으로 실행했다.

```text
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test --workspace --all-targets --all-features --locked --offline
cargo test --doc --workspace --all-features --locked --offline
cargo test -p rustinfer-cuda --features cuda --test paged_decode_attention_gpu -- --ignored --test-threads=1 --nocapture
cargo test -p rustinfer-runtime --features cuda --test llama_decode_gpu <selected-test> -- --ignored --test-threads=1 --nocapture
compute-sanitizer --tool memcheck <exact-test-binary> <test-filter> --ignored --nocapture
compute-sanitizer --tool racecheck <exact-test-binary> <test-filter> --ignored --nocapture
ncu --set launch --csv --kernel-name <producer-or-reducer> <exact-test-binary> <test-filter> --ignored --nocapture
```

Strict Clippy, independent C11 ABI compile, workspace tests `125 passed / 46 ignored`, doctests
`13 passed`, selected GPU/model tests와 sanitizer가 모두 성공했다.

## Artifact와 provenance

이 디렉터리의 `raw-events.jsonl`은 merge gate가 된 marker와 launch record를 보존하고,
`metadata.json`은 구현/metric/environment 계약을 기계가 읽을 수 있게 기록한다. 전체 stdout,
sanitizer log, NCU CSV, exact source tar는 다음 append-only 원격 경로에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr10/b0f45eb/full
payload entries: 34
regular-file bytes: 49,001,755
artifact-root apparent bytes: 49,026,331
SHA256SUMS sha256: 868ea7d12278f576965c1651a294dbc92e672459063f3d9301e2f6b2eea73150
source tar sha256: ffadca38bc96170e4858b36f97b0d6ff9d403af29b7d8ca5a56a61aa7d3e2a88
```

앞선 `03808c5`, `43b6a0f`, `1975dcf` 원격 roots는 strict Clippy 실패 원인 추적용으로만
보존되며 통과 증거가 아니다.

## 제한

- 단일 request/batch 1이며 scheduler, prefix sharing, eviction, offload는 구현하지 않았다.
- Device V1 table에는 host generation/cookie가 없다. Runtime이 table을 upload하기 전에 host
  handle을 검증하고 reservation lifetime 동안 reuse를 막는다.
- D64 optimized kernel만 제공하며 다른 head dimension은 reference capability/failure 정책을
  따른다.
- Successful Rust library decode path의 heap/device preparation 부재는 preallocation 구조와
  lexical source guard로 검사했다. CUDA driver/library 내부 allocation까지 측정한 주장은 아니다.
- Allocation accounting은 runtime 소유 buffer/descriptor 기준이며 observed peak VRAM이 아니다.
- NCU는 launch 구조 증거이며 clock-controlled latency/throughput benchmark가 아니다.
- sm89와 기록된 driver/toolkit에서만 검증했다.
