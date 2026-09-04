# 260904 계획 진행 상태

**snapshot:** `main@f1ecb0cc11df37d306257ef52a14dd2f31ab2f8e` / 2026-09-04
**규칙:** 이 파일은 진행 결과 ledger다. 계획 작성만으로 상태를 올리지 않는다.

| ID | 상태 | 근거 revision/artifact | 다음 조건 |
|---|---|---|---|
| Q01~Q04 | planned | source precursor만 존재 | Q01 review input과 reviewer/administrator decision |
| RUN-Q05/Q06 | blocked-on-review-and-authority | 없음 | Q01~Q04와 administrator/authority decision |
| G01 | CPU-contract-complete / native-GPU-pending | working tree atop `f1ecb0cc11df37d306257ef52a14dd2f31ab2f8e`; parent identity, immutable `KvLayout` layer span, metadata/scope match, in-flight lease CPU contract + 6 focused checks | native opaque parent-span ABI, executor/C05-19 owner wiring, GPU parity and lifecycle receipts |
| G02P/G02A~G02H | planned | primitive capability 7 supported, 7 unknown; owner binding incomplete | P1~P7, A, B1~B5, C, D, E, F, G, H |
| G03A~G03D | planned | full model graph 없음 | G02H aggregate Supported + owner-bound, Q05/Q06 |
| G04A~G04D | planned | C06 synthetic dispatch만 존재 | G03 M=1 GPU parity |
| B01 | blocked | actual qualification 없음 | Q05/Q06 완료 |
| B02 Tier D | blocked | current matched campaign 없음 | B01 qualified candidate |
| K00~K03 | blocked | post-graph profile 없음 | G03/G04 + B02 |
| B03/B04 Tier C/S | blocked | final candidate 없음 | K-track 결정 후 candidate 재-freeze |
| S01/S02/RUN-S02P/S03/S04A-B/RUN-S05 | planned-after-core | 없음 | core M4/M5 결과와 별도 승인 |

상태 변경 시 날짜, exact revision, 실행한 gate, 실패/waiver, immutable artifact path를 같은 행 또는
바로 아래 subsection에 추가한다. `passed`라는 단어만 기록하지 않는다.

## G01 source receipt — 2026-09-04

- 변경: `graph_decode_attention_parent_binding`은 caller-provided raw address/offset/length 대신
  non-zero parent allocation identity와 `KvLayout::layer_byte_offset`으로만 exact K/V layer span을
  만든다. key/value alias, wrong parent/device/context/stream, metadata digest mismatch, closed or
  leased parent, layer/capacity 범위를 capture 전 거부한다.
- lifecycle: graph binding이 K/V/metadata parent의 mutable lease를 보유한다. launch completion을
  잊으면 close/reuse가 fail-closed로 남고, completion 뒤의 명시적 graph close만 parent release를
  허용한다. attention capability inventory/default dispatch는 변경하지 않았다.
- CPU gate: `cargo fmt --check`; `cargo test -p riley-runtime --test
  graph_decode_attention_parent_binding_cpu`; `cargo test -p riley-runtime
  graph_decode_attention_parent_binding`.
- waiver: CUDA host/operation authority가 없으므로 `riley_cuda.h`, native C05-19 capture owner,
  executor의 actual `CudaDeviceBuffer` binding, GPU output byte parity/replay/layer-isolation은
  실행하지 않았다. 이 receipt는 TPOT 또는 graph admission 근거가 아니다.

### Remote validation attempt — 2026-09-04

- host: `ssh ai-assistant`, NVIDIA GeForce RTX 4090 / driver `580.173.02`; isolated detached
  worktree: `/home/psyche/riley-worktrees/codex-g01-parent-span-260904` at `f1ecb0c`.
- result: focused CPU contract test passed remotely. A direct `nvcc` compile of
  `kernels/src/version.cu` also completed. The native C05-19 build succeeded after setting
  `CMAKE=/data/cmake-3.31.12/bin/cmake`, and the ignored GPU test binary was started.
- blocked: `graph_c05_19_gpu -- --ignored` entered storage I/O wait (`folio_`) before any GPU
  process appeared and did not complete; the caller-owned cargo/test processes were signalled
  for termination. No parent-span parity, replay, layer-isolation, lifecycle, or performance
  receipt exists. Native parent-span ABI/FFI and executor binding therefore remain pending.
- host diagnosis: capacity/mount were healthy (`/` NVMe: 400GB free; `/data`: 1.2TB free), but
  `/proc/pressure/io` reported `some avg10=74.92` and `full avg10=70.99`. PostgreSQL and
  filesystem workers were simultaneously in D-state. This is a host-wide storage stall, not a
  Riley test failure; do not rerun GPU parity until the host I/O pressure recovers.
