# 260904 계획 진행 상태

**snapshot:** `main@f1ecb0cc11df37d306257ef52a14dd2f31ab2f8e` / 2026-09-04
**규칙:** 이 파일은 진행 결과 ledger다. 계획 작성만으로 상태를 올리지 않는다.

| ID | 상태 | 근거 revision/artifact | 다음 조건 |
|---|---|---|---|
| Q01~Q04 | planned | source precursor만 존재 | Q01 review input과 reviewer/administrator decision |
| RUN-Q05/Q06 | blocked-on-review-and-authority | 없음 | Q01~Q04와 administrator/authority decision |
| G01 | implementation/development-GPU-complete / qualification-pending | 2026-09-10: canonical packed slab + live owner parity; native GPU 11, canonical 1, actual SmolLM2 continuation 1, all exit 0; [receipt](../../benchmarks/results/20260910-g01-packed-c07/README.md) | candidate qualification; remaining G02 capabilities before full decode capture |
| G02P/G02A~G02H | partial: G02G attention + G02C FinalNorm + G02P2 layer audit + G02P3 packed RoPE + G02P4 KV write + G02P5/P6/P7 pointwise + G02A embedding cold audit development evidence | global primitive inventory remains 7 supported/7 unknown; live parity-checked owners bind only matching slots; aggregate Unknown; [HF receipt](../../benchmarks/results/20260910-g02c-hf-norm/README.md) | remaining P/A~B/D~F/H slices and qualification |
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

## G01 native parent-layer development receipt — 2026-09-10

- Additive opaque-parent ABI derives the layer span from checked geometry and
  retains exclusive whole-parent leases. Rust borrows span capture through close;
  the executor cold helper maps exact `KvLayout` to actual buffers.
- New generic/shared-GQA tests each perform 2 captures × 64 replays, isolated eager
  byte parity, complete parent preservation, wrong-capacity/context rejection and
  allocation-zero close. All new/existing focused GPU commands exited normally.
- Artifact/source hashes and raw logs: [receipt](../../benchmarks/results/20260910-g01-parent-layer-native/README.md).
- This is a native bridge substep, **not G01 complete**: packed metadata slab and
  the CPU/C07 identity/evidence chain are not connected; production dispatch is
  unchanged and Attention remains Unknown. No full-model or performance claim.
- Host I/O pressure remains high. The previous storage stall no longer blocks
  every correctness run, but the host is not qualified for latency measurements.

## G01 canonical slab and actual model receipt — 2026-09-10

- Supersedes the native-only substep above: canonical exact host/device slab
  layouts now bind the real metadata allocation to native attention capture.
  Five views share one permanent native lease, validated before capture.
- Live C07 evidence requires same-resource GPU graph/eager byte parity and exact
  metadata/KV/layer/query-head identity. Only that owner's Attention slot becomes
  Supported; the global query and full-chain aggregate remain Unknown.
- SmolLM2-135M BF16 layer 29 was exercised over 3 decode steps with 8 replays per
  step. Independent executor logits, greedy tokens and initialized KV across all
  layers match; all resources close to zero. GPU tests total 13, process exit 0.
- CUDA failure/ambiguous completion or close poisons the executor; validation
  rejection remains recoverable. Full decode capture/default dispatch is not
  enabled. Candidate qualification and broader matrix remain pending.
- Exact source overlay, model/binary hashes, successful logs and the initial
  fixture failure: [receipt](../../benchmarks/results/20260910-g01-packed-c07/README.md).

## G02C canonical FinalNorm development receipt — 2026-09-10

- Existing native canonical BF16 RMSNorm capture now borrows actual executor
  input/output/weight/stream owners. The M=1 helper derives physical weight,
  BF16 shape, epsilon and reduction profile from the prepared plan.
- Live owner evidence requires same-resource eager/graph byte parity, preserved
  input/weight/output tails and finite results. Typed binding mismatches remain
  Unknown; explicit Unsupported and the separate per-layer Norm slot are preserved.
- Canonical GPU matrix covers (1,64), (3,257), (1,576), 2 captures × 64 replays,
  context rejection/recovery, explicit close/Drop and non-finite poison.
- Synthetic Llama H64/L2 exercises the actual loader/executor helper over prefill
  plus 3 decode steps, 32 graph replays each. Final norm bytes, logits and token
  continuation match independent eager execution; all resources close to zero.
- Actual SmolLM2 selects HF RMSNorm and correctly rejects canonical capture before
  device work. Its logits and continuation `[808, 2775, 288, 536]` remain equal.
  **This does not establish SmolLM2 graph support.** G02C-HF is added as a missing
  capture/lifecycle/parity prerequisite; G02P2 also needs profile-specific audit.
- Global primitive inventory and aggregate admission remain unchanged. No full
  graph/default dispatch, qualification, benchmark or deployment promotion.
- Final gates: runtime GPU 3 + C05-12 regression GPU 2, all process exit 0;
  runtime CPU 258, CUDA CPU 79 + graph contracts 32, architecture 15 + inventory
  boundaries 2; fmt/diff passed. Normal Clippy exit 0, no new implementation-module
  diagnostics; existing repository warnings remain.
- Exact source, checkpoint generator/hashes and raw logs:
  [receipt](../../benchmarks/results/20260910-g02c-final-norm/README.md).

## G02C-HF SmolLM2 FinalNorm development receipt — 2026-09-10

- Supersedes the missing HF primitive in the preceding G02C receipt: additive
  HF begin/enqueue ABI now captures the existing eager HF RMSNorm kernel through
  one shared launch helper. No mathematical kernel or reduction order changed.
- Exact BF16 H576, rows 1..8192 and epsilon bits are checked before capture.
  The legacy norm-family resource tag now carries an immutable profile through
  capture/graph/exec. Canonical and HF enqueue reject swapped captures; abort,
  instantiate/replay/close validate and preserve the profile and existing leases.
- FinalNorm binding now includes the executor's exact reduction profile. Fixed37
  stays rejected. A same-owner parity-checked HF owner can supply only its own
  FinalNorm evidence; global query/default dispatch remain unchanged.
- Actual SmolLM2 H576/L30 executes prefill + 3 decode steps. At each step 32 graph
  replays preserve final norm output, full logits, greedy continuation and every
  initialized KV token across all layers/heads. Synthetic canonical executor
  regression also passes. Both close all resources to zero.
- Final gates: 51 GPU tests across model/owner (3), native profile (1), shared
  graph lifecycle (40), C05-22 (3), parent attention (4); all commands exit 0.
  CPU: runtime 258, CUDA 80, graph contracts 32, architecture/inventory 17.
  Normal Clippy exit 0 with no new module diagnostics; fmt/diff passed.
- G02P2 per-layer owner binding is still separate; this work does not promote it,
  aggregate admission, full decode capture, qualification or vLLM performance.
- Final command outcomes, source overlay/hashes and initial source-contract
  failure/correction: [receipt](../../benchmarks/results/20260910-g02c-hf-norm/README.md).

## G02P2 cold layer norm resource audit — 2026-09-10

- 실제 executor plan에서 모든 InputNorm/PostAttentionNorm weight/profile/epsilon을
  도출하고 dispatch의 공유 입력/출력 버퍼에 연결했다. 각 site에서 graph/eager byte parity,
  입력·가중치 보존·유한값을 확인한 뒤 graph를 닫고 원래 hidden_norm을 복원한다.
- 완료된 M=1 iteration의 standalone residual 경로만 검사한다. fused/초기화 전 실행은
  사전 거부되며, mutation 이후 실패는 executor 재사용을 차단한다. layer/operation site가
  binding에 포함되므로 per-layer 증거가 FinalNorm으로 승격되지 않는다.
- SmolLM2 30 layers × 2 sites × 4 iterations = 240회, canonical 16회 검사 완료.
  독립 executor 대비 logits·초기화된 KV·후속 토큰·할당 수 일치, close 후 자원 0.
- 최종 CUDA-enabled C07 suite 26 passed (CPU 계약 포함), runtime CPU 258,
  architecture 15 + inventory 1; 정상 exit 0. fmt/diff와 일반 Clippy 통과;
  기존 경고는 남아 있고 변경 norm 모듈 진단은 없다.
- **G02P2 전체 완료가 아니라 cold mapping 검증 완료다.** iteration 사이의 공유 버퍼를
  사용하므로 각 layer 실행 당시 activation이나 full-graph swap schedule을 증명하지 않는다.
  retained aggregate owner/G02H/G03는 남아 있다. global inventory/dispatch 승격 없음.
- 후속 성능 비교 범위는 사용자 선택에 따라 SmolLM2 단일 모델부터 검증 후 확대한다.
- 소스·원격 동일성·모델/binary 식별·로그: [receipt](../../benchmarks/results/20260910-g02p2-layer-norm/README.md).

## G02P3 packed RoPE span development receipt — 2026-09-10

- 기존 byte-zero position graph에 additive parent-span ABI를 추가했다. packed metadata
  전체 allocation의 lease를 유지하면서 정렬된 position offset을 begin/capture/graph/exec에
  결속한다. 범위·overflow·동일성·transfer/clear를 검증하며 kernel math는 변경하지 않았다.
- actual M=1 executor의 Q/K 버퍼·absolute table·PackedIterationLayout에서 자원을 도출했다.
  device position과 host mirror를 대조하고 graph/eager byte parity, 전체 metadata/input/table
  보존 및 output 복원 후 logits/KV/continuation을 검증했다. stale position은 거부한다.
- GPU 48 passed: 새 span 1 + native rejection 1 + HF/canonical model 2 + graph regression 40
  + parent attention 4. CPU CUDA 81 + source contracts 32, runtime 258 + boundaries 16.
  모든 성공 명령 exit 0; 초기 compile 실패/수정과 lint 로그도 보존한다.
- **cold resource audit까지 완료**, in-flight layer activation/retained aggregate/full decode
  graph는 미완료다. global inventory·default dispatch·qualification·성능 승격 없음.
- 증거: [G02P3 receipt](../../benchmarks/results/20260910-g02p3-rope-span/README.md).

## G02P4 packed parent KV write development receipt — 2026-09-10

- additive native/Rust bridge가 실제 K/V 전체 parent와 packed metadata를 보유하고,
  D64/page16 layer geometry로만 쓰기 위치를 도출한다. 기존 C05-18 standalone ABI는 유지한다.
- metadata field 5개는 하나의 slab lease를 공유하며 alignment/range/non-overlap과 실제
  device payload를 검증한다. capture/graph/exec에서 parent/layer/metadata 결속을 유지한다.
- 실제 SmolLM2 30 layers × 4 iterations = 120회 write/restore, canonical 8회 검사 완료.
  전체 KV parent를 CPU scatter oracle과 비교한 뒤 복원·readback하며 source/slab 보존,
  독립 executor 대비 logits/initialized KV/continuation 일치 및 zero-resource close를 검증했다.
- GPU 15 passed: 새 write 1 + native rejection 1 + model 2 + C05-18 3 + C05-19 4 + parent 4.
  CPU CUDA/source contracts 81/32, runtime/architecture/inventory 258/15/1. 명령 exit 0,
  C11 ABI/fmt/diff 통과. 일반 Clippy exit 0; 기존 경고는 남아 있다.
- **cold resource audit 완료**, in-flight layer activation/full-chain admission/full decode graph는
  별도다. shared scratch를 각 layer 실행 당시의 값으로 주장하지 않는다. 성능·배포 승격 없음.
- [G02P4 source and GPU receipt](../../benchmarks/results/20260910-g02p4-kv-write/README.md).

## G02P5/P6/P7 pointwise cold audit — 2026-09-10

- 기존 native SiLU/gated multiply/residual graph에 실제 executor 버퍼를 빌려주는 Rust
  owner를 추가했다. arity·BF16 size·context·alias·lease를 capture 전에 검증한다.
- gate_raw→gate_activated, gate_activated/up_raw→gated_product, 두 attention/MLP residual
  mapping을 actual workspace와 연결했다. graph/eager 전체 byte parity, finite output,
  입력/tail 보존, 원래 output 복원·readback 및 다음 decode 일치를 검증했다.
- GPU 43 passed: 새 primitive matrix 1 + HF/canonical model 2 + 기존 graph 40.
  CPU CUDA/contracts 82/32, runtime/architecture/inventory 258/15/1. exit 0,
  fmt/diff 통과; 최종 일반 Clippy에서 pointwise 모듈 진단 없음, 기존 경고 유지.
- 건강한 완료 executor에서도 fused residual 설정은 사전 거부하며 separate 복원 후 성공한다.
- **cold audit 완료**이며 실제 layer 실행 당시 activation/full-graph swap/aggregate admission은
  별도다. kernel math·hot dispatch·inventory·성능·배포 승격은 변경하지 않았다.
- [G02P5/P6/P7 evidence](../../benchmarks/results/20260910-g02p567-pointwise/README.md).

## G02A Embedding cold audit — 2026-09-10

- 기존 C05-20 validation/gather/status-D2H graph를 actual table/token/output/error scratch에
  연결하는 borrowed owner를 추가했다. native/kernel 변경 없이 completion 후 상태를 검증한다.
- 잘못된 토큰은 earliest position/ID와 no-output-write 결과를 반환하며, 정상 재실행이 가능하다.
  CUDA/report 오류는 terminal이다. 정확한 32-byte pinned report는 cold caller가 보유한다.
- actual M=1 executor의 physical BF16 table과 packed token prefix를 검증하고 선택한 행의
  byte oracle과 비교했다. 전체 table/token 보존, output/error scratch 복원 및 후속
  logits/KV/continuation 일치를 확인했다.
- GPU 6 passed: borrowed fixture 1 + C05-20 regression 3 + HF/canonical model 2.
  CPU CUDA/contracts 83/32, runtime/architecture/inventory 258/15/1. exit 0, fmt/diff 통과.
  일반 Clippy에서 새 embedding 모듈 진단 없음; 기존 경고 유지.
- cold resource audit이며 retained aggregate/full decode graph 연결은 별도다.
  global inventory·hot dispatch·기본값·성능·배포 승격 없음.
- [G02A evidence](../../benchmarks/results/20260910-g02a-embedding/README.md).

## G02B partial strict bridge / real-policy boundary — 2026-09-10

- Borrowed C05-21 GEMM bridge와 Q/K/V/output/gate/up/down cold mapping 추가.
  canonical H64/L2는 56개 projection 비교, 224 replay 및 복원/continuation 통과.
- 실제 SmolLM2 hidden/key_value/down은 permissive policy로 준비되어 strict graph가
  mutation 전에 거절한다. 실제 선택된 알고리즘은 split-K=1/NONE/workspace=0이다.
  **SmolLM2 GEMM graph parity는 미검증**이며 strict plan으로 재선택하지 않았다.
- 최초 GPU run 1 pass/1 failure를 보존했다. 최종 GPU 5 pass는 borrowed fixture 1,
  C05-21 regression 2, canonical graph 모델 1, SmolLM2 거절/continuation 1이다.
  CPU 83/32/258/15/1, fmt/diff pass. Clippy exit 0, 신규 audit 길이 warning 1.
- 0-byte native workspace lease에는 별도 cold sentinel을 사용한다. 실제 production
  workspace owner 결속/공유 span, 선택 policy를 보존하는 별도 graph 계약이 다음 절차다.
  G02B 전체 완료 및 G02H/G03 승격은 하지 않는다.
- [정확한 범위와 원본 로그](../../benchmarks/results/20260910-g02b-gemm/README.md).

## G02B selected actual-plan development evidence — 2026-09-10

- 기존 strict entrypoint를 유지하고 별도 selected-no-split native/Rust graph 계약 추가.
  실제 prepared policy/algorithm을 바꾸지 않고 split-K<=1/NONE을 검사한다.
- actual workspace `Option`을 그대로 결속한다. zero workspace는 `None`이며 진단용
  empty sentinel 제거. 제공된 공유 parent는 전체 lease/필요 prefix만 사용한다.
- SmolLM2 30 layers x 7 projections x 4 steps = 840 comparisons / 3,360 replays,
  canonical 56 / 224 통과. config/algorithm, 입력/가중치 보존 및 출력 복원 후 logits/KV/
  continuation 일치. SmolLM2 `[808,2775,288,536]`. 자원 수 최종 0.
- GPU 10 passed: 모델 2, selected fixture 1, native 1, strict borrowed 1,
  C05-21 2, C05-22 3. CPU 84/32/258/15/1; fmt/diff/ABI syntax 통과.
  Clippy exit 0; cold audit needless_option_as_deref warning 1과 baseline warnings.
- 모든 관측 plan의 workspace 요구량은 0. nonzero-required workspace 실제 연산과
  split-K>1은 이번 증거 밖이다. G02B retained aggregate/G02H/G03 승격은 하지 않는다.
- 다음은 LM head와 output/completion, metadata H2D chain 및 retained full graph 연결.
  [source·binary·원격 검증 자료](../../benchmarks/results/20260910-g02b-selected-gemm/README.md).

## G02D LM head cold development evidence — 2026-09-10

- 실제 LM head plan/physical weight와 `hidden_norm → logits`, actual workspace Option을
  selected-no-split graph에 연결. plan 재선택/가중치 복제/진단용 workspace 없음.
- GPU 모델 2 passed(exit 0). SmolLM2/canonical 각각 4 comparisons / 16 replays.
  captured 전체 logits를 독립 baseline과 byte 비교하고 복원/후속 logits/KV/continuation
  검증. SmolLM2 `[808,2775,288,536]`, 자원 수 최종 0. 초기화 전 audit 거절도 확인.
- CPU runtime 258/architecture 15/inventory 1, fmt/diff 통과. Clippy exit 0;
  기존 shared probe warning 유지, 신규 LM head warning 없음.
- 다음은 GPU greedy/output/completion과 metadata/retained aggregate/full graph 연결.
  G02D cold 증거이며 전체 graph 또는 성능 검증 완료로 승격하지 않는다.
- [검증 자료](../../benchmarks/results/20260910-g02d-lm-head/README.md).

## G02E GPU greedy cold bridge — 2026-09-10

- actual gathered logits/greedy result owner를 borrowed argmax graph에 결속했다.
  16회 replay completion 후 graph close → 기존 pinned staging D2H → 상태 decode →
  결과 복원/입력 보존 확인. D2H는 graph 외부이며 G02F 완료 증거가 아니다.
- GPU 8 tests: actual models 2, borrowed edge fixture 1, eager argmax 3,
  owned graph regressions 2. SmolLM2 `[808,2775,288,536]` CPU oracle와 일치.
  tie/signed-zero/NaN/±Inf/recovery/Drop/short-result 및 자원 수 최종 0 검증.
- CPU 84/32/258/15/1, fmt/diff 통과. Clippy exit 0, 신규 두 모듈 warning 없음.
- 다음 G02F: 실제 output row map/pinned destination/completion read를 retained graph에
  연결. packed offset/host capacity 검증 필요. full decode/vLLM 비교는 아직 미완료.
- [검증 자료](../../benchmarks/results/20260910-g02e-greedy/README.md).

## G02F actual output graph development evidence — 2026-09-10

- 실제 packed output-index offset과 pinned staging prefix를 지원하는 additive native
  계약 추가. row gather → argmax → D2H 3개 노드를 한 graph에서 실행한다.
  기존 zero-offset/exact-pinned 계약은 유지된다. completion 뒤에만 native 결과 read 허용.
- 실제 slab/logits/gathered/result/io_staging owner를 그대로 결속. map device/host 일치,
  pinned tail·입력·slab 보존, 결과 상태·출력 복원·후속 decode 검증.
- GPU 10: 모델 2, parent fixture 1, native lifecycle 1, 기존 output 3, embedding 3.
  SmolLM2 `[808,2775,288,536]` CPU 기준과 일치. stale 실제 device map은 명시적 오류와
  poison으로 거절. 자원 수 최종 0. CPU 84/32/258/15/1, fmt/diff/ABI 통과.
  Clippy exit 0, 신규 두 모듈 warning 없음.
- output subgraph cold 증거이며 production/full decode aggregate 승격은 하지 않는다.
  다음은 input/metadata H2D chain과 retained aggregate/full graph 연결.
- [원본 증거](../../benchmarks/results/20260910-g02f-output/README.md).

## G02P1 actual packed H2D development evidence — 2026-09-10

- 실제 pinned/device slab을 borrowed whole-slab H2D graph에 결속했다. 매 replay마다
  기존 guarded source staging으로 freshness를 부여하며 native 계약은 유지된다.
- 현재 요청을 다시 pack하여 host mirror/pinned/device 전체와 비교한 후 device를 sentinel로
  덮고 graph로 복원한다. M=1 active payload가 양쪽 전체 slab과 같은 경우만 허용한다.
- 최초 fresh-source stage 누락은 native가 거절했으며 실패 로그를 보존했다. 연결 수정 후
  GPU 5 passed(모델 2, borrowed fixture 1, 기존 H2D 2). 모델당 64 replays와 continuation
  통과. CPU 84/32/258/15/1, fmt/diff, Clippy exit 0; 신규 모듈 경고 없음.
- 다음은 하나의 native aggregate owner/capture다. 공유 자원 lease 중복 제거,
  layer별 hidden buffer swap의 고정 주소 표현, H2D freshness와 전체 상태 전달 필요.
  G02H/G03 및 성능 qualification은 미완료다.
- [검증 및 aggregate 구현 조건](../../benchmarks/results/20260910-g02p1-h2d/README.md).

## G03 hidden-buffer binding prerequisite — 2026-09-10

- batch dispatch의 layer별 owner swap을 ordinal 기반 physical A/B 참조 선택으로 변경.
  홀수 layer는 loop 종료 후 한 번 교환해 기존 final norm/후속 호출 상태를 유지한다.
- 변경 전·후 원격 C07 GPU 각각 14 passed. SmolLM2 L30, canonical L2/L3의 4회 연속
  전체 logits·초기화 KV 해시 및 토큰이 정확히 일치. 별도 batch GPU 3 passed
  (independent forward, fused/separate, mixed prefill/decode). CPU 259/15/1,
  fmt/diff 및 normal Clippy 통과. unrelated model-loader hash 보존.
- native aggregate owner 자체는 아직 구현 전이다. 이번 변경은 layer chain의 주소 표현
  선행 조건만 해소한다. 공유 lease ledger, 전체 capture, freshness/status completion,
  retained replay 및 vLLM 비교는 남아 있으며 G02H/G03 승격 없음.
- [변경 및 전후 증거](../../benchmarks/results/20260910-g03-hidden-bindings/README.md).

## G03 native resource-ledger development evidence — 2026-09-11

- native aggregate용 cold 자원 reservation과 borrowed Rust wrapper 구현. 동일 native
  handle은 한 번만 lease 획득하며, busy/capacity 오류는 역순 rollback한다. context,
  selected no-split GEMM topology, owner thread, close/Drop 및 기존 capture guard 유지.
- SmolLM2 실제 device parent 297개·GEMM plan 5개·pinned 2개를 함께 결속/해제했다.
  SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시와 continuation은 이전 기준과 동일.
- GPU 16 passed(native 2 + C07 14), 실제 모델 reservation 20회. CPU 391 passed,
  ABI/fmt/diff, CPU/CUDA normal Clippy 통과. unrelated model-loader 변경 보존.
- 아직 연산을 기록하는 native aggregate capture/exec는 없다. 다음은 ledger ownership을
  사용한 전체 연산 기록, replay freshness와 completion/status 처리다. G02H/G03,
  qualification 및 vLLM 비교 완료로 승격하지 않는다.
- [원본 검증 보고서](../../benchmarks/results/20260911-g03-resource-ledger/README.md).

## G03 retained transfer graph — 2026-09-11, GPU verification pending

- 기존 resource owner에 explicit H2D → D2D → D2H graph 기록 및 fresh-input replay,
  sync 완료 후 read, graph 해제 후 lease 반환 경로를 로컬 구현했다.
- CPU 84+32, C11 ABI/fmt/diff/normal CPU Clippy 통과. CUDA branch/native 빌드와 신규
  GPU 테스트는 미검증. 전체 모델 graph 연결이나 G02H/G03 승격은 하지 않는다.
- 자동 승인 검토가 기존 원격 scratch로 소스를 보내는 작업을 거절했다. 목적지 소유권 및
  내부 소스 전송의 명시적 승인 부족이 사유다. 원격 쓰기/테스트 실행 없음.
- 5개 소스 파일의 정확한 목록·전후 hash와 승인 후 실행 범위를
  [보고서](../../benchmarks/results/20260911-g03-transfer-replay/README.md)에 기록했다.

## G03 transfer replay remote verification — 2026-09-11

- 사용자 명시 승인 후 지정된 소스 5개만 기존 원격 scratch에 전송, 전후 hash 확인.
- CUDA/Rust/ABI 빌드 성공. GPU 3 passed: transfer graph 512회 fresh replay,
  completion read/입력 거절·복구/close·Drop 및 기존 resource ledger 2개 회귀 검증.
- 최초 cargo 실행은 모든 테스트 통과 로그 이후 timeout 124로 종료. 동일 빌드
  바이너리 별도 실행에서 3 passed, exit 0 확인. 성능 측정으로 사용하지 않는다.
- 전송 승인 차단은 해결됐다. 전체 model DAG와 G02H/G03/vLLM 비교는 여전히 미완료.
- [최종 검증 증거](../../benchmarks/results/20260911-g03-transfer-replay-verified/README.md).

## G03 SwiGLU chain local implementation — 2026-09-11

- resource owner에 실제 eager BF16 SiLU→multiply 커널을 연결하는 6노드 staged graph
  구현. 공유 intermediate와 pinned parent는 ledger가 한 번만 보유한다.
- 실제 모델 scratch 기반 eager 비교·32회 입력 교대 replay·복원 audit 및 fixture 준비.
  CPU 391/ABI/fmt/diff/normal Clippy 통과. CUDA 빌드와 GPU 테스트는 아직 미검증.
- 자동 승인 검토가 기존 5파일 승인을 넘어서는 신규 9파일 전송에 명시 승인이 필요하다며
  거절했다. 원격 쓰기 없음. [파일 목록·해시·검증 범위](../../benchmarks/results/20260911-g03-swiglu-chain/README.md).
- 전체 decode graph/G02H/G03/성능 비교 승격 없음.

## G03 SwiGLU chain remote verification — 2026-09-11

- 사용자 명시 승인 후 소스 9개만 기존 scratch로 전송, 전후 hash 확인.
- CUDA/native ABI/Rust 빌드 성공. GPU 19 passed(exit 0): SwiGLU 2, aggregate 회귀 3,
  C07 14. fixture 192회 변경 입력과 eager byte parity, 실제 모델 640회 replay 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시 및 continuation은 이전 기준과 동일.
  pinned/scratch 복원·allocation 안정성·close/Drop 확인. unrelated model-loader 보존.
- 이는 SiLU→multiply subgraph 검증이다. 전체 layer/model graph, G02H/G03 및 vLLM
  성능 비교 완료로 승격하지 않는다. [최종 증거](../../benchmarks/results/20260911-g03-swiglu-verified/README.md).

## G03 MLP chain local implementation — 2026-09-11

- 선택된 GEMM plan을 유지하는 gate/up → SiLU/multiply → down → residual graph 및 실제 모델 eager parity audit 추가.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과, 무관한 model-loader 6파일 해시 보존.
- CUDA 빌드·GPU 실행 미수행. 자동 승인 검토가 새 MLP 소스 11개 전송의 명시 승인을 요구하며 거절했다. 이번 단계 원격 쓰기 없음.
- 전체 decode graph/G02H/G03/vLLM 성능 비교 승격 없음.
- [파일 목록·검증 범위·증거](../../benchmarks/results/20260911-g03-mlp-chain/README.md).

## G03 MLP chain remote verification — 2026-09-11

- 명시 승인된 11파일 전송 및 최종 해시 확인. CUDA 활성화 Rust 타입 추론 오류 1개를 승인 파일 내 수정한 후 CUDA 빌드 성공.
- GPU 20 passed: MLP native 1, C07 14, ledger/transfer 3, SwiGLU 2. 실제 모델 MLP 640 replay eager parity 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation은 직전 기준과 동일. scratch/pinned 복원 및 model-loader 6파일 해시 보존.
- 11파일 전송 승인 차단 해소. 완료된 scratch의 MLP subgraph 검증이며 전체 decode/G02H/G03/vLLM 성능 승격 없음.
- [최종 증거와 남은 범위](../../benchmarks/results/20260911-g03-mlp-verified/README.md).

## G03 post-attention norm + MLP local implementation — 2026-09-11

- 기존 MLP 경로를 보존하며 canonical/HF RMSNorm→MLP graph와 eager parity audit 추가.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과, model-loader 6파일 보존. 새 native/CUDA feature 빌드 및 GPU 실행 미검증.
- 자동 승인 검토가 새 정규화-MLP 소스 9개 전송에 명시 승인을 요구하며 거절. 원격 쓰기 없음.
- 완료된 scratch 기반 부분 graph이며 전체 decode/G02H/G03/vLLM 승격 없음.
- [구현·승인 payload·검증 범위](../../benchmarks/results/20260911-g03-norm-mlp/README.md).

## G03 post-attention norm + MLP remote verification — 2026-09-11

- 명시 승인된 9파일 전송 및 전후 해시 확인. 추가 수정 없이 CUDA/native/Rust 빌드와 ABI 검사 성공.
- GPU 20 passed(exit 0). norm→MLP 640회 및 기존 MLP 640회 eager byte parity, scratch/pinned 복원 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation은 직전 MLP 기준과 동일. 무관한 model-loader 6파일 보존.
- 이전 전송 차단 해소. 완료된 scratch 기반 부분 graph이며 전체 decode/G02H/G03/vLLM 성능 승격 없음.
- [최종 증거와 남은 통합 범위](../../benchmarks/results/20260911-g03-norm-mlp-verified/README.md).

## G03 layer tail local implementation — 2026-09-11

- 사용자 범위: 성능 측정 전까지 통합·검증, 실제 측정 미실행.
- attention context→output projection→residual→norm→MLP graph 및 eager parity audit 추가. CPU 391/ABI/fmt/diff/일반 Clippy 통과, model-loader 보존.
- CUDA feature/native 빌드·GPU 실행 미수행. 자동 승인 검토가 새 7파일 payload 전송의 명시 승인을 요구하며 거절했다. 원격 쓰기 없음.
- 완료된 scratch 기반 tail 진단으로 전체 layer/decode 또는 G02H/G03 승격 없음.
- [승인 파일·검증 증거·성능 측정 전 잔여 절차](../../benchmarks/results/20260911-g03-layer-tail/README.md).

## G03 layer tail remote verification — 2026-09-11

- 명시 승인된 7파일 전송과 전후 해시 확인. 추가 수정 없이 CUDA/native/Rust/ABI 성공.
- GPU 20 passed(exit 0). layer-tail/norm-MLP/MLP 각각 640 replay eager parity 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation 유지. scratch/pinned 복원 및 model-loader 6파일 보존.
- 7파일 전송 승인 차단 해소. 전체 layer/decode/G02H/G03 qualification은 미완료. 성능 측정 미실행.
- [최종 증거 및 잔여 범위](../../benchmarks/results/20260911-g03-layer-tail-verified/README.md).

## G03 input norm → QKV local implementation — 2026-09-11

- actual selected plans 기반 norm→Q/K/V graph와 모델 eager parity audit 추가. 공통 capture lifecycle 추출.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과, model-loader 6파일 보존. 새 CUDA native/feature 빌드·GPU 회귀 미검증.
- 자동 승인 검토가 새 8파일 전송의 명시 승인을 요구하며 거절. 원격 쓰기 없음.
- 전체 layer/decode/G02H/G03 승격 없음. 성능 측정 미실행.
- [승인 소스·구현·미검증 범위](../../benchmarks/results/20260911-g03-norm-qkv/README.md).

## G03 input norm → QKV remote verification — 2026-09-11

- 승인된 8파일 전송 및 전후 해시 확인. 추가 수정 없이 CUDA/native/Rust/ABI 빌드 성공.
- GPU 21 passed(exit 0). norm-QKV/layer-tail/norm-MLP/MLP 각각 640 replay eager parity 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation 유지. scratch/pinned 복원 및 model-loader 6파일 보존.
- 이전 8파일 승인 차단 해소. RoPE/KV/attention 및 전체 decode/G02H/G03 qualification은 남아 있다. 성능 측정 미실행.
- [최종 증거 및 남은 범위](../../benchmarks/results/20260911-g03-norm-qkv-verified/README.md).

## G03 QKV → RoPE local implementation — 2026-09-11

- norm/QKV에 D64 indexed RoPE와 fresh packed position H2D 연결. 테이블 밖 위치를 launch 전에 거절하며 stale result 차단.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과. model-loader 6파일 보존. 새 CUDA feature/native 및 GPU 검증 미수행.
- 자동 승인 검토가 새 9파일 전송에 명시 승인을 요구하며 거절. 원격 쓰기 없음.
- KV write/attention 및 전체 decode 통합은 남아 있음. 성능 측정 미실행.
- [구현·승인 소스·미검증 범위](../../benchmarks/results/20260911-g03-qkv-rope/README.md).

## G03 QKV → RoPE remote verification — 2026-09-11

- 승인된 9파일 전송 및 전후 hash 확인. 추가 수정 없이 CUDA/native/Rust/ABI 성공.
- GPU 22 passed(exit 0). QKV-RoPE 및 기존 norm-QKV/layer-tail/norm-MLP/MLP 각각 640 replay eager parity 통과.
- packed position parent·rotated scratch 복원, 위치 범위/완료 거절, SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation 유지.
- model-loader 6파일 보존. 승인 차단 해소. KV write/attention 및 전체 decode qualification 미완료. 성능 측정 미실행.
- [최종 검증 증거와 남은 범위](../../benchmarks/results/20260911-g03-qkv-rope-verified/README.md).

## G03 QKV/RoPE → KV write local implementation — 2026-09-11

- 고정 logical/physical block 및 valid prefix 범위의 실제 KV parent scatter 연결. 전체 cache CPU oracle·복원 audit 추가.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과, model-loader 보존. 새 native/CUDA feature 및 GPU 미검증.
- 자동 승인 검토가 새 9파일 payload 전송에 명시 승인을 요구하며 거절. 원격 쓰기 없음.
- 동적 mapping/prefix 변경·attention·전체 decode qualification은 미완료. 성능 측정 미실행.
- [승인 소스·구현·미검증 범위](../../benchmarks/results/20260911-g03-qkv-kv/README.md).

## G03 QKV/RoPE → KV write remote verification — 2026-09-11

- 승인된 9파일 전송 및 전후 hash 확인. 추가 수정 없이 CUDA/native/Rust/ABI 성공.
- GPU 22 passed(exit 0). 통합 KV write 640 replay와 cache parent 전체 CPU scatter oracle·복원 통과. 기존 5개 경로 각각 640 replay eager parity 통과.
- SmolLM2 L30/canonical L2/L3 logits·initialized KV·continuation 유지. model-loader 6파일 보존.
- 고정 mapping/valid prefix의 부분 graph 검증. attention·전체 decode·동적 metadata qualification은 남아 있음. 성능 측정 미실행.
- [최종 증거 및 남은 범위](../../benchmarks/results/20260911-g03-qkv-kv-verified/README.md).

## G03 KV write → attention local implementation — 2026-09-11

- 고정 logical block 0의 canonical grouped/GQA attention 연결 및 metadata fresh staging. eager attention·cache oracle·복원 audit 추가.
- CPU 391/ABI/fmt/diff/일반 Clippy 통과, model-loader 6파일 보존. 새 CUDA native/feature 및 GPU 미검증.
- 자동 승인 검토가 새 9파일 payload 전송에 명시 승인을 요구하며 거절. 원격 쓰기 없음.
- layer tail/실제 layer/전체 decode 통합 및 동적 mapping qualification은 남아 있음. 성능 측정 미실행.
- [구현·승인 파일·미검증 범위](../../benchmarks/results/20260911-g03-attention-chain/README.md).

## G03 KV write → attention remote verification — 2026-09-11

- 승인된 9파일 전송 및 전후 hash 확인. 추가 수정 없이 CUDA/native/Rust/ABI 성공. 이전 승인 차단 해소.
- GPU 22 passed(exit 0). 통합 attention 640 replay eager parity 및 전체 cache CPU oracle·metadata/scratch 복원 통과. 기존 6개 경로도 각각 640 replay 통과.
- SmolLM2 L30/canonical L2/L3의 5개 logits·initialized KV·continuation receipt 유지. model-loader 6파일 보존.
- logical block 0/fixed mapping/prefix의 진단용 부분 graph 검증. 다음은 attention 앞부분과 layer-tail 연결이며, 전체 decode·동적 metadata·G02H/G03 qualification은 미완료. 성능 측정 미실행.
- [최종 증거 및 남은 범위](../../benchmarks/results/20260911-g03-attention-chain-verified/README.md).

## SmolLM2 M=1 full decode native/API verification — 2026-09-11

- 남은 통합 작업과 구체적 9파일/원격 scratch 전송 승인 후 전체 model native capture 및 생성 API 연결. 전송 승인 차단 해소.
- fresh metadata/embedding → 모든 layer → final norm/head/argmax → token/status D2H. prompt는 기존 eager로 prefill하며 generation의 전체 logits D2H는 제거.
- SmolLM2 L30 및 canonical L2/L3 각각 64토큰·4 block에서 모든 logits/전체 KV byte parity 통과. 17-token prompt 뒤 48-token 생성 및 eager fallback 결과 일치.
- 일반 GPU 25 + 공유 replay 오류정책 1(격리된 관측 오류 2종), CPU 377 통과. 비정상 status/잘못된 metadata 거절·완료 불명 자원 유지·정상 cleanup 확인. 9 source hash 및 별도 model-loader 6파일 보존 확인.
- 새 명시적 M=1 API의 기능 통합 증거다. 기존 C06/C07 registry/default server 전환, M>1/긴 context/다른 profile qualification은 승격하지 않는다. 실제 device-loss 실험 및 성능 측정 미실행.
- [최종 검증 기록과 제한](../../benchmarks/results/20260911-g03-full-decode/README.md).

## M=1 generation owner → C06 registry verification — 2026-09-11

- 기존 full decode 생성 API를 GraphRegistry<1>/기존 C06 selector에 연결. native 준비 후만 등록하며 exact key와 owner slot 0을 검증한다.
- 실제 weight/RoPE bytes·binding·selected GEMM·device/runtime·metadata layout에서 cold signature 생성. 공유 가능한 주소 키로 오인하지 않도록 registry는 generation owner 내부에 한정.
- Auto/Require/Disabled 정책, 미지원 Require의 prefill 전 거절, key/slot mismatch 및 poisoned owner 거절 검증. weight·metadata 변경 시 키 변화와 복원 확인.
- 원격 CUDA 빌드 및 GPU 17, CPU 281 통과. SmolLM2 L30/canonical L2/L3의 64-token logits/KV 및 17-token prompt 후 48-token 생성 유지. model-loader 6파일 보존.
- 다음은 serve 요청/취소/완료 및 scheduler KV lifecycle 연결. 기본 서버 전환·다중 bucket 확대·성능 측정 미실행.
- [검증 기록](../../benchmarks/results/20260911-g03-registry/README.md).
