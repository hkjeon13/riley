# PR-G02 — C07 remaining capability completion slices

**목적:** C07의 14개 logical slot 중 현재 `Unknown`인 7개를 “primitive가 존재한다”가 아니라
“현재 executor owner와 exact graph capture가 호환된다”는 기준으로 판정한다. 전체 completion을
한 PR에 넣지 않고 아래 slice를 각각 독립 PR로 진행한다.

## 현재 inventory truth

실제 enum 이름과 순서는 다음과 같다.

```text
MetadataH2d, Embedding, Norm, LayerProjectionGemm, Rope, KvWrite, Attention,
MlpSiluBf16, MlpGatedMultiply, Residual, FinalNorm, LmHead, GpuGreedy,
CompletionBoundary
```

현재 C05 evidence adapter가 이미 **primitive capability 기준으로** `Supported`로 mapping한 slot은 7개다.

```text
MetadataH2d, Norm, Rope, KvWrite, MlpSiluBf16, MlpGatedMultiply, Residual
```

남은 7개는 `Embedding`, `LayerProjectionGemm`, `Attention`, `FinalNorm`, `LmHead`,
`GpuGreedy`, `CompletionBoundary`다. 기존 Supported slot의 primitive를 재구현하지는 않지만,
그 capability가 실제 executor owner/span과 결속됐는지는 별도 audit해야 한다.

## 선행과 병렬성

- G02P와 G02A~G02F source work는 G01 및 Q-track과 병렬 가능하다.
- G02G attention은 G01 GPU parity receipt 이후에만 완료할 수 있다.
- G02H aggregate는 P, A~G와 기존 seven-supported evidence가 모두 닫힌 뒤 실행한다.
- 새 kernel, full graph instantiate, executor hot-path dispatch, CLI/default 변경은 모든 G02에서 금지한다.

각 mapping은 device/context/stream, buffer owner/span, dtype/layout/geometry, mutable control ownership,
completion semantics, exact C05 evidence ID를 비교한다. 하나라도 다르면 `Unknown` 또는 `Unsupported`다.
현재 capability vocabulary에 없는 `Incompatible` 상태를 계획서만으로 새로 만들지 않는다. mismatch의
세부 reason은 별도 typed binding error로 보존하고 inventory에는 fail-closed 상태로 투영한다.

## PR-G02P — existing seven-supported owner-binding audit

primitive capability를 그대로 두고 executor owner binding 여부를 다음 slice별로 확인한다.

- **G02P1:** `MetadataH2d` — 기존 C07 exact slab/H2D owner chain이 selected executor slab과 동일한지 검증
- **G02P2:** `Norm` — per-layer input/output/weight/workspace/span binding
- **G02P3:** `Rope` — Q/K buffer, position metadata, table, layer/head layout binding
- **G02P4:** `KvWrite` — parent KV allocation, layer span, block metadata와 mutation/completion binding
- **G02P5:** `MlpSiluBf16` input/output owner binding
- **G02P6:** `MlpGatedMultiply` gate/up/output owner binding
- **G02P7:** `Residual` attention/MLP residual source/destination alias와 owner binding

각 slice에서 existing binding이 충분하면 characterization test만 추가하고 끝낸다. 부족하면 그 slot의
owner adapter만 구현한다. capability query의 `Supported`를 owner-bound evidence로 자동 승격하지 않는다.

## PR-G02A — Embedding owner/evidence mapping

**한 가지 목적:** C05 embedding validation/status graph를 executor의 token input, embedding weight,
status/output owner와 결속한다.

status D2H가 full decode의 최종 `CompletionBoundary`를 대신한다는 주장은 금지한다.

**완료:** invalid token/status와 valid embedding bytes의 GPU parity receipt가 exact owner digest와 연결된다.

## PR-G02B — LayerProjectionGemm sub-evidence

한 enum slot 안의 모든 projection family가 필요하므로 다음을 각각 별도 PR로 닫는다.

- **G02B1:** Q/K/V projection GEMM geometry와 weight/workspace owner
- **G02B2:** attention output projection
- **G02B3:** gate/up projection
- **G02B4:** down projection
- **G02B5:** B1~B4의 deterministic sub-evidence aggregate를 `LayerProjectionGemm`에 mapping

composite RMSNorm→GEMM evidence를 arbitrary projection에 확장하지 않는다. B1~B4 중 하나라도
Unknown/Unsupported면 B5는 Supported가 아니다.

## PR-G02C — FinalNorm mapping

final norm의 exact rows/hidden width/input/output/weight owner와 C05 norm capture evidence를 별도 binding한다.
per-layer `Norm` slot의 Supported 상태를 그대로 복사하지 않는다.

### G02C development receipt / missing primitive card — 2026-09-10

- Canonical BF16 final norm의 borrowed graph, actual M=1 executor weight/workspace
  binding과 same-owner graph/eager parity probe를 구현했다. 성공 evidence는 live owner에만
  속하고 global inventory, C06 admission, hot-path dispatch는 그대로다.
- **G02C-HF prerequisite (development complete; qualification pending, 2026-09-10):** 실제 SmolLM2 executor는
  `LlamaRmsNormProfile::HuggingFaceSmolLm2`를 선택한다. 현재 canonical RMSNorm C05 capture는
  이 프로필의 evidence가 아니다. `hugging_face_smollm2_rms_norm`의 기존 kernel을 대상으로
  별도 exact capture/lifecycle/GPU parity slice를 구현·검증했다. 기존 HF kernel을
  공유하는 additive ABI를 사용하며 canonical profile 강제 전환은 하지 않는다.
- G02C-HF 범위: opaque input/weight/output/stream owner, reviewed geometry/profile,
  native capture/abort/instantiate/replay/close, finite byte parity, error poisoning,
  실제 SmolLM2 final norm 및 decode continuation. Live owner의 profile-specific
  FinalNorm mapping까지 연결했다. G02P2의 per-layer norm owner audit는 여전히 별도다.
  [HF 개발 검증 결과](../../benchmarks/results/20260910-g02c-hf-norm/README.md).
- Fixed37/fused normalization은 별도 미지원 범위로 유지한다. canonical shape test를 그
  프로필이나 production model qualification에 재사용하지 않는다.
- 결과와 source overlay: [G02C receipt](../../benchmarks/results/20260910-g02c-final-norm/README.md).

## PR-G02D — LmHead mapping

LM-head GEMM의 tied/untied weight identity, final-norm input, full logits output owner와 workspace를 binding한다.
canonical GEMM evidence가 exact M/N/K/layout/algo/stream owner와 일치할 때만 Supported다.

LM-head evidence는 `GpuGreedy` evidence가 아니며 full logits lifetime을 completion까지 유지한다.

## PR-G02E — GpuGreedy mapping

row-gather/argmax의 output-slot order, tie, non-finite status, token/status owner와 fixed-address graph evidence를
binding한다. LM-head output layout과 vocabulary length도 identity에 포함한다.

## PR-G02F — CompletionBoundary mapping

token/status D2H query/synchronize/close와 scheduler commit 전 completion dependency를 binding한다.
completion 미확정은 eager 재실행이 아니라 executor poison/request failure다.

## PR-G02G — Attention mapping

G01의 parent-allocation/layer-span owner와 C05-19 GPU receipt를 `Attention` slot에 연결한다.
QH/KVH/D64/page geometry, layer offset/span, metadata layout, stream/context가 exact일 때만 Supported다.

## PR-G02H — aggregate admission

**한 가지 목적:** 14 slot의 immutable status/evidence/owner-binding digest를 deterministic aggregate로 만들고
C06 selection에 read-only로 제공한다.

**테스트:** one Unknown/Unsupported, swapped evidence, composite evidence 재사용, owner digest mismatch,
slot order/version mismatch, nondeterministic serialization을 모두 fail-closed한다.

## 공통 변경 표면

```text
crates/riley-runtime/src/llama/graph_decode_capture_inventory.rs
crates/riley-runtime/src/llama/graph_decode_c05_capture_capability_evidence.rs
crates/riley-runtime/src/llama/executor/{owner,graph,dispatch,output}.rs
crates/riley-cuda/src/graph.rs
crates/riley-runtime/tests/*graph*_cpu.rs
crates/riley-cuda/tests/graph_cpu.rs
```

## 완료 판정

각 slice는 current capability vocabulary의 `Supported`, `Unknown`, `Unsupported`와 별도의 owner-binding
decision을 evidence와 함께 내면 완료다. G02H aggregate는 모든 slot이 `Supported + owner-bound`일 때만
Supported다. 그때만 G03을 시작한다. 누락 primitive가 확인되면
G03으로 넘어가지 않고 그 primitive 하나의 capture/lifecycle/GPU parity 카드를 이 문서에 먼저 추가한다.

GPU evidence는 Q05/Q06 이후에만 수집한다. source-only mapping을 Supported로 승격하지 않는다.

### G02P2 development update — 2026-09-10

모든 layer의 실제 weight/profile/epsilon과 InputNorm/PostAttentionNorm 공유 버퍼 mapping을
cold audit에 연결했다. SmolLM2 240 site transactions와 canonical 16 transactions에서
same-resource graph/eager byte parity 및 scratch 복원 후 logits/KV/continuation을 검증했다.
최종 CUDA-enabled C07 suite 26 passed, runtime CPU 258, architecture/inventory 16 passed.

이는 iteration 사이의 resource audit이며 각 layer 실행 당시 activation이나 full-graph
buffer-swap sequence 증거는 아니다. graph를 닫은 진단 receipt는 admission에 사용할 수 없다.
G02H retained aggregate와 G03 연결은 계속 필요하다. 상세:
[20260910 G02P2 receipt](../../benchmarks/results/20260910-g02p2-layer-norm/README.md).

### G02P3 development update — 2026-09-10

packed metadata 내부 position offset을 보존하는 native/Rust borrowed RoPE graph와 actual
executor Q/K cold audit를 연결했다. same-resource eager/graph parity, 전체 slab 보존,
output 복원, stale-position rejection 및 HF/canonical continuation을 검증했다.
GPU 48 passed, CUDA CPU/source contracts 81/32, runtime CPU/boundaries 258/16.

공유 버퍼를 iteration 사이에 검사한 증거이며 full-graph layer swap schedule이나 retained
aggregate owner를 증명하지 않는다. G02H/G03 및 나머지 연산 연결은 계속 필요하다.
[20260910 G02P3 receipt](../../benchmarks/results/20260910-g02p3-rope-span/README.md).

### G02P4 development update — 2026-09-10

실제 KV parent/layer span과 packed metadata를 결속하는 native/Rust write bridge 및
executor cold audit를 추가했다. SmolLM2 전 레이어 write/restore, 전체 parent CPU scatter
oracle, 입력/slab 보존과 continuation을 검증했다. GPU 15 passed, CUDA CPU/contracts
81/32, runtime CPU/boundaries 258/16; source/binary/모델 식별과 로그는
[G02P4 receipt](../../benchmarks/results/20260910-g02p4-kv-write/README.md)에 저장했다.

G02P4의 cold resource 검증이며 in-flight activation/retained aggregate/full decode graph
증거는 아니다. 다음은 P5/P6/P7 MLP·residual 연결 및 나머지 capability/aggregate 작업이다.

### G02P5/P6/P7 development update — 2026-09-10

standalone SiLU/gated multiply/attention residual/MLP residual의 실제 shared buffer mapping을
borrowed graph cold audit에 연결했다. same-resource graph/eager parity와 입력/tail 보존,
output 복원 후 HF/canonical logits/KV/continuation을 검증했다. GPU 43 passed,
CUDA CPU/contracts 82/32, runtime CPU/boundaries 258/16.

이는 iteration 사이의 resource audit다. 각 layer activation/retained aggregate/full graph는
계속 미완료이며 fused residual은 이 증거로 승인되지 않는다.
[검증 자료](../../benchmarks/results/20260910-g02p567-pointwise/README.md).

### G02A development update — 2026-09-10

실제 업로드된 embedding table, packed token prefix, output/error scratch와 cold report를
borrowed C05-20 graph에 연결했다. 성공/invalid-token no-write 상태, 전체 table/token 보존,
선택 행 byte oracle, workspace 복원 후 HF/canonical continuation을 검증했다.
GPU 6 passed, CUDA CPU/contracts 83/32, runtime CPU/boundaries 258/16.

C07 aggregate/full decode graph admission은 아직 미연결이다. 다음 핵심은 GEMM/head/output
capability 연결이다. [G02A receipt](../../benchmarks/results/20260910-g02a-embedding/README.md).

### G02B partial development update — 2026-09-10

strict canonical borrowed GEMM과 7종 projection cold mapping을 구현했다. canonical
H64/L2 graph/eager/restore 및 continuation 통과. 실제 SmolLM2는 hidden/KV/down의
permissive policy가 기존 strict graph 계약과 달라 capture 전에 거절된다. 관측된
알고리즘 자체는 split-K=1/NONE이며, policy 변경이나 heuristic 재선택은 하지 않았다.
최종 GPU 5 pass 중 SmolLM2는 거절 후 continuation 검증이고 GEMM graph 검증이 아니다.

다음은 선택된 policy/opaque algorithm을 유지하는 별도 native graph 계약과 zero/shared
workspace owner 연결이다. cold empty sentinel을 production binding으로 인정하지 않는다.
G02B/G02H/full decode graph는 여전히 미완료다.
[receipt 및 최초 실패 기록](../../benchmarks/results/20260910-g02b-gemm/README.md).

### G02B selected-plan development update — 2026-09-10

직전 strict-policy 차이를 additive selected-no-split 계약으로 해소했다. 기존 policy와
opaque algorithm을 유지하고 실제 split-K<=1/NONE만 허용한다. workspace는 실제 owner의
Option을 사용하며 진단용 빈 allocation을 제거했다. native capture/graph/exec/abort/close는
optional workspace lease와 계약 식별을 보존한다. 기존 strict 및 composite 계약은 유지된다.

SmolLM2 전 30 layers/7 projections에서 840 comparisons, 3,360 graph replays와 복원 후
logits/KV/continuation 통과. canonical 포함 GPU 10 passed. CPU 84/32/258/15/1.
실제 모든 plan의 workspace 요구량은 0이며 nonzero-required workspace 연산은 미검증이다.
이는 cold resource bridge 증거이고 retained aggregate/full decode graph 승격은 아니다.
다음은 LM head/output/completion 및 전체 capture chain이다.
[최신 receipt](../../benchmarks/results/20260910-g02b-selected-gemm/README.md).

### G02D development update — 2026-09-10

actual LM head plan/weight/input/logits/optional workspace cold bridge를 연결했다.
SmolLM2/canonical 모델 각각 4개 단계에서 graph 전체 logits가 독립 baseline과 일치했다.
원래 logits 복원, 입력/가중치/plan 보존, allocation 안정성과 continuation을 확인했다.
GPU 모델 2 passed, CPU 258/15/1. Native/ABI 변경 없음.

G02D standalone 증거이며 retained aggregate/full graph 승격은 아니다. 다음은 GPU 토큰
선택과 output/completion이다. [receipt](../../benchmarks/results/20260910-g02d-lm-head/README.md).

### G02E development update — 2026-09-10

actual gathered logits와 result owner를 standalone argmax graph에 연결했다.
SmolLM2/canonical CPU oracle 토큰 일치, completion 후 외부 D2H/status validation,
복원/후속 decode 및 edge fixture를 검증했다. GPU 8, CPU 84/32/258/15/1 통과.
G02F D2H/row mapping/retained completion graph는 별도 미완료다. 다음 절차에서 actual
metadata와 pinned destination을 결속해야 한다. aggregate/full graph 승격 없음.
[receipt](../../benchmarks/results/20260910-g02e-greedy/README.md).

### G02F development update — 2026-09-10

actual packed index span과 실제 pinned staging prefix를 결속하여 gather/argmax/D2H를
한 output graph로 실행한다. completion-gated read가 결과를 반환하며 원본 부모들은
close까지 lease를 유지한다. SmolLM2/canonical token/continuation 및 stale device map
거절, parent offset/pinned tail/native read gate 검증 통과. GPU 10, CPU 84/32/258/15/1.

standalone output graph 증거이며 aggregate/full decode graph는 아직 미완료다. 다음은
실제 input/metadata H2D와 retained aggregate 연결이다.
[receipt](../../benchmarks/results/20260910-g02f-output/README.md).

### G02P1 development update — 2026-09-10

actual whole pinned/device slab H2D와 매 replay fresh-source staging을 연결했다.
SmolLM2/canonical 전체 요청 bytes와 copy/continuation, source refresh/close/Drop 검증.
GPU 5 passed, CPU 84/32/258/15/1. Active M=1 payload가 전체 owner를 채우는 범위다.

다음은 standalone graph를 병렬 보유하는 방식이 아니라 공유 lease를 중복 없이 보유하는
native aggregate capture다. layer별 hidden buffer swap, 전체 오류 상태 및 입력 freshness를
단일 owner에서 처리해야 한다. G02H/G03 승격 없음.
[receipt 및 구체적 결합 조건](../../benchmarks/results/20260910-g02p1-h2d/README.md).

### G03 buffer-role prerequisite — 2026-09-10

일반 batch 실행의 layer 내부 owner swap을 제거하고 ordinal parity로 두 physical buffer의
역할을 선택한다. 홀수 layer의 최종 단일 swap은 기존 호출 후 상태를 보존한다. 전후 GPU
각 14 passed, 실제 SmolLM2 L30/canonical L2/L3 logits·initialized KV 해시 및 토큰 동일.
추가 batch GPU 3, CPU 259/15/1 통과. 이는 native aggregate 구현의 버퍼 매핑 선행 단계며
공유 lease/native capture/freshness/status publication 구현과 retained graph 검증은 남아 있다.
G02H/G03, bucket qualification, vLLM 비교 승격 없음.
[receipt](../../benchmarks/results/20260910-g03-hidden-bindings/README.md).

### G03 native resource-ledger update — 2026-09-11

공유 부모를 중복 없이 확보하는 native reservation과 안전한 borrowed wrapper를 구현했다.
같은 handle 반복, busy buffer/plan rollback, capacity, context/thread 및 close/Drop 검증.
GPU 16, CPU 391 passed. 실제 모델 20회 reservation/close 후 logits·KV·토큰은 이전 기준과
동일하다. SmolLM2 device parent 297/GEMM 5/pinned 2를 실제 owner에서 빌린다.

자원 reservation은 DAG 검증이나 CUDA capture authority가 아니다. 전체 연산 recorder,
fresh input replay, completion/status publication을 연결해야 하며 G02H/G03 승격은 없다.
[receipt](../../benchmarks/results/20260911-g03-resource-ledger/README.md).

### G03 transfer replay local implementation — 2026-09-11

공유 resource owner에 explicit transfer graph(H2D/D2D/D2H), fresh-source replay,
completion read 및 fail-closed 해제를 로컬 구현했다. CPU 116/ABI/fmt/diff 통과.
원격 소스 전송은 자동 승인 검토가 거절했으므로 CUDA 빌드/GPU 실행 미검증이다.
전체 연산 DAG 및 G02H/G03 완료로 승격하지 않는다.
[검증 상태와 구체적 승인 대상](../../benchmarks/results/20260911-g03-transfer-replay/README.md).

### G03 transfer replay remote verification — 2026-09-11

사용자 승인 후 소스 5개를 지정 scratch에 전송하고 CUDA 빌드 및 GPU 3개 테스트를 검증했다.
transfer graph 512회 fresh-input replay와 completion/거절/해제, ledger 회귀 통과.
최종 동일 바이너리 실행 exit 0. 전체 model graph qualification으로 승격하지 않는다.
[receipt](../../benchmarks/results/20260911-g03-transfer-replay-verified/README.md).

### G03 SwiGLU chain local implementation — 2026-09-11

실제 eager BF16 커널 기반 SiLU→multiply graph 및 actual-model scratch audit 구현.
CPU 391/ABI/fmt/diff 통과. 신규 9파일 전송은 자동 승인 검토가 거절했으므로 원격 쓰기,
CUDA 빌드/GPU 실행은 미수행. 전체 graph qualification 승격 없음.
[구체적 승인 대상 및 상태](../../benchmarks/results/20260911-g03-swiglu-chain/README.md).

### G03 SwiGLU chain remote verification — 2026-09-11

사용자 승인된 9파일 전송과 CUDA 빌드 완료. GPU 19 passed(exit 0): SwiGLU 2, ledger/transfer
회귀 3, C07 14. 실제 모델 640회 replay 및 기존 logits/KV/토큰 hash parity 통과.
최종 scratch snapshot의 SiLU→multiply subgraph 증거이며 전체 model DAG 승격은 아니다.
[receipt](../../benchmarks/results/20260911-g03-swiglu-verified/README.md).

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
