# SmolLM2 vLLM 대비 성능 개선 계획

작성일: 2026-09-12. 상태: 계획만 작성. 이 문서로 코드 구현·원격 실행·commit·push를 수행하지 않는다.

## 목표와 범위

현재 정확성 검증을 통과한 SmolLM2-135M BF16 c1/p128/o32 후보의 prefill 및 decode 비용을 줄이고, 동일 조건의 vLLM보다 빠른지 실측한다. 첫 대상은 기존 단일 모델·단일 셀이다. 더 큰 모델·동시성·길이 확대는 이 셀의 개선을 확인한 뒤 별도 단계로 진행한다.

기준 후보는 `59f02242a2993b698d5f1c6b970b96bad66fb0a4`, 수치 정책은 `vllm-smol-p128-v1`, 비교 대상은 vLLM 0.27.1이다. 기존 기본 실행 경로와 E0 계약은 유지한다. 근사 연산·양자화·speculative decoding은 이번 범위에서 제외한다.

근거: [실측 보고서](../results/20260911-g04-gui-measurement/README.md), [집계 JSON](../results/20260911-g04-gui-measurement/summary.json), [수치 정책](../profiling/VLLM_SMOL_P128_PROFILE.md).

## 현재 상태와 비용 예산

각 셀에서 새 프로세스 AB/BA 5쌍, 프로세스당 warmup 5회 제외 후 30회 측정했다. 아래 수치는 실행별 중앙값 5개의 중앙값이다.

| 지표 | Riley | vLLM | 해석 |
|---|---:|---:|---|
| 엔진 첫 토큰 | 435.240ms | 7.005ms | 가장 큰 요청 시간 차이 |
| 엔진 이후 토큰당 시간 | 2.305ms | 0.909ms | prefill 개선과 별개로 약 2.54배 비용 |
| 엔진 요청 완료 | 506.709ms | 35.549ms | 쌍별 시간 비율 중앙값 14.254 |
| HTTP 요청 완료 | 512.912ms | 35.715ms | 쌍별 시간 비율 중앙값 14.361 |

첫 토큰 시간이 Riley 요청 완료 시간의 약 86%다. 단, TTFT에는 prefill 외 첫 토큰 처리·스케줄링도 포함되므로 435ms 전체를 특정 GPU 커널 비용으로 단정하지 않는다. `TTFT + 31 × TPOT`로 보면 TTFT만 7ms까지 줄여도 약 78.4ms가 남는다. 따라서 prefill만 고쳐서는 목표를 달성할 수 없다. HTTP와 엔진의 중앙값 차이 역시 서로 다른 실행에서 얻었으므로 정확한 HTTP 오버헤드로 빼서 해석하지 않는다.

측정과 소스로 확인한 사실:

- `graph_decode_full.rs`의 owned executor는 입력 토큰 1개만 허용한다. 현재 profile 문서도 batch token budget 및 prefill chunk size 1을 요구한다.
- 128개 prompt와 32개 output 요청은 prompt 128회와 후속 decode 31회, 총 159회 실행에 해당한다. 측정 결과의 30요청/4,770 iterations와 일치한다.
- `graph_resources.cu`의 profile 2 GEMM은 canonical GEMM 뒤 prefill 보정 GEMM을 추가한다. `graph_numerics_precise.cu`의 보정 커널은 position 128 이상에서 바로 반환하지만 decode graph에 launch 자체는 남는다.
- 현재 공통 graph는 각 실행에 head·argmax 및 결과 전송을 포함한다. prompt 중간 위치에는 외부 생성 토큰을 내보낼 필요가 없으므로 분리 검토 대상이다.
- GPU event 구간과 커널별 비용은 미측정이다. 중복 GEMM·작은 launch·D2H·동기화 각각의 비용 비중은 아직 가설이다.

## 유지할 계약

1. 기존 기준 입력의 생성 32토큰은 vLLM과 정확히 일치해야 한다. 출력 길이·EOS·greedy·토크나이저·모델 해시를 변경해 속도를 얻지 않는다.
2. BF16 split-K 부분합의 반올림 위치, RMSNorm/FMA 순서, FP32 residual 수명, attention 누적 순서를 유지한다. 행 수를 늘리면서 GEMM 알고리즘이 달라지는 것을 자동 허용하지 않는다.
3. scheduler가 소유한 실제 KV와 비연속 physical block mapping을 사용한다. 외부 KV 주입·oracle 텐서 사용·eager fallback은 금지한다.
4. prefill과 decode가 같은 부모 버퍼/KV를 사용할 경우 하나의 aggregate owner가 수명과 lease를 관리한다. 별도 graph owner가 독립적으로 같은 자원을 해제하거나 동시에 실행하지 않는다.
5. replay마다 staging을 새로 쓰며 완료 전 결과·KV 유효 길이를 공개하지 않는다. 실패한 실행은 poison 처리하고 재시도나 부분 결과 공개를 하지 않는다.
6. 취소·disconnect·재사용·close 후 allocation 0 검증을 유지한다. 배치 prefill 도입으로 취소 반응 시점이 바뀌면 명시적으로 검증하고 기존 API 보장을 약화시키지 않는다.
7. 새 구조는 별도 implementation/graph signature로 식별한다. 산술 계약을 바꿔야 한다면 기존 수치 ID를 재사용하지 않고, 별도 정책·정확성 승격 없이는 성능 후보로 채택하지 않는다.

## 실행 순서와 완료 기준

의존성: PR-00 → PR-01 → PR-02 → PR-03 → PR-04 → PR-05. PR-00에서 얻은 비용 비중으로 PR-04의 세부 커널 우선순위를 정한다. 범위가 커져도 아래 완료 항목을 임의로 추가된 승인 단계로 나누지 않는다.

### PR-00: 비교 조건 정리와 비용 분해

수정 대상: `benchmarks/scripts/`, `benchmarks/lanes/vllm/riley_vllm_benchmark/adapter.py`, `crates/riley-server/src/benchmark.rs`, `crates/riley-server/src/bin/riley-profile.rs`, 필요한 CUDA 진단 훅.

- 결과 폴더의 임시 runner를 재사용 가능한 진단 runner로 옮긴다. GUI 유지 조건을 명시 필드로 기록하고 vLLM raw의 오래된 environment ID를 실제 v2 호스트/조건 ID와 일치시킨다. 닫힌 schema 변경이 필요하면 버전과 checker를 함께 변경한다. 기존 결과는 수정하지 않는다.
- TIME_WAIT를 허용하면서 활성 listener는 거부하는 probe, 동일한 실행 전 냉각, 실행 실패·제외 이유 기록을 정식화한다. 검증 실패를 정상 요청으로 집계하지 않는다.
- 엔진·HTTP 모두 GUI 유지/유휴 메모리 512MiB, 다른 CUDA compute 프로세스 없음, 시작 전 48°C 이하 대기를 동일 적용한다. 기존 256MiB 결과와 구분한다.
- 기존 Riley 후보와 vLLM을 이 통일 조건에서 다시 기준 측정한다. 이후 모든 후보는 새 기준과 비교한다.
- 측정용 release와 별도 진단 실행에서 prefill/decode의 replay 수, GPU event span, host pack/launch/wait/read/sample 비용 및 전송 바이트를 분리한다. 가능하면 타임라인으로 커널 수와 동기화를 대조한다. GPU 도구가 없으면 설치를 전제로 막지 말고 이벤트·호스트 측정으로 가능한 범위를 기록한다.
- 관측 훅이 있는 실행의 시간을 최종 성능 수치로 사용하지 않는다. 미측정 값을 0으로 쓰지 않는다.

완료: 재현 가능한 기준 결과, 비용 분해 표, 전송/launch 목록, runner 실패 경로 검증. 어떤 항목이 미측정인지 명확해야 다음 커널 변경을 선택할 수 있다.

### PR-01: prefill/decode graph 분리와 중복 GEMM 제거

수정 대상: `kernels/src/graph_resources.cu`, `kernels/src/graph_numerics_precise.cu`, `kernels/src/ffi_internal.hpp`, `crates/riley-cuda/src/{ffi.rs,graph_resources.rs}`, `crates/riley-runtime/src/llama/graph_decode_full.rs`, `executor/graph_registry*.rs`.

- 먼저 M=1 산술을 유지하며 prefill·decode 실행 구조를 분리한다. prefill에는 수치 보정 GEMM만, decode에는 해당 decode GEMM만 남긴다.
- canonical 출력을 다른 소비자가 읽지 않는지 확인한 후 중복 연산을 제거한다. 하나의 실행 중 pos 값을 바꿔 캡처 당시 분기만 믿는 방식 대신, scheduler stage와 graph signature에 맞게 graph를 선택한다.
- decode에서 즉시 반환하는 prefill 보정 커널 launch를 제거한다. 부모 버퍼 공유는 단일 owner 아래 순차 실행으로 관리한다.
- 기존 v1 후보는 보존한다. 새 graph topology/implementation ID, 노드 수와 lease 원장을 기록한다.

완료: 단일 토큰 각 단계의 중간 텐서 및 최종 생성 토큰 일치, 잘못된 stage/graph 선택 거부, 완료·취소·close 검증, 중복 및 no-op launch 제거 증거. 단기 성능은 기준 대비 유의미하게 감소하거나 계측 잡음 범위여야 하며 5% 이상 회귀하면 원인을 해결한 뒤 진행한다.

### PR-02: P128 일괄 prefill의 CUDA 실행 구현

수정 대상: `kernels/src/`의 새 prefill 연산 파일, `kernels/CMakeLists.txt`, `ffi_internal.hpp`, CUDA FFI/resource 계층, runtime의 prefill 준비·metadata·graph signature 코드.

- 처음에는 고정 P128·c1만 구현한다. 128행 embedding → 30개 layer의 norm/QKV/causal attention/MLP → 마지막 prompt 행의 head로 진행한다. prompt 중간 행에는 head/argmax/전체 logits D2H를 수행하지 않는다.
- position·causal mask·RoPE·KV mapping을 모든 행에 정확하게 적용한다. decode가 읽을 실제 KV pool에 직접 기록한다.
- split-K BF16 부분합 순서를 유지하는 다중 행 GEMM을 먼저 검증한다. 단순히 cuBLAS에 M=128을 전달하면 같은 결과가 나온다고 가정하지 않는다. attention 누적 순서도 별도로 검증한다.
- prefill scratch와 graph 자원을 준비 단계에서 할당한다. replay 중 재할당 없이 사용하고 종료 시 해제한다.
- 취소 허용 지연을 PR-00/현행 API에서 확인한다. P128 한 번 실행이 이를 넘으면 32/64토큰 chunk 경로를 선택하고 각 경계의 KV·mask·수치 계약을 재검증한다. 속도만을 위해 취소 보장을 완화하지 않는다.

완료: 실제 P128 prefill KV 및 마지막 logits의 기준 대조, 비연속 block/경계·반복 입력·다양한 토큰의 검사, 이어지는 decode 31회에서 기존 32토큰 일치. capture/steady-state 메모리와 실제 replay 수를 제출한다. 중간 텐서 불일치가 있으면 최초 layer/operator부터 조사하며 tolerance를 넓혀 통과시키지 않는다.

### PR-03: scheduler·서버·벤치마크에 일괄 prefill 연결

수정 대상: `crates/riley-runtime/src/llama/{batch_executor.rs,graph_decode_full.rs,generation.rs}`, `crates/riley-runtime/src/generation.rs`, `crates/riley-server/src/{engine.rs,benchmark.rs,main.rs,bin/riley-profile.rs}` 및 직접 연관된 metadata/batch plan 경로.

- 현재 budget/chunk=1 제한을 새 후보에 한해 해제한다. prompt 128개를 한 번 또는 검증된 chunk로 계획하고, 이후 M=1 decode graph로 전환한다.
- prompt 소비 수·KV valid length·첫 생성 토큰 commit이 중복 또는 누락되지 않도록 stage 전환을 원자적으로 처리한다. 전체 prefill 1회라면 128 prompt replays → 1회, 후속 decode 31회가 기대 구조다.
- prefill 진행 중 취소/오류 시, GPU 완료를 확인한 뒤 예약 KV를 회수한다. 사용자에게 출력되지 않은 prefix를 다음 요청에서 재사용하지 않는다.
- 서버와 engine-only runner가 동일 실행 정책을 사용하게 하고, unsupported shape는 공개 전에 거부한다. 기본 정책은 유지한다.

완료: 실제 HTTP streaming/nonstreaming, CPU/GPU greedy, 취소 전·실행 중·첫 토큰 이후, 취소 후 재요청, 자원 정리 검증. 기존 prefix-23 테스트는 삭제로 해결하지 않고 새 배치 경계의 동등한 무공개·KV 회수 검증과 함께 재정의한다. 새 correctness receipt를 발급한 후에만 성능 후보로 진행한다.

중간 목표: TTFT 25ms 이하 및 decode 회귀 없음. 이는 설계 목표이며 달성 예상치가 아니다. 미달이면 비용 분해로 원인을 확인하고, 모델/동시성 확대로 넘어가지 않는다.

### PR-04: decode 구간과 결과 전송 최적화

수정 대상: `kernels/src/{graph_resources.cu,graph_numerics.cu,graph_numerics_precise.cu}`, CUDA transfer/output API, `graph_decode_full.rs`, `crates/riley-server/src/engine.rs`의 greedy 경로.

- PR-00의 비용 상위 항목부터 한 번에 하나씩 변경하고 대조한다. graph를 쓴다는 사실만으로 launch 비용이 제거됐다고 가정하지 않는다.
- GPU greedy에서 필요한 token/status만 D2H하고 전체 logits는 진단·CPU sampling 경로에 유지하는 방안을 검증한다. 기존 `OwnedLlamaDecodeExecutor::execute`의 logits 반환 계약은 별도 API/선택으로 보존한다. 새 출력 layout은 버전화한다.
- QKV 또는 gate/up 결합, 작은 norm/residual 연산 결합, decode GEMM 개선은 산술 순서와 수명 유지가 확인된 것부터 적용한다. 결합 GEMM은 reduction 변경 가능성이 있으므로 자동 승인하지 않는다.
- metadata pack/H2D 및 host 완료 대기 중복을 제거한다. 완료 전 읽기 금지와 요청별 취소/stop 처리를 유지한다. 여러 생성 토큰을 무조건 묶어 실행하는 변경은 이번 기본안에 넣지 않는다.

완료: 출력 토큰/내부 텐서 대조, error/status 전파, CPU/GPU greedy, 반복 replay와 누수 검증. decode TPOT 0.9ms 이하를 1차 목표로 삼되 최종 요청 전체 비교에서 이득을 확인한다.

### PR-05: 고정 후보 재검증과 최종 성능 비교

수정 대상: benchmark runner/checker, profiling 문서, 새 results 디렉터리 및 상태 보고서.

- 기존 후보, 새 후보, vLLM의 source/binary/model/tokenizer/환경 해시와 정확성 보고서를 각각 고정한다. 비교 시 source dirty를 허용하거나 기존 receipt를 새 바이너리에 재사용하지 않는다.
- 기존 Hello×128/32 셀을 유지하고, 별도의 다양한 P128 입력으로 정확성 회귀를 확인한다. 다른 입력이 자연 EOS로 짧아지면 별도 EOS 정책의 correctness 검사로 기록하고 고정 길이 성능 셀과 섞지 않는다.
- 각 비교는 같은 GUI·냉각·warmup·요청 조건에서 5개 새 프로세스 쌍과 30회/실행을 유지한다. 기존 Riley 대비 개선과 vLLM 대비 차이를 각각 제시한다. instrumentation은 비활성화한다.
- engine/HTTP의 TTFT, TPOT(HTTP는 실제 토큰 경계가 입증될 때만), E2E, 처리량, tail 지연, peak VRAM 및 시작/종료 상태를 보고한다. 집계 규칙과 실패한 실행을 모두 공개한다.

완료 판정:

| 구분 | 기준 |
|---|---|
| 정확성 | 고정 기준 생성 토큰 전부 일치, KV/취소/재사용/누수 검증 통과 |
| 구조 개선 | 중복 GEMM 제거, prefill replay 감소, GPU/host 비용 증거 확보 |
| 기존 Riley 대비 개선 | 같은 조건의 기준보다 개선, tail/memory 회귀가 있으면 비용과 채택 여부 명시 |
| vLLM 우위 목표 | 엔진·HTTP 각각 쌍별 E2E 비율 중앙값 ≤0.90, 5쌍 모두 <1.0; TTFT/TPOT 회귀도 별도 보고 |
| 주장 범위 | SmolLM2 c1/p128/o32 GUI 유지 조건에 한정; 5쌍 관측 범위를 신뢰구간으로 표현하지 않음 |

10% 우위는 채택 목표이지 보장된 개선율이 아니다. 관측 변동 폭이 차이와 비슷하면 추가 쌍으로 검증하고 우위를 주장하지 않는다. 목표 미달이면 실제 결과와 남은 비용을 보고하며 “성능 개선 완료”로 닫지 않는다.

## 위험과 되돌리기

| 위험 | 대응 | 되돌리기 |
|---|---|---|
| M>1 GEMM/attention 수치 변화 | 최초 불일치 tensor 및 reduction 순서 추적 | 새 prefill 경로 비활성화, 기존 v1 사용 |
| 공유 graph/KV 수명 오류 | 단일 owner, 순차 실행, invalid-stage 및 close 검사 | 새 topology 후보 제외 |
| 취소 지연·KV 공개 오류 | 완료 전 공개 금지, chunk 선택, 취소 후 재사용 검증 | 기존 M=1 경로 복구 |
| scratch/캡처 VRAM 증가 | peak 및 해제 측정, 명시 자원 한도 | chunk 축소 또는 후보 제외 |
| 측정 도구/GUI 잡음 | 동일 조건·냉각·반복·실패 기록, 원본 보존 | 새 실행 묶음으로 재측정 |

각 PR은 독립 구현 ID와 correctness receipt를 갖게 한다. 기존 기준 소스와 release 바이너리를 덮어쓰지 않는다. 로컬의 관련 없는 dirty 변경과 `crates/riley-model` 변경은 보존한다. 원격 검증이 필요한 구현 단계에서는 기존에 합의한 GUI 유지 조건을 사용하고, Blender 일시 종료가 필요한 측정 후에는 기존 실행을 복구한다.

## 최종 산출물과 실행 범위

필수 산출물은 ① 비교 조건/계측 보고서 ② graph 분리·중복 제거 ③ 배치 prefill CUDA 구현 ④ 실제 scheduler/HTTP 연결 ⑤ decode 비용 절감 ⑥ 정확성 receipt와 동일 조건 최종 성능 보고서다. 총 6개 작업 단위이며, 구현 착수 후 이 목록을 기준으로 완료 여부를 추적한다. 상세 함수 분리가 필요해도 그 자체를 새로운 사용자 승인 단계로 만들지 않는다.

이 문서 작성 시점에는 코드를 변경하거나 GPU를 다시 사용하지 않았다. 구현 승인 후 PR-00부터 순서대로 실행한다. 전체 일정은 PR-00의 계측과 PR-02의 수치 일치 난이도를 확인하기 전에는 확정하지 않는다.
