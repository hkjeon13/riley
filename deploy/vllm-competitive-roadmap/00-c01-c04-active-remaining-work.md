# C01~C04: 지금 착수 가능한 잔여 개발 목록

**기준:** `main@05d50a3` / 2026-09-03 점검
**목적:** C05의 완료 여부와 별개로, C01~C04 중에서 저장소 안에서 바로 개발할 수 있는 미완료 작업만 분리한다.

**구현 갱신 (2026-09-03):** 이 문서에 포함했던 C01-A~C01-C와 C04-A~C04-C의
source-level 작업은 구현 및 CPU/fake-adapter 검증까지 완료했다. 실제 candidate/version pin,
remote GPU execution, C02 qualification, C03-B GPU corpus, C04 GPU parity·성능 측정,
M4/M5 closed report는 여전히 이 문서의 범위 밖이며 완료로 취급하지 않는다.

## 결론

C05가 완료된 것은 CUDA Graph의 **primitive ownership ABI와 개별 primitive의 GPU parity** 범위다. C01~C04 전체가 완료되었다는 뜻은 아니다.

- C01·C02는 경쟁 측정과 release qualification이라는 별도 축이다.
- C03은 현재 승인된 CPU fuzz 범위의 구현은 완료됐지만, formal completion에 적힌 더 넓은 general grammar는 별도 승인 범위다.
- C04는 세부 source extraction(C04-1~31)은 끝났지만, executor owner와 orchestration을 분리해 기존 `batch_executor.rs`를 얇은 facade로 만드는 최종 구조 작업이 남아 있다.

따라서 이 문서의 **활성 작업 목록이었던 C01 실행 도구화 3개와 C04 구조 분리 3개는
source-level로 모두 끝났다.** C02 및 C03에는 이 필터에서 바로 시작할 독립 작업을 넣지 않는다.

## 포함·제외 기준

### 포함

- 현재 checkout에서 구현·단위 테스트할 수 있는 source-level 작업
- C05/C06/C07 성능 경로나 C02 candidate qualification 없이도 완료 기준을 만들 수 있는 작업
- 실제 host 권한, root provisioning, GPU operation 승인을 요구하지 않는 작업

### 제외

- 실제 GPU launch, candidate freeze, Gate E, semantic replay, C02 qualification
- 실제 vLLM version/candidate pin, Tier D dry-run, M4/M5 closed report
- C03-B GPU corpus 실행과 C04의 GPU parity·before/after 성능 측정
- 별도 승인 없이는 범위를 넓힐 수 없는 C03 general grammar/fuzzer 확장

제외한 항목은 사라진 것이 아니라, 이 문서의 “지금 개발할 backlog”에 넣지 않은 것이다.

## 요약

| 영역 | 현재 상태 | 지금 할 일 | 이번 목록에서 제외한 이유 |
| --- | --- | --- | --- |
| C01 | materialized lane, injected adapter, append-only raw journal/claim gate 구현 완료 | 없음 | 실제 campaign 실행은 운영 절차 |
| C02 | P0/P1 source closure 완료. P2는 native guardian/root provisioning 전제. | 없음 | review·administrator provisioning·권한 승인 대기 |
| C03 | 현재 승인된 narrow CPU contract 구현 및 source contract는 완료. | 없음 | GPU 실행은 C02 뒤, 더 넓은 grammar는 별도 승인 범위 |
| C04 | C04-1~31 및 owner/dispatch/facade 최종 분리 완료 | 없음 | GPU parity/성능 측정만 별도 절차로 제외 |

## 구현 완료 1 — C01 경쟁 캠페인 실행 도구화

**완료 범위:** reviewed template + immutable input을 materialized lane으로 만들고, injectable
process adapter가 immutable plan의 AB/BA 순서를 소비해 append-only raw JSONL과 checker까지
연결한다. 이 범위에는 remote transport 구현, 실제 GPU launch, concrete candidate/version pin이 없다.

### C01-A. 실행 가능한 lane materialization/validation

**구현:** `benchmarks/competitive/scripts/materialize_lane.py`가 create-only materialized lane을
생성한다. immutable input file SHA-256과 expanded argv SHA-256을 receipt에 넣고, checker/plan은
reviewed template과 input을 다시 materialize해 hand-edited argv를 거부한다.

현재 Riley lane은 `contract-only`, vLLM lane은 `campaign-pin-required` template다. 실제 version, revision, lock, command를 이 문서에서 선택하거나 pin하지 않는다. 대신 supplied immutable inputs를 받아 **campaign-local executable lane**을 만들고, 하나라도 비어 있거나 hash가 맞지 않으면 fail closed하는 source 구현을 추가한다.

범위:

- immutable Riley/vLLM artifact, model/tokenizer, runtime option dump를 입력으로 받는다.
- plan의 campaign/lane/cell identity를 변경할 수 없게 검증한다.
- 실행 전에는 `available` lane만 만들고, raw template을 실행 대상으로 쓰지 못하게 한다.
- 실제 candidate 또는 vLLM release 선택은 외부 campaign admission 단계로 남긴다.

완료 기준:

- 누락·dirty·hash mismatch·plan override를 거부하는 unit test가 있다.
- fake immutable input으로 materialized lane을 만들 수 있다.
- 실제 pin 없이도 source test가 동작한다.

근거: [C01 상태](01-vllm-win-contract.md), [현재 lane 계약](../../benchmarks/competitive/README.md)

### C01-B. immutable plan 소비용 execution adapter

**구현:** `benchmarks/competitive/scripts/execute_campaign.py`가 transport-agnostic
`InvocationExecutor`를 받아 plan 순서만 실행한다. campaign-wide journal lease, pre-start retry,
timeout terminate→kill→close, stdout/stderr cap, environment receipt failure cleanup을 갖는다.

`run_campaign.py`는 실행 계획만 만들며 engine을 시작하지 않는다. immutable plan을 읽어서 Riley와 vLLM arm을 AB/BA 순서 그대로 실행하는 adapter를 추가한다. 이 작업은 **adapter source와 mock process lifecycle**까지만 포함하며, 원격 GPU host에서 실제 실행하는 것은 포함하지 않는다.

범위:

- plan에 기록된 arm/order/run만 소비하고, 실행 중 matrix·threshold·environment를 바꾸지 않는다.
- arm마다 fresh process lifecycle, timeout, stdout/stderr size limit, 실패 cleanup을 제공한다.
- preflight 결과를 raw record에 연결하되, preflight failure는 measured sample로 섞지 않는다.
- remote transport는 injectable interface로 두고 fake executor로 unit/integration test한다.

완료 기준:

- AB/BA schedule drift, 재시도 후 stale process, timeout/cleanup, duplicate run을 테스트한다.
- adapter가 계획 밖 command와 mutable option override를 거부한다.
- 실제 SSH/GPU credential 없이 fake executor 테스트가 가능하다.

근거: [plan-only 실행기 경계](../../benchmarks/competitive/README.md), [C01 구현 단계](01-vllm-win-contract.md)

### C01-C. append-only raw JSONL producer와 adapter 연결

**구현:** `raw_journal.py`가 create-or-append, fsync, file lock, ordered receipt chain을 제공하고,
claim checker는 단일 JSONL의 모든 journal field 및 lane의 executable/dependency-lock environment
receipt를 필수로 검증한다. legacy/import raw는 favorable claim이 될 수 없다.

현재 checker는 raw JSONL을 검증하지만, adapter가 source-of-truth record를 생산하는 경로는 없다. 각 plan run의 lifecycle을 `riley.competitive.raw.v1` record로 append-only 기록하고 checker 입력으로 연결한다.

범위:

- campaign/lane/cell/run/order/request/environment/success/metric/token mismatch/terminal 상태를 기록한다.
- record collision, partial write, out-of-order append, failed arm의 percentile 오염을 fail closed한다.
- raw record와 closed report의 provenance join을 fake engine 결과로 테스트한다.
- raw evidence root의 실제 운영 보존 정책은 campaign 실행 단계로 남긴다.

완료 기준:

- pass, failed, incomparable fixture가 동일한 checker 결과를 재현한다.
- append-only 및 no-overwrite 위반을 테스트한다.
- adapter 종료 후 raw record만으로 plan/run identity를 재검증할 수 있다.

근거: [raw record 계약](../../benchmarks/competitive/README.md), [C01 미완료 범위](01-vllm-win-contract.md)

## 구현 완료 2 — C04 executor 최종 구조 분리

**완료 범위:** resource owner, borrowed execution dispatch, thin public facade의 source boundary를
분리하고 CPU contract/architecture test로 확인했다. GPU correctness, allocation snapshot과
before/after 성능 값은 이 완료 범위에 포함하지 않는다.

### C04-A. resource owner를 `executor/owner.rs`로 분리

**구현:** `PreparedLlamaBatchOwner`가 forward/shape variant/KV/RoPE/metadata/output workspace의
cold lifetime과 explicit close first-error precedence를 소유한다.

목표 구조에는 weight, paged KV, stream, forward workspace, close lifecycle을 소유하는 `owner.rs`가 정의돼 있지만 현재 파일은 없다. 이 ownership은 여전히 `PreparedLlamaBatchExecutor`와 `batch_executor.rs`에 집중돼 있다.

범위:

- weight/KV/stream/workspace의 단일 owner와 explicit close 순서를 독립 module로 이동한다.
- shape variant가 owner resource를 복제하지 않는 계약을 유지한다.
- poison/cleanup의 첫 오류 우선순위와 public error vocabulary를 바꾸지 않는다.
- CUDA Graph-specific behavior나 production default 변경은 넣지 않는다.

완료 기준:

- owner module이 scheduler/server를 import하지 않는 architecture boundary test가 있다.
- resource close 순서, poison, rollback error precedence의 CPU test가 있다.
- public Rust API/C ABI/CLI diff가 없다.

근거: [목표 module 구조](04-llama-executor-refactor.md), [ownership 원칙](04-llama-executor-refactor.md)

### C04-B. execute orchestration을 `dispatch.rs`로 축소

**구현:** hot metadata transport, command batch, fixed forward dispatch, output primitive body를
borrowed-resource `executor/dispatch.rs`로 이동했다. facade가 mutation/poison와 public output state를
계속 결정한다.

현재 `batch_executor.rs`가 `execute_packed`와 `execute_fixed_graph`를 보유한다. metadata transport, shape/plan 선택, command-batch lifecycle, output-ready/poison decision의 orchestration을 responsibility별 module로 옮겨 `dispatch.rs`를 실제 실행 orchestration 경계로 만든다.

범위:

- C04-1~31에서 이미 분리한 helper를 재사용하고 duplicate policy를 만들지 않는다.
- host preflight → metadata bind/upload → forward/primitive dispatch → output/poison 순서를 보존한다.
- sync/packed metadata와 greedy/logits output의 observable behavior를 characterizing test로 고정한다.
- C05 graph ABI는 호출 가능한 extension point로만 연결하며 graph fast-path 성능 변경은 하지 않는다.

완료 기준:

- failure injection, cancellation/commit failure, close 경로의 CPU contract가 유지된다.
- call graph가 batch facade → owner/dispatch/output/metadata로 단방향화된다.
- hot-path heap allocation을 늘리지 않았음을 source/unit instrumentation으로 확인한다.

근거: [분리 순서 7단계](04-llama-executor-refactor.md), [현재 C04 상태](04-llama-executor-refactor.md)

### C04-C. `batch_executor.rs`를 compatibility facade로 축소

**구현:** 기존 nominal public type/API를 유지하면서 `batch_executor.rs`에는 config, logical
iteration/output state, owner/dispatch 연결만 남겼다.

최종적으로 기존 파일은 public re-export와 compatibility facade만 남기거나 제거 가능해야 한다. 현재는 large owner와 execution body가 여전히 이 파일에 있다.

범위:

- public nominal type, error text/category, `riley_runtime::llama::*` export를 유지한다.
- old→new symbol map, module responsibility table, public API diff, before/after call graph를 리뷰 산출물로 추가한다.
- `batch_executor.rs`의 implementation ownership을 제거하고 thin facade로 바꾼다.

완료 기준:

- C04 architecture boundary test가 목표 dependency 방향을 고정한다.
- facade가 기존 public compile contract를 유지한다.
- C05/C06 graph owner를 별도 module로 추가해도 facade 변경이 필요 없는 extension point가 확인된다.

근거: [분리 순서 8~9단계](04-llama-executor-refactor.md), [C04 승인 기준](04-llama-executor-refactor.md)

## 이 목록에서 의도적으로 뺀 미완료 항목

| 영역 | 제외 항목 | 제외 사유 |
| --- | --- | --- |
| C01 | concrete competitor/candidate pin, Tier D GPU dry-run, 실제 raw capture, M4/M5 closed report | campaign admission·원격 GPU 실행·동일성 evidence가 필요함 |
| C02-P0/P1 | frozen candidate semantic replay와 qualification | P0/P1 source closure 뒤의 C02 실행 절차임 |
| C02-P2 | native guardian/warden/PID1/ledger 구현·root bundle 설치·GPU/Docker 승인 | reviewed design, administrator provisioning, no-GPU acceptance와 명시 승인 없이는 시작하면 안 되는 보안/운영 작업임 |
| C03 | C03-B GPU corpus 실행 및 formal acceptance | C02 actual qualification 뒤에만 가능함 |
| C03 | arbitrary/unbounded mixed-operation grammar, general/global shrinker, generalized multi-event fault injection | 현재 narrow CPU contract의 미구현이 아니라 별도 승인된 future grammar 범위임 |
| C04 | GPU correctness/allocation/performance before/after 5-pair gate | C02-qualified candidate와 GPU operation이 필요함 |

`02c-c02-p2-native-guardian-readiness.md`의 host 표기는 아직 `server-4096`이지만, 실제 대상은 `ai-assistant`다. 이는 향후 운영 문서 정정 항목이며, 위 source backlog의 완료 조건에는 포함하지 않는다.

## 후속 실행 순서

1. 별도 운영 절차로 C02 candidate qualification을 완료한다.
2. C03-B GPU corpus와 C04 GPU correctness/allocation/performance non-regression을 실행한다.
3. 승인된 remote transport와 concrete Riley/vLLM pin을 admission input으로 넣어 Tier D dry-run을 수행한다.
4. 그 뒤에만 실제 C01 campaign raw capture와 M4/M5 closed report를 생성한다.

## 점검 시 검증

- `PYTHONDONTWRITEBYTECODE=1 pytest -q benchmarks/competitive/scripts/tests`: 44 passed
- `cargo test -p riley-scheduler --test model_based_routing --test general_mixed_operation_routing --test general_mixed_program_routing --test bounded_mixed_program_routing --test inflight_mixed_program_routing --test gpu_fixed_corpus_contract`: 70 passed
- `cargo check -p riley-runtime --tests`
- `cargo test -p riley-runtime --lib source_contract_tests`: 28 passed
- `cargo test -p riley-runtime --test architecture_boundary`: 15 passed
- `git diff --check`: clean

위 결과는 C01 adapter/checker와 C04 source boundary, 현재 승인된 C03 CPU/source-contract 범위를
확인한 것이다. 실제 vLLM campaign, GPU corpus execution, C02 qualification 또는 C04 GPU parity
결과는 아니다.

## 완료 판정

이 문서에서 포함한 source work의 완료는 다음을 뜻한다.

- C01: 실제 GPU를 실행하지 않아도 plan이 executable lane·fake process·append-only raw record·checker report까지 fail-closed로 연결된다.
- C04: public contract를 바꾸지 않고 owner/dispatch/facade 경계가 분리되어 `batch_executor.rs`가 implementation owner가 아니다.

이 문서의 완료는 C02 qualification, M4/M5 승리, C03-B GPU acceptance 또는 C04 GPU 성능 non-regression을 뜻하지 않는다.
