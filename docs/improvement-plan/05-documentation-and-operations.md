# W5. 문서 정합성·관측·운영 검증 개선

작성일: 2026-11-13\
종합 계획: [진행 단계와 릴리스 조건](README.md)\
구성: W5-A는 즉시 가능한 문서 정리, W5-B는 GPU 회귀·운영 검증이다.

Qwen2.5-7B CUDA 서버에서 90초 사전 검증과 cache off·on 각 30분의 합성 혼합 부하로 응답·취소·할당 회수를 확인했다. [W5-B 실행 결과](05-mixed-soak-result.md)

## 1. 현재 문제와 위험 범위

| 항목 | 확인 내용 | 개선 이유 |
|---|---|---|
| 문서의 측정 시점 차이 | README에 10월 24개 모델 판정과 11월 장문·GSM8K 재검증의 날짜와 셀을 명시했다. | 시점과 워크로드를 기준으로 근거를 찾을 수 있음 |
| 기본값 설명 드리프트 | README·configuration에서 일반 CLI, fixed-tile 한도, 프로파일 기본값과 과거 KV chunk 값을 확인했다. | 사용자가 프로파일을 재현할 수 있음 |
| API 비용 설명 | api.md는 사용 조건에 따른 CPU logits 및 GPU summary 동작을 문서화했다. | 실제 비용을 기능 조합별로 확인할 수 있음 |
| 모델 지원 의미 혼합 | 누락된 두 모델을 README의 A100 판정 목록에 복구하고, model-support의 owned 경로 기록에 조사 시점을 표시했다. | 일반 native 성능과 pinned 모델의 HF 정확성에 각각 근거가 생김 |
| GPU CI 범위 | native-cuda workflow는 nightly compile, GPU runtime/memory smoke는 수동 opt-in | 모델 추론·routing·성능 회귀가 PR마다 자동 검증되지 않음 |
| 운영 근거 | 합성 트래픽 30분 soak 기록. 실제 production trace 미검증 | 실제 길이·도착·취소·기능 혼합의 차이를 모름 |
| prefix cache | 기준 배포 이미지의 캐시 활성 30분 시험에서 익명 RSS는 616KiB 증가했다. jemalloc 파생 이미지의 정밀 RSS는 캐시 활성에서 440KiB, 비활성에서 640KiB 감소했고 장문 232건이 일치했다. 사전 성능 회귀 기준은 통과했으나 RAG C8 seed 2의 첫 토큰 p95는 12.69% 증가했다. 별도 CUDA 진단의 메모리 오류 주입 4건도 통과했다. | RAG 지연 원인과 실제 장치 장애 뒤의 운영 복구 경로를 계속 확인함 |

근거: [README](../../README.md), [configuration](../configuration.md), [API](../api.md), [serving-validation](../serving-validation.md), [model-support](../model-support.md), [prefix-cache](../prefix-cache.md), [GPU CI](../../.github/workflows/native-cuda.yml).

## 2. W5-A: 문서를 현재 후보에 맞추는 방법

### W5-A1. 주장별 근거 목록 작성

성능 숫자·기본값·지원 모델·API 보장마다 다음을 연결한다.

`주장 → 대상 버전/프로파일 → 실제 소스 또는 실험 manifest → 측정 시점 → 지원 범위 → 제한`

단일 최신 숫자를 모든 표에 복사하지 않는다. 예를 들어 과거 qualification은 fixed/natural 8셀이고 최신 TP=2 서빙 검증은 64-token 셀이다. 새 결과가 좋다고 기존 8셀 qualification을 충족했다고 선언하지 않는다. README의 "32B 미충족"은 어떤 매트릭스의 판정인지 남기고 최신 별도 결과를 함께 설명한다.

2026-11-19에 아래 주장을 소스와 실험 기록에 대조했다.

| 사용자 문서 | 확인한 근거 | 적용 범위와 제한 |
|---|---|---|
| [README 벤치마크](../../README.md#performance) | [0.1.0 목표 재측정](../goal.md), 2026-11-14 | 24개 모델의 64토큰 `closed-c1`과 opt-in prefill burst RAG를 각 설정으로 표기 |
| [README TP=2](../../README.md#two-gpus-qwen25-32b-with-tensor-parallelism) | [TP 설계의 원래 매트릭스](../tensor-parallel-design.md#status-2026-11-01), [0.1.0 목표 재측정](../goal.md) | 원래 fixed/natural 8셀과 후속 fixed 64토큰 4셀에 각각의 판정 적용 |
| [CLI 실행 옵션](../configuration.md#execution-options) | [프로파일 resolver](../../crates/riley-server/src/inference_profile.rs), [서버 옵션](../../crates/riley-server/src/main.rs), [실행 구현](../../crates/riley-server/src/engine.rs) | 프로파일 공급값, 일반 CLI 기본값, 실험값을 각각 표기 |
| [컨테이너 기본값](../configuration.md#container-defaults) | [배포 entrypoint](../../deploy/docker/entrypoint.sh), [실제 전달 인자를 검증하는 CPU 테스트](../../ci/tests/test_distribution_entrypoint.py) | 컨테이너의 bind·모델 ID·프로파일·체크포인트 파일 크기 합계와 명시적 옵션 우선순위를 표기 |
| [logprobs](../api.md#log-probabilities) | [`sample_iteration` 구현](../../crates/riley-server/src/engine.rs) | GPU 요약 사용 여부는 해당 배치 조건에 의존. 공개 분포와 응답 형태는 같은 계약을 따름 |
| [모델별 지원과 독립 참조](../model-support.md) | [네이티브 A100 성능표](../model-coverage-a100-perf-table.md), [모델별 독립 참조](../model-support.md) | 각 기록의 체크포인트·길이·기능 조건에서 whole-model 일치와 성능 자격을 각각 판정 |
| [현행 운영 메트릭](../api.md#get-metrics) | [서버 응답 구조](../../crates/riley-server/src/service.rs), [엔진 bucket 정의](../../crates/riley-server/src/engine.rs), [A100 응답 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/soak.json) | `batch_shapes.buckets` 배열은 24개 슬롯이며 측정한 프로파일은 그중 21개를 구성했다. |
| [고정 RC 릴리스 soak 메트릭](../../benchmarks/soak/README.md#target-contract) | [릴리스 검사기](../../benchmarks/scripts/check_reliability_soak.py) | 과거 SmolLM2 경로에 고정 10개 슬롯을 적용한다. |
| [README 모델·프로파일](../../README.md#supported) | [24개 모델의 원래 판정](../model-coverage-a100-perf-table.md), [프로파일 확장·경계 검증](../../crates/riley-server/src/main.rs#L7777), [GSM8K 동일 입력 대조](04-long-accuracy-result.md#gsm8k-전수-품질-대조) | 24개 모델의 10월 성능 판정, 4개 모델의 11월 정확성, 일반 CLI와 fixed-tile 한도를 각각 명시 |

### W5-A2. 사용자 문서와 연구 기록의 역할 정리

| 문서 | 최종 역할 |
|---|---|
| README | 현재 버전의 시작 방법·지원 범위·핵심 결과·대표 제한 |
| configuration | CLI 실제 기본값과 profile이 덮어쓰는 값, Docker entrypoint 추가 기본값 |
| api | 필드별 지원·거부·stream 종료·schema/tool/logprobs 보장 |
| model-support/model-coverage | native 지원·owned exact·모델별 qualification을 구분한 표 |
| serving-validation/performance | 시점·빌드·설정이 명확한 측정 기록 |
| goal/CHANGELOG | 달성·미달·구조 한계 수용·기본값 변경의 결정 기록 |

과거 결과는 삭제해 현재 결과로 치환하기보다 역사적 구간임을 표시한다. native profile·owned exact·D128 route·Docker를 같은 기본 설정으로 설명하지 않는다. `--help`와 CLI resolver, 실제 `/v1/config`가 제공되는 경우의 결과, entrypoint를 함께 대조한다.

### W5-A3. 반복 가능한 문서 검증

링크·코드 경로·필수 지원표·기본값 일치는 가벼운 검사로 확인한다. CLI 전체 help를 무조건 snapshot으로 고정하기보다 중요한 옵션의 이름·기본값·프로파일 override를 검증한다. W1의 해시 결속 검사와 문서 의미 검사는 역할이 다르다.

README의 24개 성능 행을 A100 원본 요약과 대조하고 이번에 수정한 문서의 로컬 링크를 확인했다. `native_profile_cli_tests` 4건에서 기본값, 옵션 override, 고정 타일 상한과 KV 분할을 확인했고 `ci/check_serving_source_contract.py`도 통과했다. [프로파일 테스트](../../crates/riley-server/src/main.rs#L7763)

Production CPU의 Documentation 단계에서 개선 계획서의 상대 링크와 Markdown 제목, qualification manifest에 연결된 근거 파일의 SHA-256, 배포 입력을 대조한 두 Git 리비전의 객체를 검증한다. 로컬에서는 `python3.12 ci/check_improvement_evidence.py`를 실행한다. [검증 스크립트](../../ci/check_improvement_evidence.py) · [Production CPU 워크플로](../../.github/workflows/production-cpu.yml)

파생 할당기 시험은 패키징, API, 두 캐시 모드의 30분 부하, 장문 정확성, 기본 성능, 닫힌 RAG 확장 대조, 고정 도착 대조와 메트릭 조회 두 시험을 `current_allocator_trial`의 이미지 ID에 연결한다. CI에서 패키징 증명의 소스와 실행 파일, GPU 부하·정확성의 공통 증명, 지연 시험의 이미지·실행 인자·할당기 설정을 대조한다. 워밍업 재시험은 첫 시험의 검산 해시, 체크포인트 및 부하별 완료 건수와 맞아야 한다. [후보 매니페스트](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json) · [릴리스 판정](06-candidate-qualification.md)

같은 검사에서 README의 `Status` 버전을 Cargo 워크스페이스 버전과 맞추고, 10월 A100 동시성 32·고정 길이 표의 24개 모델을 보존된 표와 대조한다. 성능 요약은 동시성 8–32의 처리량·TPOT 각 144셀, TTFT 192셀, 단일 클라이언트 처리량 48셀에서 다시 계산한다. 당시 기록에 적힌 AWS 원본 경로는 이번 점검에서 확인되지 않아 성능 검사의 근거는 저장소에 보존된 모델별 측정 표다. [워크스페이스 버전](../../Cargo.toml) · [README 성능표](../../README.md#performance) · [보존된 측정값과 당시 실행 이름](../model-coverage-a100-perf-table.md#측정-출처)

Production CPU의 계약 테스트는 실제 Rust 프로파일 resolver에 `serve --inference-profile native-bf16-varlen-v1`을 전달한다. 반환된 attention 모드, 최대 문맥, 스텝 예산, prefill 조각, 이벤트 발행 값을 사용자 문서의 표와 맞춘다. 컨테이너가 선택하는 프로파일과 명시 옵션 우선순위는 entrypoint 계약 테스트에서 확인한다. [기본값 표](../configuration.md#execution-options) · [프로파일 검사](../../ci/tests/test_improvement_evidence.py) · [entrypoint 검사](../../ci/tests/test_distribution_entrypoint.py)

성능표를 생성한다면 raw→검증된 집계→표의 입력·출력·버전이 명확한 작은 생성기를 둔다. 과거 결과와 새 후보를 섞거나 집계 중 실패 셀을 제거하지 않는다. 이 생성기는 제안이며 현재 구현되어 있지 않다.

## 3. W5-B: 실제 GPU 회귀 검증을 추가하는 방법

### W5-B1. 자원과 실행 조건 확정

사용 가능한 전용 GPU runner, 모델 접근·revision, 공간·다운로드 cache, 실행 시간을 먼저 정한다. 과거 GPU 6·7 사용 기록을 현재 상시 사용 권한으로 간주하지 않는다. 다른 사용자의 프로세스는 종료하지 않고 혼잡하면 대기 또는 명시적 미실행으로 처리한다.

runner 동시성은 GPU별로 제한하고 각 실행에 별도 포트·디렉터리·서버 PID를 부여한다. 모델 파일은 검증된 cache를 재사용하고 cache miss·접근 실패를 테스트 통과로 처리하지 않는다.

### W5-B2. 세 층으로 분리

| 층 | 시점·대상 | 검증 내용 | 결과 해석 |
|---|---|---|---|
| 빠른 GPU correctness | 영향 경로 PR 또는 merge 후보, 작은 대표 모델 | 실제 모델 추론, mixed stage·취소, batch invariance, schema/tool/logprobs | 미실행은 미실행. CPU 성공으로 대체 금지 |
| 정기 회귀 | 자원 예약한 주기·기본값 변경 후보 | 대표 모델군, closed/RAG/reason/prefix, 동일 baseline 비교 | 공유 GPU·noise 상태를 기록하고 threshold 전에 유효성 확인 |
| 릴리스 qualification | 고정된 최종 후보 | affected 전체 모델, TP=2, 품질, API, soak, 패키징 | 동일 후보에 묶인 최종 근거 |

대표 모델은 2-KV-head GQA, Qwen3 norm, MHA, 큰 14B를 단계적으로 포함한다. 작은 모델 하나의 통과를 전 모델 통과로 확대하지 않는다. runner가 없으면 설정 파일만 만든 상태와 실제 GPU 성공을 구분한다.

#### 배포 이미지의 짧은 GPU smoke

승인된 이미지의 CPU 검사 보고서, GPU UUID, 준비된 체크포인트와 manifest SHA-256을 넘긴다. 출력 경로는 해당 실행의 새 디렉터리다. `--tools`를 사용하면 호환 체크포인트의 전체 API 도구 호출 계약도 검사한다. [배포 이미지 검사기](../../ci/check_distribution_gpu.py)

```bash
python3.12 -B ci/check_distribution_gpu.py "$CPU_REPORT" "$GPU_UUID" \
  "$MODEL_DIR" "$MODEL_MANIFEST_SHA256" "$RUN_OUTPUT" --tools --serving-smoke
```

채팅 템플릿이 도구를 지원하지 않는 체크포인트는 `--expect-tools-unsupported --serving-smoke`로 실행한다. 혼합 부하에도 동일한 도구 요청을 보내며 그 응답은 HTTP 400, `unsupported_parameter`, `param=tools`와 템플릿 오류 메시지를 대조한다. 구형 logprobs는 완료 토큰 사용량과 배열 길이를 맞추고, 자연 종료에서는 출력 상한 전에 멈춘 응답도 검사한다. [API 검사기](../../benchmarks/live/api_check.py)

하나의 loopback 컨테이너에서 API 검사 뒤 초당 평균 8건의 45초 혼합 부하를 실행한다. 기본 KV 풀은 4,096블록이며 `--smoke-kv-blocks`에 모델별 용량을 지정할 수 있다. 명시한 값은 실행 보고서의 서버 인자에 기록된다. Prefix cache는 128블록이다. 스키마·도구·logprobs·장문 요청의 완료와 의도적 연결 종료를 관측한다. Drain 뒤 활성 요청·KV와 GPU 및 pinned 할당의 회수도 확인한다. 30분 지연 드리프트와 정밀 RSS는 `--mixed-soak` 모드에서 판정한다. [혼합 부하 검사기](../../benchmarks/live/soak.py)

A100 7번의 검증 이미지에서 2-KV-head GQA, Qwen3, MHA, 14B에 대해 체크포인트별 도구 지원 계약과 혼합 부하를 검증했다. 현재 실행은 전용 호스트 CLI로 재현한다. GitHub의 자동 GPU 실행은 runner와 체크포인트 경로가 확보된 뒤 설정한다. [대표 모델 결과](05-mixed-soak-result.md#배포-이미지의-대표-모델-45초-smoke)

#### 한 GPU iteration의 prefill·decode 대조

`--mixed-stage`는 단독 실행 두 건과 동시 실행 세 쌍을 같은 배포 컨테이너에서 비교한다. 배치 예산 512, 요청별 prefill 한도 256, 활성 요청 한도 2를 명시한다. 첫 요청의 생성 토큰을 8개 이상 받은 뒤 두 번째 요청을 보낸다. 이 조건에서 커밋된 실제 행 257개를 관측하면 prefill 256행과 decode 1행이 한 GPU iteration에 포함된 것이다. 패딩 행 수와 GPU 실행 시간도 함께 기록한다. 스트림의 입력·생성 토큰 ID, 텍스트, 종료 사유와 사용량을 단독 실행 값과 대조한다. [검사기](../../benchmarks/live/mixed_stage_check.py) · [커밋된 shape 기록](../../crates/riley-server/src/engine.rs#L4362) · [prefill과 decode 행 제한](../../crates/riley-scheduler/src/scheduler.rs#L1827)

```bash
python3.12 -B ci/check_distribution_gpu.py "$CPU_REPORT" "$GPU_UUID" \
  "$MODEL_DIR" "$MODEL_MANIFEST_SHA256" "$RUN_OUTPUT" --tools --mixed-stage
```

#### 32B 모델의 TP=2 장문 검증

`--tp-holdout`에는 rank 1의 GPU UUID와 고정 입력 파일을 넘긴다. 일반 GPU UUID 인자는 rank 0이다. 두 GPU의 프로세스 사용 상태를 확인하고 컨테이너 안에서는 같은 순서의 CUDA ordinal `0,1`로 실행한다. 시험 설정은 `native-bf16-varlen-v1`, 8,192토큰 문맥, KV 2,048블록이다. API 검사 뒤 11개 장문·경계 입력을 C1, C8, C32에서 비교한다. C32는 짧은 일곱 입력을 반복해 요청 32건을 구성하며 실제로 관측한 활성 요청의 최댓값을 함께 기록한다. [32B 장문 검사기](../../benchmarks/live/tp_holdout_check.py) · [TP 장치 지정 계약](../tensor-parallel-design.md#cli)

```bash
python3.12 -B ci/check_distribution_gpu.py "$CPU_REPORT" "$GPU_UUID" \
  "$MODEL_DIR" "$MODEL_MANIFEST_SHA256" "$RUN_OUTPUT" --tools \
  --tp-holdout "$PEER_GPU_UUID" "$HOLDOUT_WORKLOAD"
```

배포 이미지의 Qwen2.5-32B-Instruct 기준 실행에서 API 40건을 통과했고 C8·C32 응답 43건이 C1 기준과 일치했다. [실행 결과와 검증 범위](05-mixed-soak-result.md#배포-이미지의-32b-tp2-장문-검증)

## 4. 관측을 통해 새 문제를 찾는 방법

현재 `/metrics`는 JSON의 요청·KV·할당·batch shape·취소·과부하 관측을 제공한다. 다음은 추가 제안이며 현재 endpoint가 모두 제공하는 필드는 아니다.

| 지표 | 찾을 문제 | 연계 |
|---|---|---|
| TTFT·TPOT·inter-token gap 분포, queue wait | 평균에 숨은 tail·긴 prompt 간섭 | W3 |
| 단계별 host 시간·GPU idle·in-flight 수 | 중첩 이득·새 직렬 병목·유실 상태 | W2 |
| 선택 버킷·prompt/decode 행·graph fallback | padding·graph 예산·잘못된 실행 경로 | W2·W3 |
| KV free/held/cached, eviction·hit·대기 | cache 누수·부족·refcount 문제 | W2·W3 |
| 오류 코드·timeout·취소·종료 정리 시간 | overload와 내부 오류 구분·수명 문제 | W1·W2 |
| schema 거부·완성/length·fallback | 계약 위반·기능별 비용 | W4 |

필드 추가는 기존 JSON consumer와 호환되게 설계하고 histogram bucket·단위·카운터 수명을 명시한다. raw prompt·tool argument·키를 메트릭 label에 넣지 않는다. 요청 ID·임의 모델 revision 같은 고카디널리티 값은 bounded 진단 근거로 분리한다.

관측 on/off의 CPU·RSS·GPU·지연 비용을 비교한다. `dropped_observations`는 기본 1,024건의 종료 관측 버퍼에서 밀려난 레코드 수다. 전체 요청 수와 지연은 클라이언트 원본으로 확인하고, 서버 취소·연결 종료·과부하 카운터를 함께 대조한다. [메트릭 계약](../api.md#get-metrics) · [30분 부하 원본 대조](05-mixed-soak-result.md#종료-관측-버퍼)

동일한 31초 주기의 cache-off 부하 두 건에서 인접 매핑 읽기 경과 시간과 지연·GPU 스텝·할당 지표를 대조했다. 순서와 반환 이후 표본 시점의 영향을 분리할 반복 및 프로세스 CPU 시간은 추가 측정 범위다. [관측 비용 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/result/observer-cost.json)

파생 이미지 한 프로세스에서 90초 혼합 부하 네 구간을 실행해 `/metrics`의 초당 1회 추가 조회를 비교했다. 두 실행 순서에서 프로세스 CPU 시간 비율은 +0.01%, +0.02%, 후반 TTFT p95는 +0.35%, +1.31%였다. 같은 부하로 워밍업한 뒤 실행 순서를 뒤집자 큰 RSS 증가가 추가 조회 없는 워밍업에 나타났고, 이후 기준과 추가 조회의 RSS 차이는 −280KiB와 +260KiB로 방향이 엇갈렸다. [첫 교차 대조](05-mixed-soak-result.md#초당-1회-메트릭-조회의-비용) · [워밍업 재시험](05-mixed-soak-result.md#워밍업-후-역순-교차-대조)

## 5. 실제 트래픽 재현과 장기 안정성

### W5-B3. 재현용 workload 구성

사용 가능한 서비스 자료에서 입력·출력 토큰 길이, 도착 간격, 동시성, schema/tool/logprobs 비율, prefix 재사용, 취소·timeout을 얻는다. 본문이 필요하지 않은 성능 실험은 통계와 합성 입력으로 재현한다. 원문 사용·저장이 필요한 품질 검증은 별도 취급하고 공개 저장소에 원문을 넣지 않는다.

trace가 없으면 "production 검증" 대신 "관측 분포를 가정한 합성 재현"으로 표시한다. 부하 생성기 CPU·소켓 제한과 예정 도착 지연도 측정해 서버 병목과 구분한다.

2026-11-20에 저장소의 운영 문서와 Riley GitHub 이슈를 확인했다. 같은 날 GitHub API의 공개 조회에서 배포 기록과 환경 목록은 각각 0건이었다. 현재 연결된 운영 근거는 A100 격리 실행과 합성 요청 집계다. 실제 서비스의 요청 분포와 승인된 SLO는 근거 출처 확인부터 남아 있다. 기존 합성 부하와 격리 복구 결과는 릴리스 검토의 재현 근거로 사용한다. [GitHub 배포 목록](https://api.github.com/repos/hkjeon13/riley/deployments) · [GitHub 환경 목록](https://api.github.com/repos/hkjeon13/riley/environments) · [격리 부하와 복구 결과](05-mixed-soak-result.md) · [릴리스 판정](06-candidate-qualification.md)

### W5-B4. soak를 위험 경계별로 확대

기존 30분 혼합 soak를 먼저 같은 최종 후보로 재현한다. 이후 1시간 탐색→문제 없으면 6시간 릴리스 장기 soak를 신규 제안으로 검토한다. 실행 시간 확대는 기존 30분 통과 기록을 취소하거나 자동으로 새 필수 정책을 만드는 것이 아니다. 자원·SLO와 함께 사전 확정한다.

시나리오는 cache off/on·cold/warm·eviction, 일반/긴 prompt 혼합, decode 긴 출력, schema/tool/logprobs, 느린 소비자, 임의 disconnect, 큐 포화 후 회복, KV 압력, drain/shutdown을 포함한다. device failure는 CPU 시뮬레이션·전용 하드웨어 실험 결과를 구분한다.

기존 도구의 실행 형태는 아래와 같다. 변수는 **이 실험에서 띄운 전용 서버**로 지정하고 사용한다. 서버 PID를 읽어 RSS를 측정하므로 같은 호스트에서 실행한다.

```bash
python3 benchmarks/live/api_check.py "$TEST_ADDRESS" "$MODEL_ID" \
  --tools --out "$API_RESULT_PATH"
python3 benchmarks/live/soak.py "$TEST_ADDRESS" "$MODEL_ID" \
  --seconds 1800 --rate 8 --pid "$TEST_SERVER_PID" --shared-prefix \
  --require-zero-rss-growth --out "$SOAK_RESULT_PATH"
```

전용 서버에서 일반·스키마·logprobs·도구·장문·긴 출력·연결 종료 요청을 합성한다. 일반 완료에는 finish, 토큰 usage, `[DONE]`이 모두 필요하다. 클라이언트가 콘텐츠 네 조각을 읽은 뒤 연결을 끊는 요청은 중간 EOF와 구분해 집계하고 서버의 disconnect·cancel 카운터를 대조한다. 429, 비정상 HTTP·SSE, drain 이후 요청·활성 KV 잔류 또는 native 할당 변화는 실패다. RSS는 경과 구간별로 기록한다.

30분 비교는 cache off/on 양쪽에 같은 합성 workload와 seed를 사용한다. `--shared-prefix`를 켜면 장문 요청이 여덟 개의 고정 프롬프트 앞부분을 재사용한다. 실행 시간의 20–50% 구간과 70–100% 구간을 예약된 도착 시각으로 묶어 TTFT·전체 완료 지연 p95를 비교하며, 후반 증가가 5%를 넘거나 관측치가 비면 실패다. TTFT 표본 수는 첫 콘텐츠 또는 도구 조각이 관측된 요청 수다. 짧은 `--smoke`는 지연 판정을 빼고 종료·계약·정리 경계를 확인한다.

최종 후보의 [메모리 증가 0 조건](../goal.md#33-운영-품질)은 `--require-zero-rss-growth`로 적용한다. Linux에서 후반 첫 표본과 drain의 `/proc/PID/smaps_rollup` RSS를 비교한다. 증가량이 양수이거나 두 정밀 표본 중 하나를 읽을 수 없으면 종료 코드 1을 반환하며, 읽기 실패 상세는 해당 표본에 남긴다. `rss_growth_second_half_mib`에는 기존 `/proc/PID/status` 진단 값을 계속 기록한다. 일반 RSS가 평평하게 보이는 경우에도 정밀 판정 결과가 적용된다. [혼합 부하 검사기](../../benchmarks/live/soak.py) · [A100 정밀 대조](05-mixed-soak-result.md#정밀-rss-재실행-결과)

캐시를 켠 전용 서버에서 [운영 검사기](../../benchmarks/live/operational_check.py)로 요청의 재사용 토큰, 공통 prefix 보존, 용량 압력 뒤 A 전용 블록 퇴거, 포화의 HTTP 429와 추론 회복을 확인했다. 검사는 활성 2개·대기 1개·캐시 128블록으로 구성하고 loopback에서 수행했다. [A100 운영 결과](05-mixed-soak-result.md#캐시-재사용퇴거와-큐-포화-후-회복)

느린 스트리밍은 loopback 클라이언트의 읽기를 멈추고 소켓의 실제 송신 큐가 수신 버퍼보다 큰 상태가 이어지는지 확인한다. A100 GPU 6의 실제 Qwen2.5-7B 실행에서 느린 읽기 도중 정상 요청 완료, 읽기 재개 뒤 SSE 종료와 토큰 usage, 활성 스트림 연결 종료 뒤 취소·disconnect와 자원 회수를 확인했다. [느린 스트리밍 검사기](../../benchmarks/live/slow_stream_check.py) · [A100 실행 순서](../../benchmarks/evidence/20261119-w5-slow-stream/campaign.sh) · [검증 결과](05-mixed-soak-result.md#느린-스트리밍-소비자와-연결-취소)

현재 공개 `/metrics`의 KV 사용량은 요청이 쓰는 블록을 보여 준다. 요청별 재사용 토큰은 응답 `usage`에서 확인했다. 스케줄러 내부의 전체 prefix hit 토큰 및 eviction 블록 카운터를 운영 응답에서 조회하려면 별도 계측 계약이 필요하다. [스케줄러 메트릭](../../crates/riley-scheduler/src/metrics.rs)

[고정 RC 릴리스 soak 드라이버와 검사기](../../benchmarks/soak/README.md)는 SmolLM2 workload·정식 golden·이미지 계보와 10개 bucket 슬롯을 요구한다. 현행 서버 응답의 24개 슬롯을 검사하려면 현행 바이너리로 별도 계약을 검증해야 한다.

### W5-B5. 실패 분류와 회수

- 의도적 overload의 429, 클라이언트 취소, 내부 5xx, malformed output을 별도 집계한다. 기대한 429만 숨기고 완료율을 높여 보이지 않는다.
- memory는 warm-up·cache 충전·allocator 보유를 구분하고 RSS·device·pinned·KV의 후반부 기울기와 drain 뒤 잔류를 본다. 일반 active KV 0과 cached block 보유를 혼동하지 않는다.
- 기존 목표의 오류 0·지연 드리프트 5% 이내를 적용하되, 부하별 기대 오류와 측정 구간을 사전에 고정한다. 장기 soak의 허용 메모리 기울기·SLO는 별도 사전 정의한다.
- 문제 발생 시 manifest·직전 요청 순서·메트릭·trace로 재현을 축소해 W1/W2/W3/W4에 돌려보낸다. 이전 승인 바이너리와 프로파일을 회수 대상으로 보존한다.

#### 합성 부하의 확인 기준

Qwen2.5-7B-Instruct의 A100 네이티브 실행에서 각 30분 동안 초당 평균 8건의 개방형 도착을 생성했다. 아래 설정 가중치는 `soak.py`에 정의돼 있고 건수는 캐시를 켜고 끈 두 원본 집계에서 확인했다. 장문은 800·1,500·3,000단어 중 하나를 선택하고 공통 prefix 8개를 사용했다. 실제 운영 트래픽의 길이·도착률·기능 비율은 확인 범위에 남아 있다. [부하 정의](../../benchmarks/live/soak.py) · [cache-on 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-on/soak.json.gz) · [cache-off 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off/soak.json.gz)

| 요청 종류 | 설정 가중치 | 모드별 관측 건수 |
|---|---:|---:|
| 일반 SSE | 55% | 정상 완료 7,984 |
| JSON 스키마 | 12% | 정상 완료 1,739 |
| logprobs | 8% | 정상 완료 1,160 |
| 필수 도구 호출 | 5% | 정상 완료 708 |
| 장문 | 10% | 정상 완료 1,415 |
| 최대 512토큰 생성 | 5% | 정상 완료 743 |
| 클라이언트 연결 종료 | 5% | 의도적 종료 700 |

혼합 부하에서 예상 밖 429·5xx·잘못된 SSE 종료는 실패로 기록한다. 의도적 연결 종료는 클라이언트와 서버 카운터를 대조한다. 별도 포화 시험의 429는 기대한 과부하 거절과 이후 추론 회복을 확인하는 근거다. TTFT·전체 완료 지연 p95의 허용 드리프트는 각각 5%이며, RSS 증가 허용량은 0이다. [운영 품질 목표](../goal.md#33-운영-품질) · [포화와 회복 시험](05-mixed-soak-result.md#캐시-재사용퇴거와-큐-포화-후-회복)

#### 후보 회수와 복구 확인

1. 검증 실패 시 해당 이미지 ID, Riley와 모델 manifest의 SHA-256, 실행 옵션과 API·soak 영수증을 보존한다. 기동 실패와 요청·지연·RSS·종료 자원 실패를 기록해 해당 후보의 릴리스 판단에 연결한다.
2. 운영 환경에서 이미 교체한 후보를 회수할 때는 해당 인스턴스의 신규 요청을 차단하고 활성·대기 요청과 요청 KV가 비워지는지 확인한다. 종료 뒤 서버의 native device·pinned 할당과 전용 GPU 예약 복구 여부를 해당 환경의 종료 기록으로 확인한다.
3. 해당 환경에 보존한 직전 승인 이미지 또는 바이너리와 모델 입력을 사용한다. 동일 환경에서 `/readyz`, 대표 모델과 예상 캐시 옵션, API 및 스트리밍 종료를 확인한 뒤 요청 라우팅을 복원한다. A100 격리 시험에서 파생 이미지를 종료하고 같은 loopback 주소의 기준 이미지에서 준비 상태·캐시 재사용·스트리밍 응답을 복구했다. 실제 운영 라우팅과 서비스 SLO는 검증 범위에 남아 있다. [준비 상태 계약](../api.md#get-healthz-get-readyz) · [격리 복구 시험](05-mixed-soak-result.md#a100-격리-이미지-복구-시험)

## 6. 완료 조건과 최종 산출물

- [x] W5-A: 점검한 사용자 문서의 버전·기본값·성능 주장을 소스나 해당 실행에 연결하고 과거 측정 시점을 표시했다. [주장별 확인 기록](#w5-a1-주장별-근거-목록-작성)
- [x] W5-B: A100에서 네이티브 모델 추론을 실제로 실행했고, 통과·초기 실패·미검증 범위를 기록했다. [운영 결과](05-mixed-soak-result.md)
- [x] 배포 이미지 smoke에서 Qwen2.5-3B와 Qwen3-4B의 API 각 40건, 정상 완료 각 338건, 의도적 연결 종료 각 21건과 GPU 자원 회수를 확인했다. [대표 모델 결과](05-mixed-soak-result.md#배포-이미지의-대표-모델-45초-smoke)
- [x] SmolLM2 MHA의 지원 범위에 맞춘 API 39건과 Qwen2.5-14B의 API 40건을 확인했다. 두 모델의 혼합 부하·취소·GPU 자원 회수도 통과했다. [MHA·14B 결과](05-mixed-soak-result.md#배포-이미지의-mha와-14b-smoke)
- [x] 배포 이미지에서 prefill 256행과 decode 1행을 함께 처리한 커밋된 GPU iteration 36개를 관측했다. 혼합 응답 6건은 단독 실행과 토큰 ID까지 일치했다. [혼합 iteration 결과](05-mixed-soak-result.md#배포-이미지의-커밋된-혼합-gpu-iteration)
- [x] 기준 A100 이미지와 파생 할당기 시험 이미지의 실행 근거를 하나의 매니페스트에 연결했다. 파생 시험의 correctness·API·performance·soak·패키징은 CI가 동일 이미지 및 설정인지 검사한다. [후보 매니페스트](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)
- [x] prefix cache·취소·shutdown을 합성 부하에서 검증하고 1Hz 추가 조회의 CPU·GPU·지연·RSS 비용을 교차 대조했다. 큰 RSS 증가는 두 실행 모두 첫 부하에서 나타났고, 워밍업 뒤 추가 조회와 기준의 변화량 차이는 −280KiB와 +260KiB였다. [운영 결과](05-mixed-soak-result.md#워밍업-후-역순-교차-대조)
- [x] 합성 부하의 분포와 확인 기준, 실패 분류 및 회수 절차를 기록했다. [합성 부하의 확인 기준](#합성-부하의-확인-기준) · [후보 회수와 복구 확인](#후보-회수와-복구-확인)
- [x] 같은 이미지 두 개의 닫힌 RAG C8을 3개 seed에서 반복해 3,072건의 출력을 대조했다. Seed 2의 지연 증가는 두 순서에서 재현됐고 seed 3은 두 순서에서 줄었다. [닫힌 RAG 확장 대조](05-mixed-soak-result.md#닫힌-rag-세-seed의-확장-교차-대조)
- [ ] 실제 운영 트래픽 분포와 서비스 SLO를 확인하고 해당 환경의 복구 절차를 검증한다.

최종 산출물은 문서 주장 목록, GPU 검증 matrix, 운영 workload 정의, 관측 사양, soak 결과, known limitation과 릴리스 판정서다. 대형 nsys 원본은 관리 가능한 저장소에 두고 해시·위치·보존 정책을 기록한다. 임시 로그나 접근 불가능한 경로만 남기지 않는다.

W1이 근거의 무결성을, W4가 출력·API 의미를, W2·W3가 실행 성능을 검증한다. 이 문서는 그 결과를 사용자에게 보이는 지원 범위와 실제 운영 판단으로 연결한다.

[후보 검증 근거와 릴리스 판정](06-candidate-qualification.md)에 A100 바이너리와 배포 이미지의 소스·실행 파일·검증 범위를 각각 연결했다.
