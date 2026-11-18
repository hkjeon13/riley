# W5. 문서 정합성·관측·운영 검증 개선

작성일: 2026-11-13\
종합 계획: [진행 단계와 릴리스 조건](README.md)\
구성: W5-A는 즉시 가능한 문서 정리, W5-B는 GPU 회귀·운영 검증이다.

## 1. 현재 문제와 위험 범위

| 항목 | 확인 내용 | 개선 이유 |
|---|---|---|
| 문서의 측정 시점 차이 | README의 작은 모델 TPOT·32B 상태가 최신 목표·서빙 검증과 다름 | 사용자가 과거 셀과 현재 결과를 혼동할 수 있음 |
| 기본값 설명 드리프트 | serving-validation에 과거 KV chunk 1024 기본값 설명, 최신 configuration은 512 및 일부 생성 256 | 문서만 보고 승인 프로파일을 재현하기 어려움 |
| API 비용 설명 | api.md에 logprobs가 항상 CPU logits 비용을 낸다는 과거 설명, engine에는 GPU summary 경로 존재 | 사용 경로·기능 조합별 실제 비용을 구분해야 함 |
| 모델 지원 의미 혼합 | model-support는 owned HF 프로파일·특정 pinned 모델 검증, README는 일반 native qualification | 모델 지원·HF exact·속도 qualification을 분리해야 함 |
| GPU CI 범위 | native-cuda workflow는 nightly compile, GPU runtime/memory smoke는 수동 opt-in | 모델 추론·routing·성능 회귀가 PR마다 자동 검증되지 않음 |
| 운영 근거 | 합성 트래픽 30분 soak 기록. 실제 production trace 미검증 | 실제 길이·도착·취소·기능 혼합의 차이를 모름 |
| prefix cache | 문서상 캐시 30분 soak와 실제 device failure 검증이 미완료 | 일반 soak 성공을 cache-on 수명 보장으로 확대할 수 없음 |

근거: [README](../../README.md), [configuration](../configuration.md), [API](../api.md), [serving-validation](../serving-validation.md), [model-support](../model-support.md), [prefix-cache](../prefix-cache.md), [GPU CI](../../.github/workflows/native-cuda.yml).

## 2. W5-A: 문서를 현재 후보에 맞추는 방법

### W5-A1. 주장별 근거 목록 작성

성능 숫자·기본값·지원 모델·API 보장마다 다음을 연결한다.

`주장 → 대상 버전/프로파일 → 실제 소스 또는 실험 manifest → 측정 시점 → 지원 범위 → 제한`

단일 최신 숫자를 모든 표에 복사하지 않는다. 예를 들어 과거 qualification은 fixed/natural 8셀이고 최신 TP=2 서빙 검증은 64-token 셀이다. 새 결과가 좋다고 기존 8셀 qualification을 충족했다고 선언하지 않는다. README의 "32B 미충족"은 어떤 매트릭스의 판정인지 남기고 최신 별도 결과를 함께 설명한다.

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

관측 on/off의 CPU·RSS·GPU·지연 비용을 비교한다. dropped observation이 생기면 결과에 표시하고 누락을 0ms나 0건으로 해석하지 않는다. Prometheus exporter가 필요하다면 별도 adapter 여부를 검토하며 현재 JSON endpoint를 임의 변경하지 않는다.

## 5. 실제 트래픽 재현과 장기 안정성

### W5-B3. 재현용 workload 구성

사용 가능한 서비스 자료에서 입력·출력 토큰 길이, 도착 간격, 동시성, schema/tool/logprobs 비율, prefix 재사용, 취소·timeout을 얻는다. 본문이 필요하지 않은 성능 실험은 통계와 합성 입력으로 재현한다. 원문 사용·저장이 필요한 품질 검증은 별도 취급하고 공개 저장소에 원문을 넣지 않는다.

trace가 없으면 "production 검증" 대신 "관측 분포를 가정한 합성 재현"으로 표시한다. 부하 생성기 CPU·소켓 제한과 예정 도착 지연도 측정해 서버 병목과 구분한다.

### W5-B4. soak를 위험 경계별로 확대

기존 30분 혼합 soak를 먼저 같은 최종 후보로 재현한다. 이후 1시간 탐색→문제 없으면 6시간 릴리스 장기 soak를 신규 제안으로 검토한다. 실행 시간 확대는 기존 30분 통과 기록을 취소하거나 자동으로 새 필수 정책을 만드는 것이 아니다. 자원·SLO와 함께 사전 확정한다.

시나리오는 cache off/on·cold/warm·eviction, 일반/긴 prompt 혼합, decode 긴 출력, schema/tool/logprobs, 느린 소비자, 임의 disconnect, 큐 포화 후 회복, KV 압력, drain/shutdown을 포함한다. device failure는 CPU 시뮬레이션·전용 하드웨어 실험 결과를 구분한다.

기존 도구의 실행 형태는 아래와 같다. 변수는 **이 실험에서 띄운 전용 서버**로 지정하고 사용한다. 서버 PID를 읽어 RSS를 측정하므로 같은 호스트에서 실행한다.

```bash
python3 benchmarks/live/api_check.py "$TEST_ADDRESS" "$MODEL_ID" \
  --tools --out "$API_RESULT_PATH"
python3 benchmarks/live/soak.py "$TEST_ADDRESS" "$MODEL_ID" \
  --seconds 1800 --rate 8 --pid "$TEST_SERVER_PID" --out "$SOAK_RESULT_PATH"
```

현재 도구의 인증·동시 서버·endpoint 가정은 실행 전 확인한다. 기존 soak가 모든 신규 시나리오를 생성한다고 가정하지 말고 빠진 cache·drain·quota 시나리오는 별도 추가한다.

### W5-B5. 실패 분류와 회수

- 의도적 overload의 429, 클라이언트 취소, 내부 5xx, malformed output을 별도 집계한다. 기대한 429만 숨기고 완료율을 높여 보이지 않는다.
- memory는 warm-up·cache 충전·allocator 보유를 구분하고 RSS·device·pinned·KV의 후반부 기울기와 drain 뒤 잔류를 본다. 일반 active KV 0과 cached block 보유를 혼동하지 않는다.
- 기존 목표의 오류 0·지연 드리프트 5% 이내를 적용하되, 부하별 기대 오류와 측정 구간을 사전에 고정한다. 장기 soak의 허용 메모리 기울기·SLO는 별도 사전 정의한다.
- 문제 발생 시 manifest·직전 요청 순서·메트릭·trace로 재현을 축소해 W1/W2/W3/W4에 돌려보낸다. 이전 승인 바이너리와 프로파일을 회수 대상으로 보존한다.

## 6. 완료 조건과 최종 산출물

- [ ] W5-A: 사용자 문서의 버전·기본값·성능 주장이 소스·동일 설정 근거와 연결되고 과거 측정은 시점이 표시된다.
- [ ] W5-B: 모델 추론 GPU lane을 실제 실행한 성공·실패·미실행 근거가 있다.
- [ ] 후보별 correctness·API·performance·soak·패키징 결과가 같은 manifest로 연결된다.
- [ ] prefix cache·취소·shutdown·관측 비용을 검증하고 미검증 범위를 명시한다.
- [ ] 운영 재현 분포·SLO·실패 분류·회수 절차가 기록된다.

최종 산출물은 문서 주장 목록, GPU 검증 matrix, 운영 workload 정의, 관측 사양, soak 결과, known limitation과 릴리스 판정서다. 대형 nsys 원본은 관리 가능한 저장소에 두고 해시·위치·보존 정책을 기록한다. 임시 로그나 접근 불가능한 경로만 남기지 않는다.

W1이 근거의 무결성을, W4가 출력·API 의미를, W2·W3가 실행 성능을 검증한다. 이 문서는 그 결과를 사용자에게 보이는 지원 범위와 실제 운영 판단으로 연결한다.
