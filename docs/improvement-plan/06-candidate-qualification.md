# Riley 0.1.0 후보 검증 근거와 릴리스 판정

2026-11-19에 `main` `68788bee9`의 배포 입력을 확인했다. 현재 판정은 **릴리스 보류**다. A100에서 확인한 네이티브 실행 파일과 배포 이미지의 실행 파일은 각자의 SHA-256으로 추적한다. [후보별 원본 목록과 해시](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

## 실행 파일과 소스 연결

| 실행 대상 | 소스 리비전 | 확인한 범위 |
|---|---|---|
| A100 네이티브 빌드 | `f7c956d3c` | CUDA 12.4, SM80. API, 장문 정확성, Qwen2.5-14B 성능, Qwen2.5-7B 운영 검증 |
| Linux AMD64 배포 이미지 | `d3d8fe6e5` | CUDA 12.4, SM80·SM86·SM89. 이미지 빌드, 비루트 사용자, CLI, 동적 링크, 런타임 파일 무결성 |
| A100에서 재빌드한 배포 이미지 | `d3d8fe6e5` | 이미지 ID `0a907408ec73`. A100 6번의 CUDA 드라이버 주입, Qwen2.5-7B-Instruct API 40건, 장문 출력 대조 232건, GPU 자원 회수 |
| jemalloc을 포함한 파생 시험 이미지 | `d3d8fe6e5`의 실행 파일 | 이미지 ID `1fc444b1840e`. CPU 패키지 무결성, Qwen2.5-7B-Instruct API 40건, 두 캐시 모드 30분 운영 기준, 장문 출력 232건, 사전 성능 회귀 기준 통과. RAG C8 seed 2 꼬리 지연 증가 |

초기 감사에서 `f7c956d3c` 네이티브 빌드와 `96c04dbe4` 배포 이미지의 프로그램 입력 Git 객체를 비교했다. 그 뒤 헤더·Rust 바인딩·서버 런타임을 보완해 배포 이미지의 Riley 실행 파일 해시가 갱신됐다. 네이티브 시험은 각 실행 파일로 계속 추적한다. [초기 네이티브 빌드](04-structured-output-result.md) · [이미지 빌드 이력](01-validation-result.md#현행-배포-이미지)

현행 배포 검증 이미지 `d3d8fe6e5`의 이미지 ID·Riley 실행 파일 해시를 보존했다. 이 이미지와 첫 감사 시점 `68788bee9`, 후속 감사 시점 `838103b02`의 배포 입력 Git 객체 14개가 모두 일치한다. 동일 Rust·CUDA 프로그램 입력인 `77c367003`에서는 CUDA 12.8 네이티브 빌드·C ABI·Clippy도 통과했다. [CUDA 12.4 이미지 원본](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/receipt.json) · [CUDA 12.8 네이티브 CI](https://github.com/hkjeon13/riley/actions/runs/44410425140)

릴리스 검토에서는 아래 명령으로 원격 `main`의 배포 입력을 검증 이미지에 연결한다. 기록된 입력이 달라지면 변경된 경로를 출력하고 실패한다. 이미지·바이너리·모델의 실행 근거는 아래 표의 검증 시점과 장비에 연결된다. [검사 코드](../../ci/check_improvement_evidence.py)

```bash
python3 -B ci/check_improvement_evidence.py --distribution-revision origin/main
```

같은 명령은 `current_allocator_trial`의 일곱 검증 보고서를 대조한다. 파생 이미지의 패키징, 두 캐시 모드의 30분 부하, 장문 정확성, 성능 대조, 고정 도착 대조와 메트릭 조회 두 시험을 연결한다. 성능 시험은 실제 이미지·실행 인자·할당기 설정이 일치하고, 메트릭 조회 재시험은 첫 조회 시험의 검산 해시·체크포인트·요청 수에 맞아야 한다. 닫힌 부하의 RAG 꼬리 지연과 실제 운영 도착 분포·서비스 SLO 확인이 남아 있어 파생 이미지 릴리스를 보류한다. [파생 후보 매니페스트](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

## 현재 판정

| 조건 | 확인된 근거 | 남은 확인 |
|---|---|---|
| W1 Linux·과거 계약 | hosted Linux CPU·CUDA 검사와 고정 Python의 RC3 격리 테스트 34건 통과. A100에서 재빌드한 이미지의 NVIDIA 드라이버 주입과 실제 모델 API 40건 확인 | W5의 최종 운영 기준 충족 |
| W2·W3 성능 | 이벤트 발행·스케줄 네 조합과 두 14B 모델의 실측을 완료했다. W2의 정렬된 KV 테이블과 이벤트 복사 제거는 SmolLM2의 단일 iteration에서 채택 이득을 충족하지 못했다. W3의 1,024행 QKV 후보는 Coder-14B 출력 1,280건을 재현했지만 역순 동시 유입의 decode TPOT이 1.99% 늘어 1% 기준을 넘었다. 형상 공통 적용을 기각했다. | 기존 설정으로 기준을 유지하며 새 후보의 단독·결합 회귀와 영향 모델을 검증 |
| W4 장문·정확성 | 고정 입력 18개와 holdout 11개의 엔진 비교, 두 사례의 첫 BF16 차이와 28개 레이어의 residual·최종 norm 재계산, 네 모델의 동일 입력 GSM8K 품질 기준 통과. A100 재빌드 이미지의 장문 출력 232건 일치. 두 decode 사례에서 첫 레이어 QKV를 정확한 HF 투영값으로 교체해도 선택 토큰과 전체 생성 결과가 같았다. | 다른 위치와 레이어의 영향, 향후 수치 변경의 영향 모델·품질 대조 |
| W4 스키마 | 문서화한 부분집합에서 GPU 일반 API 40건과 schema 계약 11건을 통과했다. A100 재빌드 이미지의 대표 모델 API 40건도 통과했다. | W5의 최종 운영 기준 충족 |
| W5 운영 | 기준 배포 이미지의 익명 RSS는 cache-on +616KiB, cache-off +700KiB였다. jemalloc 파생 이미지에서 각각 −440KiB와 −640KiB로 두 설정의 30분 운영 기준을 통과했다. 장문 232건이 일치했다. 사전 성능 기준은 통과했고 RAG C8 seed 2 닫힌 부하의 TTFT p95는 +12.69%였다. 고정 시각 대조에서는 +1.08%와 −2.29%였다. | 닫힌 부하에서 관측한 비용과 실제 운영 도착 분포를 반영한 기본값 판정 |

기존 이벤트 발행 정책과 서빙 설정을 기준선으로 삼았다. 이전 빌드에서 관찰된 성능 실험은 입력·장비·바이너리에 연결하며, 새로운 수치 기준이나 기본값을 채택할 때는 종합 계획의 출력 불변성과 회귀 기준을 적용한다. [W2 결과](02-async-step-result.md) · [W3 결과](03-long-context-result.md) · [W4 정확성](04-long-accuracy-result.md) · [W5 운영 결과](05-mixed-soak-result.md)

Qwen2.5-14B-Instruct의 CUDA 진단 바이너리에서 prefill chunk와 KV 한도 네 조합의 장문 308건이 입력·생성 토큰 ID까지 일치했다. 90초 입장 제한에서 완료한 정확성 실험이다. KV 768블록·chunk 512의 기본 30초 실행에서는 C32의 HTTP 408을 확인했다. 새 기본값의 회귀 검증과 실제 서비스 분포 확인은 남아 있다. [W3·W4 경계 부하 대조](03-long-context-result.md#장문-chunk와-kv-한도의-교차-대조)

Qwen2.5-7B-Instruct의 두 decode 분기에서 28개 레이어의 residual 교체를 모두 대조했다. HF 접두부를 고정한 상태에서 Riley 선택을 재현하는 첫 경계가 사례별로 레이어 0의 MLP 이후와 attention 직후로 달랐다. 수치 변경 후보의 채택과 영향 모델의 재검증은 남아 있다. [W4 residual 경계 검증](04-long-accuracy-result.md#residual-경계-교체로-decode-토큰-선택을-재현)

동일 입력의 레이어 0 비교에서 사례 12는 down projection 출력 1,536개가 달랐고, `numbers-1023`은 attention 출력 투영의 세 값이 달랐다. 각 사례의 해당 출력 행을 사용한 재실행에서 Riley 토큰 선택을 재현했다. [연산별 대조](04-long-accuracy-result.md#같은-입력으로-레이어-0의-mlp와-attention-출력-투영-대조)

SmolLM2-1.7B의 같은 A100 바이너리에서 제출 후 발행의 32행 decode 커널 사이 공백 중앙값은 두 비교에서 51–72µs 줄었다. 네 일반 서빙 비교에서 처리량은 0.38–0.89% 늘었고 측정한 요청의 출력이 일치했다. `after-submit-v1`을 W2와 W3 후속 시험의 기준 정책으로 유지한다. [발행 정책별 계측](02-async-step-result.md#발행-정책과-32행-decode-커널-간격)

커밋된 토큰 텍스트와 logprobs의 복사를 줄인 후보는 SmolLM2의 일반 서빙 512건과 C1·C8 생성 ID·스트리밍 logprobs를 재현했다. 네 성능 비교의 처리량 변화는 −0.35%에서 +0.24%였다. 정렬된 KV 테이블과 이번 후보의 미채택을 확정하고, 다중 스텝 중첩의 성능·자원 수명은 추가 검증 범위에 둔다. [이벤트 복사 제거](02-async-step-result.md#커밋된-이벤트-데이터의-복사-제거)

W4의 A100 진단 빌드에서 보존된 C1 응답 8건을 재현하고 해당 BF16 점수 행을 검산했다. 첫 분기 7건 중 Riley·HF ID의 공동 최대 5건과 Riley 단독 최대 2건을 확인했다. 큰 ID를 고르는 동점 규칙을 계산하면 이미 일치한 대조 사례도 바뀌어 적용을 보류했다. [실제 점수 행 결과](04-long-accuracy-result.md#실제-bf16-점수-행의-분기-검산)

Riley 단독 최대 두 사례의 레이어 0에서 같은 BF16 입력과 가중치로 bias 전 QKV 차이를 재현했다. CUDA 12.4에서 기존 병합 투영의 알고리즘 ID를 21에서 31로 바꾸면 두 행의 결과가 PyTorch 병합 투영과 비트 단위로 일치했다. 각 사례에서 대체 계획의 결과와 HF의 정확한 값을 순전파에 넣어도 선택 토큰과 전체 생성 결과는 같았다. [동일한 피연산자의 투영 대조](04-long-accuracy-result.md#동일한-bf16-피연산자의-투영-대조) · [순전파 대조](04-long-accuracy-result.md#대체-qkv-행이-최종-decode에-미친-영향)

Riley ID가 단독 최대였던 두 decode 사례에서는 같은 RMSNorm 계산식과 출력 가중치로 두 ID의 BF16 점수를 재현했다. 연산별 추적에서 두 사례 모두 입력 norm이 일치했고 최초 차이는 레이어 0의 bias 전 QKV 투영이었다. 달랐던 9개 내적의 FP64 결과를 BF16으로 반올림하면 8개가 HF 재계산 값, 1개가 Riley 관측값과 일치했다. Riley는 병합 QKV에 cuBLASLt 알고리즘 21과 Split-K 1을 사용했고, HF 재계산은 Q에 GEMM, K·V에 GEMV 커널을 사용했다. 개별 내적의 누산 순서와 정밀도는 후속 대조 범위에 남아 있다. [두 decode 사례 결과](04-long-accuracy-result.md#실제-bf16-점수-행의-분기-검산)

W3의 QKV 추적 416건은 해당 비계측 요청·응답과 일치했다. 50ms 간격 부하의 동일 장문 요청에서 늘어난 첫 토큰 지연 528–541ms 중 381–392ms는 prefill 조각 사이 대기였다. RAG 첫 토큰 p95 개선이 일관되게 재현될 때까지 후보는 격리 시험에만 둔다. [두 순서의 단계별 결과](03-long-context-result.md#50ms-간격-유입의-추가-지연-위치)

후속 480건에서 45번 요청의 도착 시각을 ±12ms 조정하면 기준과 후보에서 prefill 순서가 일치했고 후보의 TTFT p95는 2.44–2.59% 줄었다. 기존 도착 시각에서는 두 실행 순서 모두 prefill 순서가 갈렸고 p95는 1.77–2.15% 늘었다. 실제 도착 시각 대조는 한 seed에 해당하며 기본값 적용은 보류다. [실제 요청의 도착 시각 대조](03-long-context-result.md#실제-요청의-도착-시각-대조)

첫 우선순위에 명령 송신 시각을 반영한 CPU 재생은 보정 폭 76.879µs에서 이미 두 번째 스텝을 바꿨다. 그보다 작은 보정 폭은 기존 도착 시각의 순서 차이를 해소하지 못해 후보에서 제외했다. [보정 폭 대조](03-long-context-result.md#명령-송신-시각을-사용한-우선순위-재생)

Qwen2.5-14B에서 두 RAG seed와 기준·후보의 상대 도착 시각표 네 개를 교차 재생했다. 측정 1,024건의 입력·응답 지문과 종료 사유가 일치했고 두 실행 순서에서 TTFT p95는 3.54–12.88%, TPOT p50은 3.85–9.35% 감소했다. 실제 송신 시각 차이 p95의 최댓값은 1.51ms였다. 같은 바이너리의 Coder-14B 대조에서 decode 회귀가 발생해 이 후보의 형상 공통 적용을 기각했다. [두 RAG seed의 고정 도착 재생](03-long-context-result.md#두-rag-seed의-고정-도착-재생) · [Coder-14B 결과](03-long-context-result.md#coder-14b의-1024행-후보-검증)

같은 실행에서 요청 단계와 CUDA 커널을 연결한 결과 50ms 유입의 41번 요청은 prefill 조각 사이 대기가 399ms 늘었다. 본인 요청이 대기하는 동안 GPU는 387ms 더 실행됐고 GEMM·attention이 그중 대부분을 차지했다. 시험의 KV 예약 상한은 1,832블록으로 설정 한도 2,576블록보다 낮았다. [요청–GPU 결합 계측과 KV 예약 상한](03-long-context-result.md#같은-실행의-요청-대기와-cuda-커널-대조)

같은 두 부하에서 KV를 768블록으로 제한하면 68건에 실제 예약 부족이 기록됐다. 비교한 출력 104건은 일치했고 종료 전 할당 KV는 0이었다. 운영 후보의 도착 분포를 구할 때는 입력·출력 길이와 동시성에서 KV 예약량을 함께 산출해 입장 전 대기와 입장 후 스텝 대기를 구분한다. [KV 압력 대조](03-long-context-result.md#실제-kv-예약-압력과-회수)

W5의 30분 실행에서 종료 뒤 GPU·pinned 메모리와 활성 KV는 0이었다. Cache-off 정밀 실행에서 일반 RSS의 변화는 0이었고 `smaps_rollup`은 익명 RSS +700KiB를 기록했다. 따라서 RSS 증가 0 조건은 미충족이다. 정밀 판정기를 적용한 다음 실행에서 +812KiB를 확인했고 매핑별 합계에서 전부 익명 private에 속했다. 요청·지연·종료 회수 검사는 통과했고 RSS 검사만 실패했다. GPU 메모리 오류 주입은 A100 6번의 CUDA 12.4 SM80 테스트 바이너리에서 네 독립 사례가 통과했다. [운영 품질 기준](../goal.md#33-운영-품질) · [정밀 RSS 결과](05-mixed-soak-result.md#정밀-rss-재실행-결과) · [오류 주입 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/verification.json)

같은 A100 바이너리의 후속 대조에서도 RSS가 기본값에서 +932KiB, `MALLOC_ARENA_MAX=2`에서 +3,648KiB 증가했다. 두 조건 모두 RSS 기준만 실패했고 arena 제한은 채택하지 않았다. [비교 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/result/paired-comparison.json)

초기 자동 승인 검토 뒤 `96c04dbe4`와 `d3d8fe6e5` 아카이브의 전송을 각각 승인받았다. `aws-a100`의 전용 경로에서 두 아카이브의 SHA-256이 원래 CI에 기록된 값과 일치함을 확인했다. GPU 7의 RSS 비교 이후 `d3d8fe6e5`를 빌드했다. AWS 이미지 ID는 `0a907408ec73`, GitHub CI 이미지 ID는 `8cb92e573ef1`이다. Riley 바이너리를 포함한 패키지 런타임 파일 21개의 SHA-256은 모두 같았다. AWS 이미지에서 GPU 6의 체크포인트 무결성, API 40건, 정상 종료와 20GiB 예약 복귀를 확인했다. [A100 스테이징 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-staged-sources.json) · [A100 패키지·GPU 증거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json) · [종합 완료 조건](README.md#8-종합-완료-체크리스트)

같은 이미지에서 장문 29개에 대한 232건을 확인했다. 초기 GPU 확인의 이미지 ID, CPU 보고서, Riley 바이너리 및 체크포인트 해시와 일치했다. Cache-on과 cache-off의 30분 혼합 부하는 각각 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. 두 서버 모두 정상 종료하고 자원을 회수했다. 후반 익명 RSS는 cache-on에서 616KiB, cache-off에서 700KiB 증가해 두 설정 모두 RSS 증가 0 기준을 충족하지 못했다. [장문 대조](04-long-accuracy-result.md) · [배포 이미지의 혼합 부하](05-mixed-soak-result.md#배포-이미지의-혼합-부하)

A100 격리 복구 시험에서 파생 이미지 `1fc444b1840e`를 종료하고 같은 loopback 주소에 기준 이미지 `0a907408ec73`를 기동했다. 136.37초 뒤 기준 이미지의 준비 상태를 확인했다. 두 이미지가 API 40건씩 통과했고 스트리밍 출력과 캐시 재사용 112토큰이 일치했다. 종료 시 device·pinned 할당 및 KV는 0이었다. 복구 시간은 이번 A100 loopback 실행의 측정값이다. [복구 시험](05-mixed-soak-result.md#a100-격리-이미지-복구-시험)

원본 보존 경로에서 패키지 영수증과 GPU API, 장문 비교, 두 혼합 부하의 체크섬을 다시 검증했다. 두 부하에서 후반 첫 정밀 표본과 drain 표본을 직접 빼서 RSS를 확인했다. 같은 바이너리·체크포인트·홀더를 사용했고 서버 인자의 차이는 `--prefix-cache-blocks 128`이었다. 최초 cache-off의 기동 실패도 기록에 남겼다. [통합 판정과 원본 해시](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/verified-a100-candidate.json)

```bash
python3.12 -B benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/verify_a100_candidate.py \
  "$A100_EVIDENCE_ROOT"
```

`A100_EVIDENCE_ROOT`에는 검증한 실행 디렉터리의 공통 상위 경로를 지정한다. 검사기는 배포 이미지의 패키지·API·장문·혼합 부하를 판정한다. 종료 코드 0은 통과, 1은 검증된 원본의 기준 실패, 2는 원본 누락 또는 불일치다. 해당 이미지의 재검산은 코드 1을 반환했다. [재현 검사기](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/verify_a100_candidate.py)

주기적으로 glibc가 메모리를 반환하도록 한 진단군의 cache-off 30분 실행은 익명 RSS가 60KiB 증가해 증가 0 조건을 충족하지 못했다. 같은 이미지의 비계측 기준은 700KiB 증가했다. Cache-on에서는 기준 +616KiB에서 주기적 반환 적용 시 −108KiB로 바뀌어 모든 검사를 통과했다. 두 설정 모두 API 40건과 지연·종료 검사를 통과했다. Cache-off가 남아 릴리스 판정은 보류다. [주기적 반환 결과](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

31초 주기의 별도 cache-off 표본 진단에서도 익명 RSS가 500KiB 증가했다. 중간과 drain의 반환 이후 경과 시간을 확인했고, 증가 0 기준은 계속 미충족이다. [표본 시각과 매핑 대조](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

7초 주기의 첫 시도는 라이브러리 적재 실패로 후보 판정에서 제외했다. 같은 이미지에서 서비스 계정의 적재를 먼저 검증한 재실행에서 cache-off 익명 RSS는 44KiB, cache-on은 292KiB 증가했고 두 모드 모두 기존 RSS 증가 0 항목만 실패했다. Drain 표본은 각각 직전 반환 완료 후 0.19–0.59초와 4.70–5.05초에 읽혔다. 릴리스 보류 상태이며 동일한 7초 주기에 스레드별 할당 캐시 비활성화 대조를 등록했다. [재실행 결과](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

7초 주기 반환과 스레드별 할당 캐시 비활성화를 함께 적용한 동일 이미지에서 `cache-off` 정밀 익명 RSS는 144KiB 감소했고 `cache-on`은 88KiB 증가했다. 두 시험 모두 API 40건과 요청·지연·자원 회수 항목을 통과했다. 증가 0 기준을 `cache-on`에서 충족하지 못해 릴리스 판정은 보류다. [두 설정의 검산과 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/paired-verified.json)

같은 배포 이미지에서 스레드별 할당 캐시 비활성화만 적용한 두 30분 실행은 `cache-on`에서 200KiB, `cache-off`에서 148KiB의 익명 RSS 증가를 기록했다. 각 실행의 API 40건과 요청·지연·종료 검사는 통과했다. 별도 HTTP 프레이밍 이미지 `e019db74d2f0`은 패키지 검사를 통과했다. [단독 대조 결과](05-mixed-soak-result.md#스레드별-할당-캐시-설정-단독-대조) · [프레이밍 이미지 검증](05-mixed-soak-result.md#http-요청-버퍼-용량의-cpu-대조)

HTTP 프레이밍 이미지의 30분 실행에서 정밀 RSS는 `cache-on`에서 632KiB, `cache-off`에서 580KiB 증가했다. 기준 이미지의 증가량은 각각 616KiB와 700KiB였다. 두 실행은 각각 API 40건과 지연·종료 자원 검사를 통과했고 같은 20GiB 예약으로 복귀했다. RSS 항목에서 두 모드가 모두 실패해 후보를 채택하지 않았고, 조건부 장문 정확성 대조를 건너뛰었다. 릴리스는 두 모드의 RSS 기준을 충족할 때까지 보류한다. [두 부하와 정확성 판정 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/paired-verified.json)

기본 이미지의 추가 정밀 계측에서 `cache-off`의 15분 이후 RSS는 480KiB 증가했다. 29분에 460KiB가 이미 관측됐고, drain 뒤 60초 표본까지 같은 RSS가 유지됐다. RSS 항목만 실패해 채팅 SSE 후보의 조건부 검증 단계로 넘어갔다. [시점별 계측과 종료 검산](05-mixed-soak-result.md#rss-표본과-실행-중-자원-지표-대조) · [채팅 SSE 후보](05-mixed-soak-result.md#채팅-스트림의-임시-json-할당)

채팅 SSE 후보 `06eed36dcc60`의 두 30분 부하에서 정밀 RSS는 `cache-on`에서 508KiB, `cache-off`에서 608KiB 증가했다. 기준 이미지의 616KiB와 700KiB보다 각각 108KiB와 92KiB 적었다. 두 모드에서 API 40건과 요청·지연·종료 자원 검사를 통과했고 RSS 증가 0 항목만 실패했다. 조건부 장문 비교를 건너뛰었으며 릴리스는 보류다. [두 실행의 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/paired-verified.json) · [장문 검증 판정](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/accuracy-follow-up/accuracy-skip-verified.json)

기준 `d3d8fe6e5` 이미지에 jemalloc을 연결한 별도 대조에서 정밀 RSS는 `cache-on` −680KiB, `cache-off` −672KiB로 두 모드의 운영 기준을 통과했다. 각 모드의 API 40건과 요청·지연·자원 회수에 이어 같은 설정의 장문 정확성 232건도 일치했다. 별도 시험 이미지 `1fc444b1840e`의 CPU 검증을 마쳤고, 이 이미지 ID의 GPU 검증을 진행 중이다. [할당기 시험과 운영 기준](05-mixed-soak-result.md#즉시-반환-설정의-할당기-대조)

파생 이미지 `1fc444b1840e`에서 `cache-on`과 `cache-off`의 정밀 RSS는 각각 440KiB와 640KiB 감소했다. 각 모드에서 API 40건과 요청·지연·종료 자원 검사를 모두 통과했고, 같은 20GiB 예약이 복구됐다. 같은 이미지에서 기준 144건과 홀드아웃 88건의 출력이 일치했다. [두 30분 부하 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/verified-soaks.json) · [장문 정확성 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/accuracy-verified.json)

두 부하와 장문 검증을 통과한 파생 이미지를 기준 이미지와 교대로 두 번씩 실행했다. 여섯 부하에서 총 2,560건의 요청을 오류 없이 처리했고 대응하는 출력이 모두 일치했다. 짧은 요청·decode 회귀 기준은 통과했다. RAG C8 seed 2의 TTFT p95는 두 실행 순서에서 7.43%와 18.19% 증가했다. Qwen2.5-7B-Instruct, A100 6번, 고정 부하에 해당하는 결과다. 이 RAG 비용을 확인해 파생 설정의 기본값 적용과 릴리스를 보류한다. [성능 대조와 지연 위치](05-mixed-soak-result.md#같은-gpu에서-비교한-서빙-성능)

같은 RAG 요청의 두 시각표를 네 실행에서 재생했다. 측정 512건은 기존 출력과 일치했다. 각 시각표에서 기준 대비 TTFT p95는 +1.08%와 −2.29%였고, 시각표 자체를 바꾸면 두 이미지의 TTFT p95가 각각 31.00%, 26.63% 증가했다. 이 검증은 관측된 두 시각표에서 지연 차이를 분리한 증거다. 릴리스 판정에는 기존 닫힌 부하 결과와 실제 운영 도착 분포도 필요하다. [RAG 고정 시각 재생](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-replay/verified-replay.json)
