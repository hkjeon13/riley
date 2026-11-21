# W5-B A100 혼합 부하와 종료 검증

2026-11-19 KST에 `aws-a100`의 물리 GPU 7에서 Qwen2.5-7B-Instruct와 `native-bf16-varlen-v1`을 실행했다. 모델의 문맥 상한은 8,192, 출력 상한은 512토큰이다. CUDA 바이너리는 W4의 `f7c956d3c` 빌드와 SHA-256이 같다. [W5 계획](05-documentation-and-operations.md) · [W4 CUDA 빌드](04-structured-output-result.md) · [실행 인자와 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/receipt.json)

## 배포 이미지의 대표 모델 45초 smoke

2026-11-20 KST에 A100 7번에서 배포 이미지 `0a907408ec73`의 `--serving-smoke`를 실행했다. 2-KV-head GQA의 Qwen2.5-3B-Instruct와 Q·K 정규화를 쓰는 Qwen3-4B의 체크포인트 SHA-256을 고정했다. 각 컨테이너에서 같은 이미지의 CPU 보고서와 실행 중 바이너리 해시를 대조하고 API 검사 뒤 초당 평균 8건으로 45초 동안 혼합 요청을 보냈다. [실행 순서](../../benchmarks/evidence/20261120-w5-serving-smoke/campaign.sh) · [판정과 원본 해시](../../benchmarks/evidence/20261120-w5-serving-smoke/verified-smoke.json)

| 항목 | Qwen2.5-3B-Instruct | Qwen3-4B |
|---|---:|---:|
| 일반·동시성·스키마·도구·logprobs API | 40/40 | 40/40 |
| 혼합 부하 정상 완료 | 338 | 338 |
| 스키마의 정상 종료 / 필수 도구 종료 | 42 / 23 | 42 / 23 |
| 의도적 연결 종료 / 서버 취소 증가 | 21 / 21 | 21 / 21 |
| 예상 밖 HTTP·SSE 오류 / 429 | 0 / 0 | 0 / 0 |

두 실행은 drain 뒤 활성·대기 요청과 요청 KV가 0이었고 종료 코드 0과 device·pinned 할당 0을 기록했다. GPU 6·7은 기존 홀더의 20GiB 예약을 확인했다. 배치 정확성의 이번 범위는 API의 동시 요청에서 확인한 텍스트와 스키마 응답이다. 같은 배포 이미지의 혼합 iteration은 아래에서 별도 요청 두 건으로 대조했다. [검사기](../../benchmarks/evidence/20261120-w5-serving-smoke/verify_smoke.py)

초기 시작 시도에서는 Python 의존 모듈 누락으로 컨테이너 생성 전에 종료했다. 의존 모듈 보완 뒤 두 원본 모델 디렉터리의 권한이 `0700`, 파일은 `0600`이라 비루트 이미지 사용자의 기동이 실패했다. 검증 전용 경로에 manifest와 payload 사본을 만들고 비루트 계정의 읽기를 확인한 뒤 실제 실행했다. 체크포인트 해시는 같았고 공유 모델의 권한을 보존했다. [체크포인트 준비 코드](../../benchmarks/evidence/20261120-w5-serving-smoke/prepare_model.py) · [집계 재현](../../benchmarks/evidence/20261120-w5-serving-smoke/fetch-summary.sh)

### 배포 이미지의 MHA와 14B smoke

같은 이미지 `0a907408ec73`를 A100 7번에서 실행했다. SmolLM2-1.7B-Instruct는 attention head와 KV head가 모두 32개인 MHA 모델이다. 이 모델에는 기본 KV 4,096블록을 썼다. Qwen2.5-14B-Instruct에는 이전 장문 검증에서 쓴 2,576블록을 `--smoke-kv-blocks`로 넘겼다. 두 실행은 45초·평균 초당 8건·prefix cache 128블록을 유지했다. [최초 실행과 14B 판정](../../benchmarks/evidence/20261120-w5-serving-smoke-extended/verified-smoke.json) · [MHA 재검증](../../benchmarks/evidence/20261120-w5-serving-smoke-extended/recheck/verified-smoke.json)

| 항목 | SmolLM2-1.7B-Instruct 재검증 | Qwen2.5-14B-Instruct |
|---|---:|---:|
| API 통과 | 39/39 | 40/40 |
| 혼합 부하 정상 완료 | 315 | 338 |
| 지원 범위에 맞는 도구 응답 | HTTP 400 거부 23 | 필수 호출 완료 23 |
| 의도적 연결 종료와 서버 취소 증가 | 21 / 21 | 21 / 21 |
| 예상 밖 HTTP·SSE 오류와 429 | 0 / 0 | 0 / 0 |

SmolLM2 최초 실행은 39건 중 33건이 통과했다. 일반·동시성·스키마 검사는 통과했고 해당 채팅 템플릿의 도구 요청은 `unsupported_parameter`로 거부됐다. 구형 logprobs의 자연 종료에서 토큰 3개가 돌아왔고 기존 검사는 출력 상한인 5개를 요구했다. 재검증은 사용량과 네 logprobs 배열 길이·종료 사유를 맞추고, `--expect-tools-unsupported`에서 비스트리밍과 스트리밍 요청의 정확한 오류 응답을 각각 확인했다. [API 검사기](../../benchmarks/live/api_check.py) · [MHA 재검증 실행기](../../benchmarks/evidence/20261120-w5-serving-smoke-extended/recheck/campaign.sh)

두 완료 실행은 drain 및 shutdown 검사를 통과하고 GPU 6·7의 기존 20GiB 예약으로 복귀했다. 14B는 최초 캠페인의 도구 지원 모드로 통과했고, MHA는 새 도구 미지원 검사 모드에서 통과했다. 두 모델의 45초 혼합 부하는 작업 종료와 API 계약을 검증한다. [14B 용량과 준비된 모델 해시](../../benchmarks/evidence/20261120-w5-serving-smoke-extended/models.tsv) · [원본 재검산](../../benchmarks/evidence/20261120-w5-serving-smoke-extended/recheck/fetch-summary.sh)

## 배포 이미지의 커밋된 혼합 GPU iteration

2026-11-20 KST에 A100 7번에서 배포 이미지 `0a907408ec73`과 Qwen2.5-7B-Instruct를 검증했다. `native-bf16-varlen-v1`에 배치 예산 512, 요청별 prefill 한도 256, 활성 요청 한도 2를 지정했다. API 40건을 통과한 뒤 제어 요청과 혼합 요청을 같은 컨테이너에서 실행했다. 첫 기동은 512의 예산과 256에서 끝나는 기본 shape 목록의 불일치로 중단됐으며, 명시적으로 512행 버킷을 추가한 뒤 검증했다. [실행 순서](../../benchmarks/evidence/20261120-w5-mixed-stage/campaign.sh) · [대조와 체크섬](../../benchmarks/evidence/20261120-w5-mixed-stage/verified-mixed.json)

단독 제어군은 26토큰 입력에서 256개를 생성한 요청과 3,312토큰 입력에서 47개를 생성한 요청이다. 단독 실행에서 관측한 최대 실제 행 수는 각각 26과 256이었다. 혼합 실행에서는 첫 요청의 생성 토큰을 8개 이상 받은 뒤 활성 요청 하나와 1행짜리 커밋을 확인하고, 같은 긴 입력을 두 번째 요청으로 보냈다. 세 쌍에서 각각 12개, 합계 36개의 커밋된 iteration을 관측했다. 각 관측의 실제 행은 257, 준비된 dense shape는 512행, 패딩은 255행이었다. 같은 스냅샷의 GPU 실행 시간은 양수였다. 첫 요청은 decode 중이고 두 번째 요청의 prefill 한도는 256이므로 각 관측은 prefill 256행과 decode 1행을 같은 iteration에 포함한다. [스케줄러의 요청별 행 한도](../../crates/riley-scheduler/src/scheduler.rs#L1827) · [실행 행 기록](../../crates/riley-server/src/engine.rs#L4276) · [관측값 검산기](../../benchmarks/evidence/20261120-w5-mixed-stage/verify_mixed.py)

혼합 응답 6건 모두 각각의 단독 실행과 입력·생성 토큰 ID, 텍스트, 종료 사유 및 사용량이 일치했다. Drain의 활성·대기 요청과 KV는 각각 0이었다. 정상 종료에서 device·pinned 할당이 0이었고 GPU 6·7의 기존 20GiB 예약을 확인했다. 이 결과는 지정한 512행 예산의 실험 설정과 두 고정 요청에 해당한다. [실행 명령](05-documentation-and-operations.md#한-gpu-iteration의-prefilldecode-대조) · [집계 재현](../../benchmarks/evidence/20261120-w5-mixed-stage/fetch-summary.sh)

## 배포 이미지의 32B TP=2 장문 검증

2026-11-20 KST에 배포 이미지 `0a907408ec73`의 Qwen2.5-32B-Instruct를 NVLink로 연결된 A100 6·7번에서 실행했다. 컨테이너에 두 GPU의 UUID만 노출했고 같은 Riley 프로세스의 CUDA 컨텍스트를 두 장에서 확인했다. `native-bf16-varlen-v1`, 문맥 8,192토큰, 활성 요청 한도 32, KV 2,048블록을 사용했다. 도구 호출을 포함한 API 40건을 통과했다. [실행 조건과 집계](../../benchmarks/evidence/20261120-w5-tp2-holdout/verified-tp2.json)

고정 입력은 256·512·1,024토큰 앞뒤의 아홉 경계와 7,836·7,917토큰의 장문 두 개다. 출력을 요청당 최대 32토큰으로 제한했다. C32는 모든 입력을 한 번씩 사용하고 짧은 일곱 입력을 세 번씩 추가했다. 이 구성의 최대 KV 수요는 블록당 16토큰 기준 2,018블록이다. [입력 구성과 대조 코드](../../benchmarks/live/tp_holdout_check.py)

| 클라이언트 동시성 | 완료 요청 | 관측한 최대 활성 요청 | C1과 정확히 일치 |
|---|---:|---:|---:|
| C1 | 11 | 1 | 기준 응답 |
| C8 | 11 | 8 | 11/11 |
| C32 | 32 | 31 | 32/32 |

동시 요청 43건의 입력·생성 토큰 ID, 텍스트, 종료 사유와 사용량이 각각의 C1 응답과 일치했다. 서버는 종료 코드 0을 기록했고 활성·대기 요청, 요청 KV 및 device·pinned 할당은 모두 0이었다. GPU 6·7은 각각 기존 홀더의 20GiB 예약으로 복귀했다. 이 시험의 판정 범위는 해당 배포 이미지와 고정 입력에서 동시성에 따른 응답 재현성 및 종료 자원 회수다. [실행 명령](05-documentation-and-operations.md#32b-모델의-tp2-장문-검증) · [집계 검산](../../benchmarks/evidence/20261120-w5-tp2-holdout/verify_tp2.py)

## 취소 순서 수정 이후의 배포 이미지

2026-11-21 KST에 `36aa18805`의 CUDA 12.4 배포 이미지를 검증했다. 호스팅 CI는 전체 커밋 아카이브를 사용했다. A100의 재빌드는 기승인 `d3d8fe6e5` 사본에 변경 패치를 적용하고, 배포 입력 1,037개를 `36aa18805`의 Git 객체 ID와 대조한 뒤 별도 입력 아카이브를 만들었다. CPU 패키지 검사에서 비루트 사용자·오프라인 CLI·동적 링크를 확인했고, 두 이미지의 런타임 파일 21개가 모두 같은 SHA-256을 기록했다. [호스팅 CPU 보고서](../../benchmarks/evidence/20261121-w5-current-package/hosted-report.json) · [A100 CPU 보고서](../../benchmarks/evidence/20261121-w5-current-package/a100-report.json) · [빌드 재현](../../benchmarks/evidence/20261121-w5-current-package/campaign.sh)

A100 7번에서 이미지 `b731f48cc6ec`와 Qwen2.5-7B-Instruct를 사용했다. API 40건이 통과했다. 활성 요청 한도 32, 문맥 8,192토큰, KV 16,384블록, prefix cache 128블록으로 45초간 초당 8건을 보냈다. 정상 완료 338건, 의도적 연결 종료와 서버 취소 각각 21건, HTTP·SSE 오류 및 과부하 0건을 확인했다. 서버는 코드 0으로 종료했고 이후 활성·대기 요청, 요청 KV, device·pinned 할당은 모두 0이었다. GPU 6·7번은 같은 홀더의 20GiB 예약을 확인했다. [실행 검산](../../benchmarks/evidence/20261121-w5-current-package/verified-summary.json)

이 45초 부하에서는 새 배포 이미지의 API와 취소·종료 회수를 확인했다. A100 빌드를 재현할 때는 `engine.patch.gz`와 `source-objects.tsv.gz`를 원래 이름으로 풀고 `dispatch.sha256`을 확인한다. [릴리스 판정](06-candidate-qualification.md) · [원본 검산과 요약 가져오기](../../benchmarks/evidence/20261121-w5-current-package/fetch-summary.sh)

### 새 이미지의 30분 혼합 부하

같은 이미지 `b731f48cc6ec`와 Qwen2.5-7B-Instruct에서 캐시 비활성은 A100 6번, 캐시 활성은 A100 7번으로 동시에 실행했다. 두 조건에 같은 합성 요청과 seed를 적용했고 각 GPU에서 초기와 후반 지연을 대조했다. API 40건이 각 조건에서 통과했다. 서버 실행 인자는 prefix cache 128블록 지정의 유무만 달랐다. [실행 절차](../../benchmarks/evidence/20261121-w5-current-image-soaks/campaign.sh) · [검산 요약](../../benchmarks/evidence/20261121-w5-current-image-soaks/verified-summary.json)

| 30분 조건 | 정상 완료 / 의도적 연결 종료 | TTFT p95 변화 | 전체 완료 p95 변화 | 정밀 RSS 증가 |
|---|---:|---:|---:|---:|
| 캐시 비활성 | 13,749 / 700 | +1.41% | −13.56% | +572KiB |
| 캐시 활성 | 13,749 / 700 | +1.16% | −13.57% | +636KiB |

각 조건에서 서버 연결 종료·취소 카운터가 각각 700 증가했고 HTTP·SSE 오류와 과부하는 0건이었다. 지연은 허용 범위 안이었다. 두 조건의 `smaps_rollup`에서 RSS 증가량과 익명 RSS 증가량이 같았고, RSS 증가 0 기준만 실패했다. 서버는 각각 코드 0으로 종료했고 활성·대기 요청, 요청 KV와 device·pinned 할당은 모두 0이었다. 시험 뒤 두 GPU의 기존 20GiB 예약을 확인했다. [정밀 RSS 및 종료 검산](../../benchmarks/evidence/20261121-w5-current-image-soaks/verify_soaks.py)

## 90초 사전 검증

프롬프트 KV 캐시에 128블록을 할당했다. 초당 평균 8건의 합성 요청으로 일반·스키마·logprobs·필수 도구·장문·긴 출력·클라이언트 연결 종료를 섞었다. 장문 요청은 여덟 개의 프롬프트 앞부분을 반복 사용했다.

| 측정 | 결과 |
|---|---:|
| 정상 완료 | 691건 |
| 스키마 완료·독립 값 검사 | 90건 |
| 필수 도구 완료·인수 검사 | 40건 |
| 의도적으로 연결을 끊은 요청 | 39건 |
| 서버 disconnect / cancel 카운터 증가 | 각각 39건 |
| HTTP·SSE 오류와 429 | 0건 |
| drain 뒤 활성·대기·요청 KV 블록 | 각각 0 |
| 종료 뒤 device·pinned 할당과 바이트 | 각각 0 |

부하 직전과 drain 뒤 native 할당 메트릭이 같았고, `SIGTERM`을 받은 서버는 종료 코드 0으로 최종 메트릭을 남겼다. GPU 7의 동일 홀더는 이 검증 뒤 20GiB 대기 예약으로 복귀했다. 90초 구간에는 장시간 지연 가드레일을 적용하지 않았다. [요청 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/soak.json) · [종료 메트릭](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/shutdown-metrics.json) · [예약 복귀](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/idle-after.json)

## 30분 본 실행

같은 바이너리·모델·입력 생성기와 16,384블록 KV 풀에서 30분씩 실행했다. 첫 조건은 2026-11-19 01:39 KST에 시작했고, 두 번째 조건에는 prefix cache 128블록을 할당했다. 두 조건의 바이너리·체크포인트·토크나이저·부하 생성기 해시와 서버 인자를 대조했다. [GPU 실행 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/run_soak.py) · [순차 실행 명령](../../benchmarks/evidence/20261119-w5-mixed-soak/campaign.sh)

| 측정 | cache off | cache on |
|---|---:|---:|
| 정상 완료 / 클라이언트 연결 종료 | 13,749 / 700 | 13,749 / 700 |
| 스키마 유효 완료 / 필수 도구 완료 | 1,739 / 708 | 1,739 / 708 |
| HTTP·SSE 오류와 429 | 0 | 0 |
| 서버 disconnect / cancel 증가 | 700 / 700 | 700 / 700 |
| TTFT p95 초기 → 후반 | 327.14 → 330.79ms (+1.11%) | 320.57 → 323.36ms (+0.87%) |
| 전체 완료 지연 p95 초기 → 후반 | 4,784.31 → 4,149.00ms (−13.28%) | 4,770.78 → 4,128.77ms (−13.46%) |
| RSS 후반 변화 | +0.75MiB | 0.00MiB |
| drain 뒤 활성·대기·요청 KV 블록 | 각각 0 | 각각 0 |
| 종료 뒤 device·pinned 할당과 바이트 | 각각 0 | 각각 0 |

두 조건 모두 p95 후반 증가 5% 이하와 자원 회수 조건을 통과했다. 부하 직전과 drain 뒤 native 할당값이 같았고 서버·클라이언트 모두 종료 코드 0을 남겼다. GPU 7은 각 조건 종료 뒤 동일 홀더의 20GiB 대기로 복귀했다. 전체 시험 종료 시점에는 GPU 6·7 모두 20GiB 예약 상태였다. [cache off 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off/summary.json) · [cache on 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-on/summary.json) · [GPU 6 예약](../../benchmarks/evidence/20261119-w5-mixed-soak/gpu6-final.json) · [GPU 7 예약](../../benchmarks/evidence/20261119-w5-mixed-soak/gpu7-final.json)

각 조건의 운영 메트릭 32개 표본에서 24개 `batch_shapes` 슬롯 중 21개가 구성됐다. 비어 있는 슬롯은 모두 0이었고, 저장된 스텝 수와 GPU 시간 표본 수가 각 bucket에서 일치했다. 두 조건 모두 진단 링에서 오래된 요청 요약 13,425개가 제거됐다. 부하 측정의 오류 수와 표본 수는 별도로 기록됐다. [bucket 검증 집계](../../benchmarks/evidence/20261119-w5-mixed-soak/live-batch-shapes.json) · [cache off 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off/soak.json.gz) · [cache on 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-on/soak.json.gz) · [양쪽 실행 근거 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/SHA256SUMS)

90초 사전 검증과 30분 시험의 도구는 native 할당과 RSS를 각각 기록한다. Cache off의 후반 RSS 표본은 900초부터 15,161.14MiB였고 1,560초 표본부터 15,161.89MiB였다. Cache on은 900초부터 종료 직전까지 15,163.53MiB였다. Cache off의 0.75MiB 변화는 [운영 품질 기준](../goal.md#33-운영-품질)의 RSS 증가 0 조건에 남겨 둔다.

### 종료 관측 버퍼

기본 cache off/on, cache-off 반복, allocator 계측, 정밀 RSS 실행의 30분 결과 다섯 건에서 정상 완료 13,749건과 의도적 연결 종료 700건을 각각 확인했다. `dropped_observations`는 모든 실행에서 0에서 13,425로 증가했다. 종료 레코드 14,449건에서 기본 보존 한도 1,024건을 빼면 기록된 증가량과 같다. 이 값은 최근 종료 관측 레코드의 퇴거 수로 기록했고 요청 종료와 지연은 클라이언트 원본으로 대조했다. [관측 버퍼와 API](../api.md#get-metrics) · [정밀 실행 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/smaps-rollup/result/cache-off/soak.json.gz)

### Cache off RSS 재현

동일 GPU 7, 바이너리, 모델, 실행 옵션과 부하 도구로 30분 cache-off를 반복했다. 정상 완료 13,749건, 의도적 연결 종료 700건, HTTP·SSE 오류와 overload 0건을 기록했다. 후반 TTFT p95는 1.24% 증가했고 완료 지연 p95는 13.33% 감소했다. Ready와 drain 시 native 할당이 같았고 종료 후 GPU·pinned 할당과 활성 KV는 0이었다. GPU 7은 동일 홀더의 20GiB 예약으로 복귀했다. [재현 명령](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off-repeat/campaign.sh) · [원본 및 입력 대조](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off-repeat/result/comparison.json) · [종료 후 예약](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off-repeat/result/idle-after.json)

후반 RSS는 900초의 15,162.35MiB에서 15,163.10MiB로 0.75MiB 증가했다. 1,080초부터 drain까지 같은 값을 유지했다. 서버 `/proc`의 15초 간격 표본 95개에서 두 RSS 수준의 차이는 `RssAnon` 768KiB였고 파일·공유 페이지와 RSS 최고치는 일정했다. 관측기는 실행 도중 시작했다. 두 cache-off 시험에서 후반 증가가 재현되어 RSS 증가 0 항목은 미충족 상태다. 익명 페이지의 증가 원인을 확인하려면 allocator에 남은 공간과 사용 중인 호스트 할당을 추가로 측정해야 한다. [RSS 표본](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off-repeat/result/rss-observer/samples.jsonl) · [대조 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/compare_cache_off_repeat.py) · [원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/cache-off-repeat/result/SHA256SUMS)

같은 GPU 7과 Riley 바이너리로 allocator를 계측한 30분 진단을 마쳤다. 서버 PID의 `/proc`과 glibc `mallinfo2` 표본 130개를 수집했고, 정상 완료 13,749건과 의도적 연결 종료 700건 및 오류·overload 0건을 기록했다. 후반 첫 표본과 drain의 RSS는 각각 15,162.31MiB였다. 종료 뒤 native 할당과 활성 KV는 모두 0이었고, GPU 7은 같은 홀더의 20GiB 예약으로 돌아갔다. [진단 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/campaign.sh) · [원본과 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/result/SHA256SUMS) · [glibc 2.35 통계 정의](https://www.sourceware.org/glibc/manual/2.35/html_node/Statistics-of-Malloc.html)

`soak.json`의 원격 수정 시각과 마지막 drain 표본을 기준으로 930–990초와 1,710–1,770초의 표본을 네 개씩 대조했다. 익명·파일·공유 RSS와 `malloc` mmap은 두 구간에서 동일했다. Glibc arena 사용 바이트 중앙값은 +1.25MiB였으며 두 구간의 사용 바이트 범위가 겹쳤다. 다른 조건이 일치하는 직전 비계측 반복 대비 이 실행의 후반 TTFT p95는 +0.32%, 전체 완료 p95는 +0.01%였다. 실행당 한 번의 비교로는 계측 비용과 앞선 RSS 증가 원인을 확정할 수 없다. [두 실행 원본 대조](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/result/comparison.json) · [집계 코드](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/compare_allocator_trace.py)

### A100 호스트의 RSS 계측 단위

같은 호스트의 Linux `6.8.0-1017-aws`에서 별도 CPU 프로세스가 익명 페이지 2MiB를 64KiB씩 사용했다. `/proc/self/status`의 `RssAnon`은 32개 구간 중 29개에서 값이 유지됐고 세 구간에서 768KiB씩 증가했다. 같은 시점의 `smaps_rollup`은 구간별로 64–68KiB씩 증가했다. Linux는 `status` RSS를 비동기로 집계하며, `smaps_rollup`은 페이지 테이블을 읽은 결과를 합산한다. 두 cache-off 시험의 0.75MiB는 이 호스트에서 확인한 768KiB 계측 단위와 같다. Riley의 실제 상주 페이지 증가량을 확인하려면 같은 부하에서 후반 기준점과 drain의 `smaps_rollup`을 대조해야 한다. [호스트 CPU 실험](../../benchmarks/evidence/20261119-w5-mixed-soak/rss-accounting/probe.json) · [실험 소스](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/probe_linux_rss_accounting.py) · [Linux `/proc` 문서](https://docs.kernel.org/filesystems/proc.html)

같은 바이너리와 30분 cache-off 부하를 사용하고 후반 첫 표본과 drain에서 `status`와 `smaps_rollup`을 함께 읽는 시험을 등록했다. A100 호스트의 별도 CPU 프로세스에서 `smaps_rollup` 파싱을 확인했다. 집계기는 기존 반복과 바이너리·부하·GPU 예약을 대조하고, 같은 두 시점의 원본에서 RSS 지표별 증가량과 판정을 다시 계산한다. GPU 실행은 14B prefill 후보 시험 종료 뒤 시작했다. [정밀 계측 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/smaps-rollup/campaign.sh) · [CPU 파싱 사전검사](../../benchmarks/evidence/20261119-w5-mixed-soak/smaps-rollup/smaps-cpu-check.json) · [두 RSS 지표 대조 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/compare_smaps_rollup.py)

### 정밀 RSS 재실행 결과

GPU 7의 30분 재실행에서 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. HTTP·SSE 오류와 overload는 0건이었다. Drain 뒤 활성 요청·대기 요청·요청 KV는 0이었고, 서버 종료 후 device·pinned 할당도 각각 0이었다. GPU 7은 같은 홀더 PID와 시작 시각을 유지해 20GiB 예약으로 돌아갔다. [실행 원본과 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/smaps-rollup/result/SHA256SUMS)

같은 서버 PID의 900초와 drain 시점에서 `/proc/PID/status` RSS는 15,161.46MiB로 같았다. `smaps_rollup` RSS는 15,544,012KiB에서 15,544,712KiB로 증가했고 익명 페이지도 700KiB 증가했다. 이 실행에서 RSS 증가 0 조건을 충족하지 못했다. 정밀 표본의 읽기 시간은 180ms와 252ms였으며, 기존 비계측 반복 대비 후반 TTFT p95와 전체 완료 p95는 각각 0.17%, 0.28% 낮았다. 이 대조는 증가한 페이지가 익명이라는 점까지 확인하며, 할당 지점은 아직 확인하지 못했다. [원본 대조와 판정](../../benchmarks/evidence/20261119-w5-mixed-soak/smaps-rollup/result/comparison.json)

GPU 7에서 같은 바이너리·모델·부하로 30분을 다시 실행했다. 후반 첫 표본과 drain에서 `smaps_rollup` RSS와 익명 RSS가 각각 **812KiB 증가**했다. `smaps`의 매핑별 합계도 같았으며, 812KiB 전체가 익명 private 매핑의 `Rss`·`Anonymous`·`Private_Dirty`에 반영됐다. 가장 큰 개별 매핑의 익명 페이지는 304KiB 증가했다. `[heap]`과 이름이 표시된 스택의 RSS는 유지됐다. 두 원본의 결과는 증가한 매핑 종류를 확정하며, 할당 호출 위치는 추가 확인 범위다. [재실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/anon-mappings/campaign.sh) · [A100 파싱 사전검사](../../benchmarks/evidence/20261119-w5-mixed-soak/anon-mappings/smaps-cpu-check.json) · [대조 검사기](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/compare_anon_mappings.py)

이번 실행은 `237f12546`의 정밀 판정을 적용했다. 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했고, HTTP·SSE 오류와 429 및 종료 후 native 할당 잔류는 0이었다. RSS 증가 0 항목이 실패해 검사 프로세스는 코드 1을 반환했다. 서버는 코드 0으로 종료했고 GPU 7은 동일 홀더의 20GiB 예약으로 돌아갔다. [원본과 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/anon-mappings/result/SHA256SUMS) · [원본 대조](../../benchmarks/evidence/20261119-w5-mixed-soak/anon-mappings/result/comparison.json)

익명 private 매핑 증가에 대한 단일 변수 대조를 등록했다. 같은 Riley 바이너리와 cache-off 부하를 기본 glibc 설정 그리고 서버 프로세스에만 `MALLOC_ARENA_MAX=2`를 지정한 설정으로 각각 30분 실행한다. 실행 중 서버 PID에서 해당 환경값을 확인하고, 두 조건의 익명 RSS·매핑 분포와 후반 TTFT·전체 완료 p95를 대조한다. A100 6번의 14B 시험이 끝나면 GPU 7의 동일 홀더로 순서대로 실행한다. [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/campaign.sh) · [결과 대조 검사기](../../benchmarks/evidence/20261119-w5-mixed-soak/tools/compare_glibc_arenas.py)

glibc 2.35의 두 조건을 각각 30분 실행했다. 두 조건 모두 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했고 요청·지연·종료 자원 검사를 통과했다. 정밀 RSS 증가 0 조건만 실패했다. 두 실행을 마친 뒤 GPU 7은 같은 홀더의 20GiB 예약으로 복귀했다. [기준군 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/result/control-comparison.json) · [두 조건 대조와 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/result/paired-comparison.json)

| 설정 | 정밀 RSS 증가 | 익명 private RSS 증가 | 가장 많이 증가한 유지 매핑 |
|---|---:|---:|---:|
| 기본값 | +932KiB | +932KiB | +460KiB |
| `MALLOC_ARENA_MAX=2` | +3,648KiB | +3,648KiB | +3,648KiB |

제한 조건의 후반 TTFT p95와 전체 완료 p95는 각각 0.22%, 0.19% 감소했다. 같은 조건의 RSS 증가는 2,716KiB 더 컸고 증가분 전체가 유지된 익명 private 매핑 하나에 집중됐다. arena 제한은 채택하지 않았다. 이 매핑을 만든 할당 경로는 두 조건의 측정만으로 판단할 수 없다.

### 배포 이미지의 혼합 부하

배포 이미지 GPU 검사기에 운영 시험을 연결했다. API 검사 후 30분 동안 초당 8회의 합성 도착과 공통 prefix 8개를 적용한다. Cache-off는 `--prefix-cache-blocks`를 생략하고 cache-on은 128개를 지정한다. 두 조건 모두 활성 요청 32개, 문맥 8,192토큰, 생성 상한 512토큰, KV 블록 16,384개를 사용한다. [배포 GPU 검사기](../../ci/check_distribution_gpu.py)

Docker가 보고한 호스트 PID와 컨테이너 ID를 각 표본 전후로 대조하고, 컨테이너 사용자로 `/proc/1/status`와 `/proc/1/smaps_rollup`을 읽는다. 기존 부하 검사기가 요청 처리, 취소, 지연 변동, RSS 증가 0을 판정한다. 정밀 표본 누락과 RSS 증가는 실패로 기록한다. 정상 종료와 자원 회수 검사가 끝나면 이미지 ID, 바이너리 해시, 혼합 부하 원본 및 정리 지표의 해시를 보존한다. CPU 테스트에서 프로세스 변경 검출과 RSS 실패 후 종료 경로를 확인했다. [Linux 부하 검사기](../../benchmarks/live/soak.py) · [이미지 검사 방법](01-validation-result.md#배포-이미지-gpu-검사)

첫 cache-off 시도에서는 검사기가 `--prefix-cache-blocks 0`을 넘겼고 서버가 준비 단계에서 거부했다. 부하 측정은 시작되지 않았고 GPU는 동일 홀더의 20GiB 예약으로 복귀했다. 인자 생략과 128개 지정의 회귀 검사를 추가했다. cache-on 종료 뒤 같은 이미지와 체크포인트에서 수정한 cache-off 시험을 완료했다. [첫 시도와 재실행 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json)

AWS에서 재빌드한 `d3d8fe6e5` 이미지의 cache-on과 cache-off 시험은 A100 6번에서 각각 30분 동안 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. 두 실행의 이미지·Riley 바이너리·체크포인트·GPU 홀더는 같았으며 서버 인자의 차이는 cache-on의 `--prefix-cache-blocks 128`이었다. 요청 오류와 과부하는 0건이었다. 약 900초 시점과 모든 요청을 처리한 약 1,804초 시점의 정밀 RSS를 대조했다. [혼합 부하 결과와 원본 체크섬](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json)

| 설정 | 후반 TTFT p95 | 후반 전체 완료 p95 | 후반 정밀 RSS 증가 | 익명 RSS 증가 |
|---|---:|---:|---:|---:|
| Cache-on | +1.04% | −13.71% | +616KiB | +616KiB |
| Cache-off | +0.82% | −13.93% | +700KiB | +700KiB |

두 실행에서 일반 `VmRSS` 변화는 0MiB였고, 정밀 RSS 증가 0 조건만 실패했다. 두 서버는 각각 종료 코드 0으로 끝났고 활성·대기 요청, KV 블록, device·pinned 할당은 모두 0이었다. GPU 6은 각 시험 뒤 같은 홀더의 20GiB 예약으로 복귀했다.

### 배포 이미지의 glibc 반환 진단

같은 이미지와 cache-off 부하에서 약 900초 시점, drain 완료 직후, `malloc_trim(0)` 직전·직후의 `smaps`를 읽는 진단을 준비했다. 시험은 정밀 RSS 판정과 요청·지연 검사를 마친 뒤 컨테이너의 Riley PID 1에서 `malloc_trim(0)`을 한 번 호출한다. 두 구간에서 범위와 속성이 같은 매핑을 비교해 증가한 익명 RSS, drain 후 자연 변동, 반환 호출 뒤 감소량을 각각 기록한다. 같은 매핑의 변화량으로는 개별 페이지나 최초 할당 호출 위치까지 판별할 수 없다. [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/packaged-allocator-trim/campaign.sh) · [`malloc_trim` 동작](https://man7.org/linux/man-pages/man3/malloc_trim.3.html)

배포 이미지의 CPU 사전시험에서 glibc 2.35와 같은 계측 라이브러리로 8MiB 할당을 사용한 뒤 해제했다. 반환 호출 전후의 익명 RSS는 8,308KiB에서 124KiB로 줄었고 glibc가 기록한 사용 중인 arena 바이트는 944바이트로 같았다. A100 호스트의 실제 `smaps` 파싱도 확인했다. Riley 부하 진단은 W4의 HF·Riley KV 대조 이후 GPU 6에서 실행하도록 등록했다. 재현할 때는 기존 `anon-mappings/attribute_anon_rss.py`를 같은 검사 디렉터리에 복사하고 `gzip -dc harness.patch.gz >harness.patch`로 검사기 패치를 복원한 뒤 `dispatch.sha256`을 확인한다. [CPU 사전시험과 라이브러리 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/packaged-allocator-trim/cpu-preflight.json) · [재사용 파서](../../benchmarks/evidence/20261119-w5-mixed-soak/anon-mappings/attribute_anon_rss.py)

W4의 첫 시도가 파서 체크섬 검사에서 종료된 뒤 GPU 6의 RSS 진단이 30분 부하를 시작했다. W4 재실행은 진행 중인 RSS 진단 뒤로 배치했다. [W4 재실행 기록](../../benchmarks/evidence/20261119-w4-layer-trace/hf-kv-comparison/retry/retry.json)

같은 배포 이미지의 cache-off 진단은 30분 동안 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. 요청·지연·종료 자원 검사가 통과했고, 반환 호출 전 정밀 `smaps_rollup` RSS는 612KiB 증가해 증가량 0 기준을 넘었다. `smaps` 네 저장본을 별도로 대조했다. 부하 중 증가한 익명 RSS 604KiB는 시작 위치와 메타데이터가 일치하는 익명 private 매핑 32개에 있었다. Drain 이후 반환 직전까지 익명 RSS는 같았다. [동일 이미지 검산 및 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/packaged-allocator-trim/verified-result.json)

| 매핑의 부하 중 범위 | 증가한 매핑 | 익명 RSS 증가 | 반환 호출 뒤 같은 매핑의 익명 RSS 감소 |
|---|---:|---:|---:|
| 그대로 유지 | 13개 | +272KiB | 1,120KiB |
| 끝 위치 변경 | 19개 | +332KiB | 492KiB |

`malloc_trim(0)` 뒤 32개 모두 익명 RSS가 줄었고, 두 매핑에 합계 16KiB가 중간 시점보다 더 남았다. 전체 프로세스의 익명 RSS는 42,796KiB 줄었고 glibc의 arena 사용 중 바이트는 407,718,704로 같았다. 이 결과는 부하 중 커졌던 매핑에서 glibc가 반환할 수 있는 페이지가 있었음을 확인한다. 개별 페이지와 최초 할당 지점은 매핑 단위 집계의 범위 밖이다. 서버는 코드 0으로 종료했고 GPU 6은 동일 홀더의 20GiB 예약으로 복귀했다. 운영 판정은 반환 전 기준에 따라 보류다. [검산 코드](../../benchmarks/evidence/20261119-w5-mixed-soak/packaged-allocator-trim/verify_allocator_trim.py)

### 배포 이미지의 주기적 메모리 반환 후보

같은 `d3d8fe6e5` 이미지에서 Riley 프로세스의 실험용 스레드가 31초 간격으로 `malloc_trim(0)`을 호출한다. 첫 호출 시각은 프로세스 기동에서 정해지며, 이후 호출은 직전 반환 종료 시각부터 센다. 31초는 검사기의 60초 표본 주기와 서로 다른 위상에서 동작하도록 정한 실험값이다. 각 호출의 소요 시간·반환 여부를 저장한다. 캐시 off·on에서 기존 API 40건과 같은 30분 부하를 각각 실행하고, 저장된 기준 실행과 결과를 비교한다. `soak.py`·`load.py` 원본의 해시와 RSS 증가량 0·지연 드리프트 판정을 그대로 사용한다. [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/campaign.sh) · [대조 검사기](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/verify_periodic_trim.py)

동일 이미지의 CPU 사전시험은 400ms 간격에서 네 번 호출했고, 익명 RSS가 8,316KiB에서 124KiB로 줄었다. A100 6번의 cache-off 기동·API 검사·부하 초반에서 Riley PID 1의 반환 호출 14건을 검산했다. 첫 14건의 호출 시간 p95는 2.27ms였다. 두 캐시 설정의 실제 RSS와 요청 지연을 확인한 뒤 후보를 판정한다. 재현 절차는 이전 `packaged-allocator-trim/trim_smoke.c`를 검사 디렉터리에 복사하고 `gzip -dc harness.patch.gz >harness.patch`로 패치를 복원한 뒤 `dispatch.sha256`을 검증한다. [사전시험 원본·체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/cpu-preflight.json) · [A100 초반 반환 기록](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/early-telemetry.jsonl)

Cache-off의 30분 실행에서 후반 RSS는 60KiB 증가했다. 기준 실행의 증가량은 700KiB였으며, 두 실행 모두 익명 RSS 변화가 전체 증가분과 같았다. 주기적 반환 실행의 유일한 실패 항목은 RSS 증가 0 조건이었다. 후반 TTFT p95는 335.42ms, 완료 지연 p95는 4,169.17ms로 기준 실행 대비 각각 0.37%, 0.28% 높았다. 같은 실행의 전반 대비 후반 p95 변화는 TTFT +1.58%, 완료 지연 −13.46%로 5% 기준 이내였다. [원본 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/result/cache-off-comparison.json) · [전용 경로 원본의 해시](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

API 40건이 통과했고 합성 부하는 기준 실행과 동일하게 정상 완료 13,749건, 의도적 연결 종료 700건으로 집계됐다. 반환 호출 66회에서 모두 페이지를 반환했고, 호출 시간 p95는 1.38ms였다. 종료 자원을 모두 회수한 뒤 GPU 6은 동일 홀더의 20GiB 예약으로 복귀했다.

Cache-on은 API 40건과 30분 부하의 모든 항목을 통과했다. 기준 실행의 정밀 RSS는 616KiB 증가했고, 주기적 반환 실행에서는 108KiB 감소했다. 두 값 모두 익명 RSS의 변화와 같았다. 후반 TTFT p95는 327.79ms로 기준 대비 0.61% 높았고, 완료 지연 p95는 4,126.81ms로 0.11% 낮았다. 같은 실행의 전반 대비 후반 p95 변화는 TTFT +1.49%, 완료 지연 −13.74%였다. 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했고 66번의 호출에서 모두 메모리를 반환했다. 호출 시간 p95는 1.15ms였다. 종료 후 GPU 6의 동일한 20GiB 예약을 확인했다. [두 설정의 검산 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/result/paired-comparison.json)

두 실행에 모두 적용되는 RSS 증가 0 조건은 cache-off에서 미충족이다. 동일 원본의 집계를 A100과 로컬에서 다시 계산해 파일의 바이트 일치를 확인했다. [검산 코드](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/verify_periodic_trim.py)

같은 cache-off 원본에서 일반 `/proc/PID/status`의 `VmRSS`는 688KiB 감소했다. 같은 구간의 `smaps_rollup`은 익명 RSS가 60KiB 증가했다고 기록했다. 중간과 drain의 정밀 RSS는 주기적 반환 실행에서 기준 실행보다 각각 44,596KiB와 45,236KiB 낮았다. 상주 메모리 절대량은 낮아졌으며 증가 0 판정은 정밀 표본을 따랐다.

후속 cache-off 진단은 원본 검사기가 `smaps_rollup`을 읽은 시간 범위를 기록한 뒤 인접한 매핑을 읽는다. 호스트와 Riley PID 1의 단조 시계 이름공간이 같은지 확인하고, 31초 주기 반환 호출의 시각·반환 여부를 정밀 표본 및 매핑에 연결한다. 두 읽기는 순차 실행되므로 매핑 합계와 정밀 표본의 차이도 결과에 남긴다. 같은 이미지·라이브러리·부하·RSS 판정을 사용하며 A100 6번에서 W3 추적 뒤 실행하도록 등록했다. [간격·매핑 진단 사전 검증](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/preflight.json)

이 cache-off 후속 실행에서 정밀 RSS와 익명 RSS는 각각 500KiB 증가했고, RSS 증가 0 조건만 실패했다. 중간과 drain의 정밀 표본은 각각 가장 최근 반환 호출이 끝난 뒤 4.16–4.45초와 8.99–9.27초에 읽혔다. 두 표본의 읽는 동안 반환 호출은 겹치지 않았다. 같은 31초 주기에서도 두 표본의 반환 이후 경과 시간이 달랐다. 이번 대조는 주기가 결과에 미친 정도까지 확정하는 데 필요한 반복은 포함하지 않는다. [판정과 반환 시각](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/result/timing-summary.json)

정밀 읽기 직후 매핑 합계는 중간에서 104KiB 높았고 drain에서 일치했다. 매핑 표본의 익명 RSS는 396KiB 증가했다. 크기와 메타데이터를 그대로 유지한 익명 private 매핑에서 28KiB, 나머지 익명 private 매핑의 순변화에서 368KiB가 관측됐다. API 40건·지연·자원 회수 검사는 통과했고 A100 6번은 동일한 20GiB 예약으로 돌아왔다. 서버의 자동 후처리는 이전 검산기 파일이 없어 종료 코드 2를 남겼다. 이미 완료된 GPU 파일을 보존하고 저장소의 검산기로 로컬에서 재계산했다. [원래 판정 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/result/original-gate.json) · [복구 및 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/result/recovery.json)

같은 이미지의 31초 cache-off 두 실행에서 인접 매핑 읽기를 추가한 쪽의 직접 읽기 경과 시간은 중간 314.34ms, drain 306.64ms였다. 인접 읽기가 있는 실행의 후반 TTFT p95는 기준보다 0.05% 높고 완료 지연 p95는 0.22% 낮았다. 측정 구간 GPU 스텝은 133,229회와 133,112회였고 CUDA 실행 시간 합계는 각각 1,757.30초와 1,756.25초였다. 네 할당 지표의 시작·종료 값은 두 실행에서 일치했다. 이는 한 쌍의 실행 간 차이이며 관측에 따른 지연과 RSS 변화를 판정하려면 반복이 더 필요하다. 프로세스 CPU 시간은 별도 측정 범위로 남아 있다. [관측 비용 대조](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/result/observer-cost.json) · [재검산 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/sample-timing/compare_observer_cost.py)

다음 대조는 독립 반환 스레드의 주기만 31초에서 7초로 바꾼다. 이전 drain 표본의 반환 후 대기는 약 9초였다. 두 캐시 설정에서 같은 30분 부하를 각각 실행하고 지연·종료·정밀 RSS 증가 0을 검산한다. 각 표본과 직전 반환 사이의 시차도 기록해 주기 변경의 영향을 본다. 검산기와 의존 파일은 새 실행 디렉터리에 함께 전송했고 두 기존 기준군의 원본 체크섬을 확인했다. GPU 7번의 QKV 서빙 대조가 종료되면 A100 6번에서 실행한다. [7초 대조 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/registration.json) · [고정 입력](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/inputs.json)

첫 실행에서 반환 라이브러리를 복사할 때 파일 권한이 `0600`으로 바뀌어 컨테이너의 UID 65532가 라이브러리를 읽지 못했다. 로더가 `LD_PRELOAD`를 무시한 상태로 cache-off 부하 30분과 API 40건을 끝냈다. 정밀 RSS와 익명 RSS는 각각 544KiB 증가했으며 주기적 반환 성능 판정에서 이 결과를 제외했다. 서버 종료 자원을 회수했지만 반환 로그가 없어 최종 GPU 보고서 생성이 실패했고 cache-on은 시작되지 않았다. A100 6번은 동일 홀더의 20GiB 예약으로 복귀했다. [첫 시도와 보존 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/first-attempt.json)

재실행은 새 디렉터리에 같은 라이브러리를 `0644`로 설치했다. 동일 이미지의 UID 65532에서 파일 체크섬과 실제 프로세스 매핑을 먼저 확인했다. 각 30분 부하는 Riley PID 1의 라이브러리 매핑과 7초 주기의 첫 반환 기록을 확인한 뒤 시작한다. Cache-off의 실제 서버에서 이 검사를 통과했다. 부하·API·정밀 RSS 판정 코드는 고정된 버전을 사용한다. 재현 시 등록된 `campaign.sh`와 원래 `observe_soak.py`를 복사해 두 gzip 패치를 적용하고 `dispatch.sha256`을 검증한다. [재실행 등록과 패치 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/registration.json)

7초 주기의 cache-off 30분 부하는 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. API 40건과 지연·종료 자원 검사가 통과했다. 정밀 RSS와 익명 RSS가 각각 **44KiB 증가**해 RSS 증가 0 항목만 실패했다. 같은 이미지의 비계측 기준은 700KiB 증가했고, 7초 조건의 drain RSS 절대값은 그 기준보다 46,328KiB 낮았다. 후반 TTFT p95와 완료 지연 p95는 기준 대비 각각 0.16%, 0.30% 높았다. 같은 홀더의 20GiB 예약을 복구한 뒤 cache-on 시험을 시작했다. [원본 체크섬과 두 검산 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/cache-off-verified.json)

반환 호출 294회에서 모두 메모리가 반환됐고 호출 시간 p95는 0.77ms였다. 정밀 중간 표본은 직전 반환이 끝난 뒤 6.44–6.79초에, drain 표본은 0.19–0.59초에 읽혔다. 두 읽기 모두 반환 호출과 겹치지 않았다. Drain에서 직전 반환과의 간격이 더 짧아도 익명 RSS는 늘었다. 인접 `smaps`의 익명 RSS 순증가는 184KiB였다. 시작 주소와 속성이 같고 끝이 바뀐 16개 매핑은 합계 228KiB 늘었고, 범위가 유지된 매핑은 합계 44KiB 줄었다. 두 표본 중 한쪽에만 나타난 시작 주소의 익명 RSS는 모두 0이었다. 중간 매핑을 읽은 값은 먼저 읽은 정밀 표본보다 140KiB 낮아 판정에는 원래의 `smaps_rollup` 두 표본을 사용한다. [반환 시각과 인접 매핑 대조](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/cache-off-verified.json)

7초 주기의 cache-on 30분 부하도 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. API 40건과 지연·종료 자원 검사가 통과했고, 정밀 RSS와 익명 RSS가 각각 **292KiB 증가**해 RSS 증가 0 항목만 실패했다. 같은 이미지의 비계측 기준은 616KiB 증가했으며, 7초 조건의 drain RSS는 그 기준보다 45,828KiB 낮았다. 후반 TTFT p95와 완료 지연 p95는 기준 대비 각각 1.00%, 0.24% 높았다. 두 cache 모드 모두 같은 20GiB 홀더로 복귀했다. [두 모드의 원본 판정 재현](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/paired-verified.json)

Cache-on의 반환 호출은 293회였고 모두 메모리를 반환했다. 중간 정밀 표본은 직전 반환 후 3.93–4.25초에, drain 표본은 4.70–5.05초에 읽혔다. 인접 매핑의 익명 RSS 순증가 196KiB는 전부 보조 힙과 같은 배치에서 나왔다. 두 모드 모두 7초 후보의 RSS 증가 0 기준을 충족하지 못해 이 후보의 적용 판단은 보류다. [정밀 표본 시각과 매핑 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/shorter-period/retry/paired-verified.json)

#### 보조 힙 배치 대조

7초 및 이전 31초 cache-off의 인접 매핑을 다시 검산했다. 두 실행에서 각각 64MiB에 정렬된 쓰기 가능 익명 매핑 36개가 바로 뒤의 접근 불가 구간까지 합해 정확히 64MiB를 이루었다. 이 배치는 64비트 glibc 2.35의 보조 힙 할당 코드와 일치한다. 두 매핑 표본의 익명 RSS 순증가인 7초 +184KiB와 31초 +396KiB는 각각 전부 이 배치에서 나왔다. 주소와 권한으로 판별한 결과이며 개별 allocator 청크의 생존 여부는 추가 검증 범위다. [매핑 검산과 입력 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/arena-shape-comparison.json) · [glibc 2.35 힙 구현](https://raw.githubusercontent.com/bminor/glibc/glibc-2.35/malloc/arena.c) · [glibc 2.35 할당 상한](https://raw.githubusercontent.com/bminor/glibc/glibc-2.35/malloc/malloc.c)

동일 이미지와 7초 메모리 반환을 사용하고 `glibc.malloc.tcache_count`만 0으로 바꿔 두 cache 모드의 30분 부하를 등록했다. glibc의 스레드별 소형 할당 캐시 비활성화가 RSS와 지연에 미치는 영향을 앞선 7초 시험과 비교한다. GPU 검사는 이미지의 서비스 계정으로 소형 할당·해제 동작을 CPU에서 검증한 후 시작한다. 실제 Riley PID 1의 환경과 7초 반환 기록을 확인하고 원래 RSS·요청 지연·API·종료 판정을 적용한다. CPU 검산기 모의시험에서 기존 RSS 실패와 설정 확인 누락의 처리 경로를 확인했다. 선행 QKV 시험 이후 A100 6번에서 실행된다. [검사 등록과 재현 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/registration.json) · [glibc 2.35 설정 문서](https://raw.githubusercontent.com/bminor/glibc/glibc-2.35/manual/tunables.texi)

스레드별 할당 캐시를 끈 `cache-off` 30분 실행에서 정밀 RSS와 익명 RSS가 각각 **144KiB 감소**해 증가 0 기준을 통과했다. 동일한 7초 반환 주기를 사용한 기본 할당 캐시 대조에서는 44KiB 증가했다. API 40건과 지연·자원 회수 검사가 모두 통과했다. 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했고, 같은 GPU 홀더의 20GiB 예약을 복구한 뒤 `cache-on`을 시작했다. 후반 TTFT p95와 완료 지연 p95는 앞선 대조보다 각각 0.17%, 0.16% 낮았다. [원래 판정과 독립 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/cache-off-verified.json)

정밀 RSS는 중간에 직전 반환 후 4.30–4.67초, drain에서 5.08–5.47초에 읽었고 두 읽기 모두 반환 호출과 겹치지 않았다. 인접 매핑의 익명 RSS는 220KiB 줄었으며 순변화는 앞서 관측한 보조 힙 배치에서 나왔다. 한 쌍에서 관측한 결과이고 이전 시험과 표본의 반환 후 시각이 달랐다. [표본 시각과 매핑 합계](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/cache-off-verified.json)

같은 설정의 `cache-on`에서 정밀 RSS와 익명 RSS가 각각 **88KiB 증가**해 증가 0 항목만 실패했다. 동일한 7초 반환 주기의 기본 할당 캐시 대조는 292KiB 증가했다. API 40건과 지연·종료 검사는 통과했고 정상 완료 13,749건, 의도적 연결 종료 700건을 기록했다. 앞선 대조에 견줘 후반 TTFT p95는 0.89% 낮고 완료 지연 p95는 0.05% 높았다. 두 실행 뒤 각각 기존 GPU 6 홀더의 20GiB 예약 복귀를 확인했다. [두 설정의 원본 판정과 독립 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/paired-verified.json)

`cache-on` 정밀 표본은 직전 반환 후 중간에서 5.15–5.38초, drain에서 5.93–6.19초에 읽었다. 인접 매핑에서는 익명 RSS가 48KiB 증가했고, 순증가 전부가 보조 힙과 같은 주소 배치에 있었다. 관측 결과의 합계는 allocator의 개별 보존 청크나 할당 호출을 식별하지 않는다. 이 설정의 두 캐시 조건 중 하나가 RSS 기준을 충족해 후보 적용과 릴리스 판정은 보류다. [매핑·표본 시각](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/paired-verified.json)

#### 스레드별 할당 캐시 설정 단독 대조

같은 이미지의 환경에 `GLIBC_TUNABLES=glibc.malloc.tcache_count=0`만 지정하는 대조를 등록했다. 준비 완료 직후와 부하 종료 뒤의 Riley PID 1에서 설정과 메모리 반환 라이브러리의 부재를 확인한다. W2 시험 종료 뒤 GPU 6번에서 `cache-on`과 `cache-off` 순으로 각각 API 40건과 30분 부하를 실행한다. 이전 기준군, 스레드별 할당 캐시 설정만 준 실행, 7초 주기 반환을 함께 적용한 실행의 이미지·실행 파일·부하·자원 회수를 검산한다. 양쪽 캐시 조건에서 고정 환경 설정만으로 RSS 증가 0을 충족하는지 판정한다. [실행 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/tcache-only/registration.json) · [검사기 패치와 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/tcache-only/campaign.sh) · [원본 대조 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/tcache-only/compare_tcache_only.py)

두 설정의 30분 부하가 완료됐다. 각 실행에서 정상 완료 13,749건, 의도적 연결 종료 700건을 기록했고 API 40건, 요청·지연·종료 자원 검사를 통과했다. 각각 종료 후 같은 GPU 6 홀더의 20GiB 예약이 복원됐다. [두 실행 검산과 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/tcache-only/paired-verified.json) · [원본 재현 검사기](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/tcache-only/verify_pair.py)

| 설정 | 기본 이미지 | 스레드별 할당 캐시 비활성화 | 7초 반환과 스레드별 할당 캐시 비활성화 |
|---|---:|---:|---:|
| `cache-on` 익명 RSS 변화 | +616KiB | +200KiB | +88KiB |
| `cache-off` 익명 RSS 변화 | +700KiB | +148KiB | −144KiB |

단독 설정은 두 모드 모두 증가량 0 기준에서 실패했다. 같은 두 단독 실행의 후반 TTFT p95는 초기 구간 대비 각각 1.33%, 1.29% 늘었다. 조건부 프레이밍 비교는 RSS 실패를 확인하고 W4 종료를 기다리는 상태로 전환했다. [후속 실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/campaign.sh)

### HTTP 요청 버퍼 용량의 CPU 대조

배포 기준 소스 `d3d8fe6e5`의 HTTP 프레이밍은 파싱 전부터 매 요청에 최대 헤더와 최대 본문을 합친 버퍼를 예약한다. 후보는 헤더 한도까지 먼저 예약하고, 검증한 `Content-Length`에 맞춰 필요한 용량만 추가로 확보한다. W5의 30분 요청 생성기를 같은 seed로 재생해 합성 요청 14,449개를 만들었다. 기준과 후보를 교대로 각 10회 실행해 모든 요청의 메서드·경로·본문 바이트를 확인했다. 분할 수신한 최대 본문은 양쪽에서 처리했고, 불완전·한도 초과 본문과 같은 읽기에 들어온 파이프라인은 양쪽에서 거부했다. [프레이밍 구현](../../crates/riley-server/src/http.rs) · [후보 패치](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/reserve-validated-body.patch.gz) · [재생 도구와 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/verified-framing.json)

| 요청 객체의 본문 버퍼 용량 | 기준 | 후보 |
|---|---:|---:|
| p50 | 1,064,960바이트 | 16,384바이트 |
| p99 | 1,064,960바이트 | 24,872바이트 |
| 최대 | 1,064,960바이트 | 25,040바이트 |

macOS ARM64에서 후보의 서버 라이브러리 테스트 101건이 통과했고 별도 루프백 시간 측정 1건은 기본값대로 건너뛰었다. 포맷 및 Clippy 검사도 통과했다. 위 용량은 CPU에서 확인한 벡터 예약량이다. 두 캐시 설정의 배포 이미지 30분 부하와 종료 시 정밀 RSS가 후보 채택 조건이다. [CPU 검증 기록](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/cpu-tests.json) · [재현 명령이 있는 스크립트](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/compare_framing.py)

두 캐시 설정의 스레드별 할당 캐시 단독 대조가 완료되면 이 후보의 실행 여부를 판단하도록 등록했다. 두 설정 모두 기준을 통과하면 별도 후보를 실행하지 않는다. RSS만 실패한 경우에는 W4 GPU 시험 뒤 승인본 `d3d8fe6e5`에 HTTP 후보 패치를 적용하고 같은 CUDA 12.4 Dockerfile로 별도 이미지를 빌드한다. 이미지 라벨에 기준 커밋과 변경된 아카이브·패치 해시를 결속한다. Riley 실행 파일을 제외한 런타임 파일 20개의 바이트가 같을 때 A100 6번에서 기존 API·30분 부하·RSS·지연·종료 검사를 진행한다. [A100 조건부 실행 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/registration.json) · [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/campaign.sh) · [이미지와 부하 비교기](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/compare_candidate.py)

별도 이미지 `e019db74d2f0`의 CUDA 12.4 빌드와 CPU 패키지 검사가 완료됐다. 빌드에 사용한 두 소스 아카이브의 항목 8,995개를 대조해 `crates/riley-server/src/http.rs` 한 파일만 달라진 것을 확인했다. 패키지의 나머지 런타임 파일 20개는 기준과 바이트가 일치했다. [소스 비교와 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/source-inventory-verified.json) · [이미지 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/cpu-comparison-verified.json)

A100 6번의 `cache-on` 컨테이너에서 API 40건이 통과했고 30분 부하가 시작됐다. 실행 중인 Riley의 바이너리 해시, GPU 프로세스, 캐시 128블록과 `GLIBC_TUNABLES`·`LD_PRELOAD`·`MALLOC_ARENA_MAX` 추가 설정의 부재를 확인했다. GPU 7번에는 같은 20GiB 예약이 유지됐다. 두 설정의 최종 RSS와 종료 후 자원 회수는 완료 영수증으로 판정한다. [실행 식별과 API 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/first-launch-verified.json)

두 부하의 운영 기준을 모두 통과하면 같은 이미지에서 W4의 고정 장문 입력 29개를 대조하도록 등록했다. 기존 입력의 144건과 홀드아웃의 88건을 같은 체크포인트로 비교하고, 실제 실행 파일과 이미지의 해시를 W5 영수증에 연결한다. CPU 사전시험에서 한 조건만 실패하면 GPU 실행을 건너뛰고 영수증 해시가 다르면 오류로 끝나는 것을 확인했다. 재현할 때는 기존 [이미지 판정 검사기](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/verify_a100_candidate.py)를 같은 실행 경로에 놓는다. [조건부 정확성 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/accuracy-follow-up/registration.json) · [실행과 판정 조건](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/accuracy-follow-up/campaign.sh)

두 모드의 30분 부하를 완료했다. 각각 기준과 동일한 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했고 API 40건·요청·지연·종료 자원 검사를 통과했다. 정밀 RSS와 익명 RSS의 증가량은 각 실행에서 같았다. Riley PID 1의 약 900초와 1,804초 표본을 직접 대조했다. [두 모드의 검산과 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/paired-verified.json) · [검산 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/verify_framing_soak.py)

| 모드 | 기준 정밀 RSS 증가 | 후보 정밀 RSS 증가 | 후보 후반 TTFT p95 | 후보 후반 완료 지연 p95 |
|---|---:|---:|---:|---:|
| `cache-on` | +616KiB | **+632KiB** | 328.24ms | 4,141.07ms |
| `cache-off` | +700KiB | **+580KiB** | 334.21ms | 4,160.60ms |

두 모드 모두 RSS 증가 0 항목에서만 실패해 프레이밍 후보를 채택하지 않았다. 각각 종료 뒤 기존 GPU 6 홀더의 20GiB 예약을 확인했다. 조건부 장문 정확성 시험은 양쪽 결과를 확인한 뒤 실행을 건너뛰었다. [첫 부하 별도 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/a100-follow-up/cache-on-verified.json) · [정확성 판정 코드](../../benchmarks/evidence/20261119-w5-mixed-soak/http-framing-capacity/accuracy-follow-up/accuracy_gate.py)

### RSS 표본과 실행 중 자원 지표 대조

기준 이미지와 glibc 대조군, HTTP 프레이밍 두 모드를 포함한 A100 6번의 완료 실행 12건에서 부하 시작 전·60초 간격 표본 30개·drain의 메트릭을 집계했다. 각 표본의 CUDA device 할당은 291건·30,495,658,464바이트, pinned host 할당은 2건·4,297,400바이트로 같았다. 약 900초의 메트릭에는 활성 요청이 10개였고 drain에서는 활성·대기 요청과 활성 KV가 0이었다. 두 메트릭 표본은 각각 해당 `smaps_rollup` 읽기 직전에 수집됐다. [12개 실행의 집계와 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/sample-comparison.json) · [원본 대조 도구](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/compare_soak_samples.py)

기본 이미지·스레드별 할당 캐시 비활성화 단독군·HTTP 프레이밍 두 모드의 일반 `VmRSS`는 각 실행의 10–15분과 20–25분 표본에서 같은 값을 유지했다. 같은 실행들의 정밀 RSS는 후반 기준에서 drain까지 148–700KiB 증가했다. 호스트 힙의 증가 추세를 판별하려면 추가 정밀 표본과 동일한 요청 처리 상태에서 수집한 대조가 필요하다. 현재 RSS 증가 0 판정은 기존 두 정밀 표본을 적용한다.

같은 기본 이미지에서 `cache-off` 30분 부하 중 5·10·20·25·29분과 drain 뒤 10·30·60초에 PID 1의 정밀 RSS를 추가로 읽는 진단을 등록했다. 원래 검사기가 선택한 약 900초 표본과 drain 표본으로 기존 RSS 증가 0 판정을 내린다. 진단은 요청 제출이 끝난 상태에서 60초를 더 관측한 뒤 유휴 상태를 한 번 확인한다. CPU 사전시험에서는 원래 RSS 실패 결과가 유지됐고, 같은 이미지의 UID 65532에서 실제 PID 1의 정밀 RSS도 읽었다. 프레이밍 두 실행의 판정과 동일 GPU 홀더의 예약 복구를 확인한 뒤 이 진단을 시작했다. [진단 등록과 CPU 검사](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/rss-series/registration.json) · [계측 코드](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/rss-series/observe_rss_series.py) · [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/rss-series/campaign.sh)

기준과 동일한 이미지·실행 파일·체크포인트에서 API 40건과 30분 부하를 완료했다. 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. 900.15초에서 1,804.00초까지 정밀 RSS와 익명 RSS가 각각 **480KiB 증가**했다. 기존 비계측 `cache-off`의 증가량은 700KiB였다. 후반 TTFT p95는 333.77ms, 완료 지연 p95는 4,157.68ms였다. 요청·지연·종료 자원 조건은 통과했고 RSS 증가 0 항목만 실패했다. [계측 결과와 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/rss-series/verified-series.json) · [독립 재검산](../../benchmarks/evidence/20261119-w5-mixed-soak/allocator-trace/rss-series/replay_rss_series.py)

| 정밀 표본 | 15분 대비 RSS 및 익명 RSS 변화 | 표본 직전 활성 요청 |
|---|---:|---:|
| 5분 | −2,360KiB | 5 |
| 10분 | −632KiB | 11 |
| 15분 | 0KiB | 10 |
| 20분 | +176KiB | 13 |
| 25분 | +368KiB | 9 |
| 29분 | +460KiB | 6 |
| Drain | +480KiB | 0 |
| Drain 뒤 10·30·60초 | +480KiB | — |

29분부터 drain까지의 추가 증가는 20KiB였고, drain 뒤 세 표본의 값은 같았다. 마지막 표본 뒤 서비스가 유휴 상태임을 확인했다. 표본 사이 변화와 개별 할당 지점은 이 실행의 계측 범위에 포함되지 않는다. 종료 뒤 A100 6·7번에서 각자의 기존 홀더와 20GiB 예약을 확인했다.

### 채팅 스트림의 임시 JSON 할당

`d3d8fe6e5`의 채팅 SSE는 토큰 이벤트를 텍스트 완성 SSE로 직렬화한 뒤 JSON을 다시 읽어 채팅 형태로 변환한다. 격리한 후보는 일반 토큰 이벤트의 구조체를 JSON 값으로 바로 전달해 이 중간 문자열과 재파싱을 줄인다. 같은 도구 호출 변환과 최종 사용량·오류·`[DONE]` 처리를 재사용했다. [현재 채팅 변환](../../crates/riley-server/src/chat.rs) · [격리 후보 패치](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/chat-delta.patch.gz)

초기 후보의 logprobs 점수 4,096개 가운데 562개에서 출력 바이트가 달랐다. JSON 부동소수점 왕복을 거친 기존 값과 직접 변환한 값의 마지막 자릿수가 달라진 사례를 확인했다. 최종 후보는 logprobs가 포함된 조각에 기존 직렬화 경로를 적용했다. 점수 4,096개 및 일반 텍스트·Hermes·Llama·불완전 도구 호출·토큰 ID·logprobs·오류의 합성 스트림 7종에서 기준과 출력 바이트가 일치했다. 별도로 빌드한 기준 커밋과 스트림 해시 및 기준 할당 건수도 일치했다. [CPU 원본 대조와 수치 예시](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/verified-cpu.json) · [재현 스크립트](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/run_cpu_probe.sh)

| 합성 스트림 64회 | 기준 할당·재할당 호출 | 후보 호출 | 기준 요청 바이트 | 후보 요청 바이트 |
|---|---:|---:|---:|---:|
| 일반 텍스트 | 25,984 | 21,696 | 2,801,344 | 1,725,504 |
| 토큰 ID | 21,120 | 17,664 | 54,154,304 | 26,760,320 |
| logprobs | 46,784 | 46,784 | 6,281,984 | 6,281,984 |

할당 요청 바이트에는 재할당 시 요청한 새 크기가 포함된다. macOS ARM64에서 계측기를 뺀 후보의 서버 테스트 101건과 포맷·Clippy 검사가 통과했다. 이 후보는 CPU 대조로 보관하며, A100의 실제 혼합 부하에서 지연과 정밀 RSS를 확인해야 채택 여부를 판단할 수 있다.

기본 이미지의 추가 계측이 완료되고 RSS 항목만 실패하면, 승인본에 이 패치만 적용해 별도 CUDA 12.4 이미지를 빌드하도록 등록했다. 빌드 입력에서 달라진 파일 세 개와 Riley 실행 파일을 제외한 패키지 파일의 바이트를 검증한다. A100 6번에서 두 캐시 모드의 API 40건과 동일한 30분 부하를 각각 실행하고 매번 같은 20GiB 예약을 복구한다. 재현 시 상위 디렉터리의 `chat-delta.patch.gz`를 실행 디렉터리에 놓고 `dispatch.sha256`을 검증한다. [A100 조건부 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/registration.json) · [실행 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/campaign.sh)

후보 이미지 `06eed36dcc60`의 Linux AMD64 빌드와 CPU 패키지 검사를 완료했다. 소스 항목 8,995개에서 `chat.rs`·`openai.rs`·`service.rs`의 변경을 확인했고, 패키지의 Riley 실행 파일 이외 20개 파일은 기준과 바이트가 일치했다. A100 6번의 `cache-on`에서 실행 중인 PID 1 바이너리 해시가 패키지 보고서와 일치했고 API 40건과 캐시 128블록을 확인했다. 혼합 부하를 시작한 시점에 GPU 7은 같은 홀더의 20GiB 예약을 유지했다. [패키지 검산](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/initial-package-verified.json) · [실제 실행 식별과 API 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/cache-on-launch-verified.json)

두 `cache-on`·`cache-off` 부하가 기존 운영 기준을 통과하면, 같은 이미지에서 기준 장문 144건과 홀드아웃 88건을 대조하도록 등록했다. 두 모드의 요청·RSS·지연·회수 영수증을 재검산해 실행 여부를 정한다. CPU 사전시험에서는 양쪽 통과, 한쪽 실패, 다른 바이너리와 달라진 GPU 7 예약의 처리 경로를 확인했다. [조건부 장문 검증 등록](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/accuracy-follow-up/registration.json) · [판정기](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/accuracy-follow-up/verify_chat_accuracy.py)

동일 이미지의 두 30분 혼합 부하를 끝냈다. 각 실행은 정상 완료 13,749건, 의도적 연결 종료 700건, API 40건을 확인했고 요청·지연·종료 자원 검사는 통과했다. 두 실행에서 약 900초와 1,804초의 RSS를 대조했다. 일반 `VmRSS` 변화는 0이었고, 정밀 RSS와 익명 RSS는 표의 값만큼 증가했다. 각각 종료 후 GPU 6이 기존 홀더의 20GiB 예약으로 복귀했고 GPU 7도 같은 예약을 유지했다. [두 실행의 독립 검산과 원본 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/paired-verified.json) · [cache-off 실행 식별](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/a100-follow-up/cache-off-launch-verified.json)

| 모드 | 기준 정밀 RSS 증가 | 후보 정밀 RSS 증가 | 후보 후반 TTFT p95 | 후보 후반 완료 지연 p95 |
|---|---:|---:|---:|---:|
| `cache-on` | +616KiB | **+508KiB** | 327.85ms | 4,143.58ms |
| `cache-off` | +700KiB | **+608KiB** | 334.80ms | 4,158.05ms |

두 실행 모두 RSS 증가 0 항목만 실패해 후보 채택을 보류했다. 조건부 장문 검증기는 두 영수증과 이미지 해시를 확인한 뒤 실행을 건너뛰었으며, 그 판정을 서버에서 독립적으로 재계산했다. [조건부 정확성 미실행 영수증](../../benchmarks/evidence/20261119-w5-mixed-soak/chat-delta-cpu/accuracy-follow-up/accuracy-skip-verified.json)

### 즉시 반환 설정의 할당기 대조

기준 이미지에서 glibc의 반환·스레드 캐시 대조가 두 캐시 모드의 RSS 기준을 함께 충족하지 못했다. 기존 `d3d8fe6e5` 이미지에 Ubuntu Jammy의 `libjemalloc2` 패키지 `5.2.1-4ubuntu1`을 읽기 전용으로 연결해 비교한다. 패키지는 APT 인덱스의 SHA-256과 대조해 검사 경로에 풀었다. 이미지, Riley 바이너리, 모델, 서빙 옵션과 부하 검사기는 기준 실행의 것을 사용한다. 컨테이너 시작 시에만 `LD_PRELOAD`와 `MALLOC_CONF`를 지정한다. dirty·muzzy 페이지의 반환 지연을 0으로 두고 별도 초대형 할당 arena 기준을 비활성화했다. [등록과 패키지 해시](../../benchmarks/evidence/20261120-w5-jemalloc/registration.json) · [jemalloc 설정 문서](https://jemalloc.net/jemalloc.3.html)

같은 이미지의 UID·GID 65532에서 16KiB 메모리 2,048개를 할당해 쓰고, 일부를 같은 간격으로 유지한 채 나머지를 해제했다. 해제 전후의 RSS는 glibc에서 34,376→32,472KiB, jemalloc에서 38,504→6,184KiB였다. 실제 Riley 바이너리의 `malloc`·`calloc`·`realloc`·`free`가 같은 jemalloc 파일에 연결됐고 CLI 버전 출력도 일치했다. 이 CPU 시험은 반환 설정과 로더 연결을 확인한다. 실제 추론의 RSS와 지연은 두 캐시 모드의 부하에서 판정한다. [사전시험 집계](../../benchmarks/evidence/20261120-w5-jemalloc/cpu-preflight.json) · [재현 스크립트](../../benchmarks/evidence/20261120-w5-jemalloc/preflight.sh)

A100 6번의 `cache-on`에서 추론 PID의 할당기 매핑과 API 40건을 확인했고 첫 30분 부하를 시작했다. 그 시점에 GPU 7은 같은 홀더의 20GiB 예약을 유지했다. 두 모드에서 준비 직후와 종료 직전의 할당기 설정을 대조하고, 같은 PID에서 읽은 RSS·후반 p95·종료 자원·GPU 예약 복귀를 검산한다. 재현 시 승인 이미지의 기존 검사 도구와 `verify_a100_candidate.py`를 등록 파일의 경로 및 해시에 맞춘다. [기동과 API 검산](../../benchmarks/evidence/20261120-w5-jemalloc/startup-verified.json) · [GPU 실행 순서](../../benchmarks/evidence/20261120-w5-jemalloc/campaign.sh) · [집계기](../../benchmarks/evidence/20261120-w5-jemalloc/verify_zerodecay.py)

두 모드의 30분 부하가 RSS 증가 0과 API 40건, 요청·지연·종료 자원 검사를 모두 통과했다. 각 모드에서 정상 완료 13,749건과 의도적 연결 종료 700건이 기준과 일치했다. 시험 뒤 같은 홀더의 GPU 6·7 예약 20GiB를 확인했다. 두 모드에서 정밀 RSS 변화와 익명 RSS 변화는 각각 같았다. [두 모드 독립 검산과 원본 해시](../../benchmarks/evidence/20261120-w5-jemalloc/paired-verified.json)

| 모드 | 기준 정밀 RSS 변화 | 후보 정밀 RSS 변화 | 후보 후반 TTFT p95 | 후보 후반 완료 지연 p95 |
|---|---:|---:|---:|---:|
| `cache-on` | +616KiB | **−680KiB** | 326.88ms (+0.33%) | 4,131.06ms (−0.01%) |
| `cache-off` | +700KiB | **−672KiB** | 334.37ms (+0.05%) | 4,162.46ms (+0.12%) |

첫 부하 도중 같은 호스트에서 W3 재생용 상대 시각을 집계했다. 집계 전송의 관측 구간은 UTC 15:24:18–15:24:25였으며, CPU 작업이 지연에 미친 영향은 별도로 측정하지 않았다. [W3 CPU 시각 집계](../../benchmarks/evidence/20261119-w3-rag-profile/qkv-1024/serving/traced-follow-up/command-clock-cpu/cpu-ready-inputs.json)

같은 이미지와 할당기의 장문 검증에서 기준 144건과 홀드아웃 88건의 출력이 모두 일치했다. 고정 입력·결과·실행 파일을 해시로 검산했고, API 40건과 종료 자원 회수를 확인했다. 시험 종료 뒤 GPU 6·7은 기존 홀더의 20GiB 예약으로 복귀했다. [장문 정확성 영수증](../../benchmarks/evidence/20261120-w5-jemalloc/accuracy-follow-up/verified-outcome.json)

별도 시험 이미지 `1fc444b1840e`에 할당기 파일과 패키지 고지문, `serve` 실행 스크립트를 넣었다. CPU 검사는 Riley 실행 파일을 포함한 기준 파일 21개의 해시, 추가 파일 3개의 해시, CLI 여섯 경로의 출력, Riley의 할당 심볼 네 개를 검증했다. 첫 검사기는 로더가 남긴 `riley` 이름을 절대 경로로만 찾았다. 실행 경로를 함께 대조하도록 수정해 같은 이미지 ID로 재검사했고, 로더 기록의 해시도 일치했다. [패키징 등록](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/registration.json) · [재검사 이력과 CPU 결과](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/recheck/registration.json)

이 파생 이미지 ID로 GPU 6에서 `cache-on` API 40건을 통과하고 첫 30분 부하를 시작했다. 실제 Riley PID가 이미지 내부 할당기를 적재했고 컨테이너 마운트는 체크포인트 경로 하나였다. GPU 7의 20GiB 예약도 확인했다. 두 캐시 모드의 부하에 모두 통과하면 같은 이미지로 장문 232건을 재검증한다. [패키징된 이미지의 GPU 검증 등록](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/registration.json) · [실제 기동 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/cache-on-launch-verified.json)

파생 이미지 `1fc444b1840e`의 두 30분 부하에서 각 모드의 API 40건, 정상 완료 13,749건, 의도적 연결 종료 700건과 지연 기준을 확인했다. 각 실행에서 정밀 RSS와 익명 RSS의 변화량이 같았다. 종료 시 활성·대기 요청, 할당된 KV 블록, GPU·pinned 메모리는 모두 0이었다. GPU 6번은 두 실행 뒤 모두 같은 홀더의 20GiB 예약으로 복귀했고 GPU 7번은 기존 예약을 유지했다. [패키징 이미지의 두 부하 독립 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/verified-soaks.json)

| 모드 | 후반 정밀 RSS 변화 | 전반 대비 후반 TTFT p95 | 전반 대비 후반 완료 지연 p95 |
|---|---:|---:|---:|
| `cache-on` | **−440KiB** | +1.63% | −13.56% |
| `cache-off` | **−640KiB** | +0.61% | −13.35% |

두 모드의 통과 뒤 같은 이미지에서 장문 출력 232건이 기준과 일치했다. API 40건과 종료 자원 회수를 확인했고 GPU 6·7은 동일 홀더의 20GiB 예약 상태로 복귀했다. [장문 출력과 실행 파일 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/gpu-qualification/accuracy-verified.json)

#### 같은 GPU에서 비교한 서빙 성능

Qwen2.5-7B-Instruct를 A100 6번에서 기준 이미지 `0a907408ec73` → 파생 이미지 `1fc444b1840e` → 파생 → 기준 순으로 실행했다. 네 실행의 체크포인트와 서빙 인자가 같았다. 실행마다 API 40건과 성능 요청 640건을 통과했고, 요청별 출력 해시·입력과 생성 토큰 수·종료 사유가 네 실행에서 일치했다. 종료 뒤 같은 GPU 홀더의 20GiB 예약을 확인했다. [성능 시험 등록](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/registration.json) · [네 실행의 독립 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/verified-performance.json)

아래 수치는 기준과 후보의 두 실행 평균을 대조한 결과다. 짧은 C1·C32의 TTFT p95·처리량 2%와 여섯 부하의 decode TPOT 1% 회귀 기준을 통과했다.

| 부하 | TTFT p95 | TPOT p50 | 출력 처리량 |
|---|---:|---:|---:|
| 짧은 C1 | +0.46% | −0.02% | +0.01% |
| 짧은 C32 | −8.67% | −0.19% | +0.50% |
| RAG C8 seed 1 | −3.23% | +0.60% | +0.62% |
| RAG C8 seed 2 | +12.69% | −1.94% | −1.77% |
| 긴 생성 C32 | +0.68% | −0.10% | +0.08% |
| prefix 공유 C8 | −0.10% | +0.01% | −0.005% |

짧은 C32의 TTFT p95 변화는 첫 실행 쌍에서 +14.45%, 두 번째에서 −24.26%였다. 기준 이미지의 두 p95는 58.57ms와 86.84ms였고 후보는 67.03ms와 65.77ms였다. 평균은 사전 기준을 충족했으며 두 실행 쌍에서 개선 방향은 일치하지 않았다. RAG C8 seed 2는 두 순서에서 TTFT p95가 각각 7.43%와 18.19% 증가했다. [C32 지연 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/ttft-audit-closed-c32.json) · [RAG seed 1 지연 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/ttft-audit-rag-c8-seed1.json) · [RAG seed 2 지연 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/ttft-audit-rag-c8-seed2.json)

RAG C8 seed 2에서 요청 시작부터 클라이언트의 소켓 쓰기까지의 p95 변화는 +0.18ms와 +0.01ms였다. 쓰기 뒤 첫 텍스트까지의 p95는 +66.12ms와 +154.29ms였다. 이 구간에는 소켓 전달, 서버 대기·처리, GPU 계산, SSE 전달이 포함된다. 후속 대조에서는 같은 쓰기 시각표를 두 이미지에 적용했다.

고정 시각 재생의 CPU 사전 검사에서 첫 기준·후보 기록의 요청 본문 해시를 각각 64건씩 재현했다. 두 소켓 쓰기 시각의 간격은 첫 측정 요청부터 마지막까지 19.137초와 19.354초였다. 두 이미지에 각 시각표를 적용하고 실제 쓰기 시각의 오차도 기록한다. 이 검사는 기존 닫힌 부하의 응답 속도에 따라 다음 요청 시각이 달라지는 효과를 조사하기 위한 후속 대조다. [고정 입력 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-replay/preflight.json) · [재생 클라이언트](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-replay/fixed_arrival.py)

GPU 7의 W4 진단이 끝난 뒤 GPU 6에서 기준 → 후보 → 후보 → 기준 순으로 두 시각표를 각각 재생하는 검증을 시작했다. 첫 실행 쌍에서는 기준에서 얻은 시각표를 먼저, 두 번째 쌍에서는 후보에서 얻은 시각표를 먼저 사용한다. 실행마다 API 40건과 시각표별 64건을 확인한다. 출력·쓰기 시각 오차·TTFT p95를 함께 기록하며 실제 쓰기 오차 p95 5ms를 재생 정밀도의 기준으로 사용한다. [고정 시각 검증 등록](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-replay/registration.json)

네 실행에서 각각 API 40건을 통과했고, 측정 요청 512건의 출력 해시·토큰 수·종료 사유가 기존 기록과 일치했다. 설정한 시각과 실제 소켓 쓰기의 절대 오차 p95는 모든 실행에서 1.81ms 이하였다. 대응하는 요청끼리 두 이미지의 실제 상대 쓰기 시각을 비교한 절대 차이 p95는 1.24ms 이하였다. 두 GPU는 기존 홀더의 20GiB 예약으로 복귀했다. [고정 시각 재생 결과](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-replay/verified-replay.json)

| 고정 시각표의 출처 | 기준 이미지 TTFT p95 평균 | 후보 이미지 TTFT p95 평균 | 후보 변화 |
|---|---:|---:|---:|
| 첫 기준 실행 | 728.60ms | 736.46ms | +1.08% |
| 첫 후보 실행 | 954.45ms | 932.56ms | −2.29% |

같은 이미지에서 첫 기준의 시각표를 첫 후보의 시각표로 바꾸면 TTFT p95 평균이 기준 이미지에서 31.00%, 후보 이미지에서 26.63% 늘었다. 두 시각표 모두에서 기존 닫힌 부하의 +12.69% 증가가 재현되지 않았다. 이 결과는 고정된 RAG C8 seed 2와 기록된 두 시각표에 적용한다. 기존 닫힌 부하의 지연 비용과 실제 운영 도착 분포를 릴리스 판정에 남긴다. 파생 설정의 기본값 적용은 보류한다.

#### 닫힌 RAG 세 seed의 확장 교차 대조

A100 6번에서 기준 이미지 `0a907408ec73`과 jemalloc 파생 이미지 `1fc444b1840e`를 기준 → 후보 → 후보 → 기준 순으로 실행했다. 두 이미지의 Riley 실행 파일, Qwen2.5-7B-Instruct 체크포인트와 서버 인자는 앞선 성능 비교와 같다. 닫힌 RAG C8의 프롬프트 길이와 최대 생성 128토큰을 유지했다. 기존 seed 1·2에 seed 3을 추가했으며 seed마다 워밍업 8건과 측정 요청 256건을 실행했다. 첫 쌍의 seed 순서는 1→2→3, 두 번째 쌍은 3→2→1이다. [실행 순서](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-confirmation/campaign.sh) · [부하 도구](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-confirmation/performance_client.py)

네 실행의 API는 각각 40건을 통과했다. 측정 요청 3,072건은 모두 성공했고 세 seed의 대응 요청이 네 실행에서 입력·출력 해시, 토큰 수, 종료 사유까지 일치했다. 다음 변화율은 각 순서에서 기준 대비 후보를 비교한 값이다. 평균 열은 두 기준값의 평균과 두 후보값의 평균을 비교했다. [검산 결과와 원본 해시](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-confirmation/verified-confirmation.json)

| seed | TTFT p95, 기준 먼저 | TTFT p95, 후보 먼저 | TTFT p95, 평균 기준 | TPOT p50, 평균 기준 |
|---|---:|---:|---:|---:|
| 1 | −1.81% | +3.18% | +0.65% | −0.63% |
| 2 | +9.14% | +7.27% | +8.30% | −0.69% |
| 3 | −17.56% | −20.61% | −19.06% | +0.90% |

64건에서 관측한 seed 2의 TTFT p95 증가 방향이 256건에서도 두 번 재현됐다. Seed 1은 순서에 따라 방향이 바뀌었고, seed 3은 두 순서에서 감소했다. 세 seed의 평균 비교는 decode TPOT 1% 회귀 한도를 충족했다. Seed 2에서 소켓 쓰기 전 구간의 p95 변화는 +0.107ms와 −0.010ms였고, 쓰기부터 첫 콘텐츠까지의 p95 변화는 +84.66ms와 +54.95ms였다. 쓰기 이후에는 서버 대기·GPU 처리·SSE 전달이 함께 포함된다. 현행 닫힌 부하에서 seed 2의 비용이 반복됐고, 실제 운영 도착 분포와 승인된 SLO를 확인하기 전까지 기본값 채택과 릴리스를 보류한다. [요청 타이밍 검산기](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-confirmation/verify_confirmation.py)

네 서버는 정상 종료했고 GPU 6·7은 각각 동일 홀더의 20GiB 예약 상태였다. 재현 시 기존 `audit_allocator_ttft.py`를 전송 파일에 포함하고 `dispatch.sha256`과 재사용 원본의 해시를 검증한다. [공유 검산 코드](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/audit_allocator_ttft.py) · [집계 재현](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/performance/rag-confirmation/fetch-summary.sh)

## 캐시 재사용·퇴거와 큐 포화 후 회복

같은 Qwen2.5-7B 바이너리를 A100 물리 GPU 6에서 cache 128블록, 활성 요청 2개, 대기 요청 1개로 실행했다. 토큰 재사용은 응답의 `usage.prompt_tokens_details.cached_tokens`에서 관측했다. [운영 검사기](../../benchmarks/live/operational_check.py) · [실행 인자와 해시](../../benchmarks/evidence/20261119-w5-operational/rerun/receipt.json)

| 순서 | 요청 | 입력 토큰 | 재사용 토큰 |
|---|---|---:|---:|
| 1 | 프롬프트 A 최초 요청 | 1,403 | 0 |
| 2 | A 반복 | 1,403 | 1,392 |
| 3 | 캐시의 2,048토큰 용량을 넘는 프롬프트 B | 2,798 | 16 |
| 4 | B 이후 A | 1,403 | 16 |
| 5 | A 재반복 | 1,403 | 1,392 |

A를 처음 보낸 뒤 두 요청에 공통인 16토큰 블록을 B가 재사용했다. B 이후 A에서 같은 수만큼만 재사용됐고, 다음 A에서는 1,392토큰 재사용이 복구됐다. 최초 시험은 4단계에서 0을 요구해 실패했다. 공통 블록 재사용을 반영해 검사기 판정을 교정하고 동일 조건으로 재실행했다. A 응답의 출력 바이트 해시와 종료 사유는 모든 단계에서 같았다. [최초 결과](../../benchmarks/evidence/20261119-w5-operational/initial/operational.json) · [재실행 결과](../../benchmarks/evidence/20261119-w5-operational/rerun/operational.json)

16개 스트리밍 요청을 동시에 보냈을 때 3개가 정상 완료되고 13개가 HTTP 429를 반환했다. 과부하 카운터도 13 증가했다. 큐가 비워진 뒤 단건 추론이 정상 완료됐고 활성·대기 요청과 요청 KV가 0이었다. 서버는 종료 코드 0과 native 할당 0을 남겼으며 GPU 6의 동일 홀더는 20GiB 예약으로 돌아왔다. [종료 메트릭](../../benchmarks/evidence/20261119-w5-operational/rerun/shutdown-metrics.json) · [GPU 6 예약 복귀](../../benchmarks/evidence/20261119-w5-operational/rerun/idle-after.json)

## 느린 스트리밍 소비자와 연결 취소

A100 GPU 6에서 같은 Qwen2.5-7B-Instruct CUDA 바이너리로 Linux loopback SSE 연결의 읽기를 중단했다. 수신 버퍼는 8,192바이트였고 서버 TCP 송신 큐가 16,250·39,189·63,042바이트로 세 번 연속 그 크기를 넘었다. 그 동안 별도의 정상 스트림이 약 222ms에 완료됐으며, 그 시점에도 느린 연결에 송신 큐 95,398바이트가 남아 있었다. 읽기를 재개한 연결은 SSE 종료와 토큰 usage를 받고 약 27.53초에 완료했다. [느린 스트림 원본](../../benchmarks/evidence/20261119-w5-slow-stream/result/slow-stream.json)

두 번째 연결의 송신 큐가 차고 활성 요청이 하나인 상태에서 연결을 닫았다. `disconnects`와 `cancellations`는 각각 1 증가했다. 활성·대기 요청과 요청 KV는 0으로 돌아갔고, 서버 실행 중 native 할당은 시작 전 값으로 회복됐다. 종료 코드는 0이었고 종료 시 device·pinned 할당은 0이었다. GPU 6은 동일 홀더의 20GiB 예약으로 복귀했다. [실행 기록](../../benchmarks/evidence/20261119-w5-slow-stream/result/receipt.json) · [종료 메트릭](../../benchmarks/evidence/20261119-w5-slow-stream/result/shutdown-metrics.json) · [예약 복귀](../../benchmarks/evidence/20261119-w5-slow-stream/result/idle-after.json)

장치 메모리 갱신 실패를 가정한 CPU 단위 테스트는 tentative KV 블록 회수와 sidecar 무효화를 확인했다. 배포 이미지의 A100 추론과 혼합 부하는 같은 문서에 기록했다. 이 절의 느린 소비자 TCP 송신 큐 대조는 네이티브 바이너리에서 실행했다. 실제 운영 트래픽 분포와 물리 장치 장애의 복구 검증이 남아 있다. [장치 실패 시 KV 회수 테스트](../../crates/riley-runtime/src/paged_kv.rs#L2186) · [배포 이미지의 혼합 부하](#배포-이미지의-혼합-부하)

기존 테스트는 할당 생성·해제의 결과가 불명확한 경우와 복사 제출·완료 오류를 독립 자식 프로세스로 검사한다. 첫 시도는 C ABI 헤더에서 cuBLAS probe 전용 타입의 기능 조건이 누락돼 종료 코드 101을 기록했다. 헤더 수정 이후 실제 빌드는 Rust 바인딩의 기능 조건 여섯 곳에서 종료 코드 101을 기록했다. 두 수정 사항을 적용한 재시험은 정밀 RSS 대조가 끝난 뒤 GPU 6에서 실행했다. [실패 주입 테스트](../../crates/riley-cuda/tests/memory_fault_injection_gpu.rs) · [C ABI 실패 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/initial-build/SHA256SUMS) · [Rust 실패 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/rust-build/SHA256SUMS) · [재시험 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/campaign.sh)

CUDA 12.4의 SM80 빌드에서 소스·패치 입력 체크섬 144개를 확인한 뒤 메모리 오류 주입 테스트를 실행했다. 생성 rollback, 명시적 해제, 지연 제출, 완료 복구의 네 사례가 서로 다른 자식 프로세스에서 종료 코드 0을 반환했다. 로그 검사기가 자식 네 건과 부모 한 건의 통과, 순차 격리, 총 16개 단계 표지를 검증했다. 물리 GPU 6은 같은 홀더의 20GiB 예약으로 돌아갔다. [실행과 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/SHA256SUMS) · [오류 주입 판정](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/verification.json)

## A100 격리 이미지 복구 시험

A100 6번의 한 loopback 주소에서 파생 이미지 `1fc444b1840e`를 실행한 뒤 기준 이미지 `0a907408ec73`로 교체했다. 두 실행에 같은 Riley 실행 파일, Qwen2.5-7B-Instruct 체크포인트와 prefix cache 128블록을 사용했다. 새 요청 송신을 끝낸 뒤 활성·대기 요청과 요청 KV가 모두 0인 것을 확인했다. 파생 이미지 종료 요청 뒤 포트 연결이 거부되는 것을 확인하고, 같은 포트에서 기준 이미지의 `/readyz`가 성공했다. 종료 요청부터 기준 이미지 준비 상태까지 136.37초가 걸렸다. 이 값은 단일 격리 시험의 복구 시간이다. [실행 순서](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/recovery-drill/campaign.sh) · [이미지·포트·시각 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/recovery-drill/verified-summary.json)

두 이미지가 각각 API 40건을 통과했다. 같은 접두사를 다시 보냈을 때 각각 112토큰을 재사용했다. 같은 스트리밍 요청의 텍스트 해시, 종료 사유와 입력·생성 토큰 수가 일치했다. 두 컨테이너는 코드 0으로 종료했고 device·pinned 할당과 KV가 0이었다. 각 실행 뒤 GPU 6·7의 동일 홀더와 20GiB 예약을 확인했다. 실제 운영 라우팅과 서비스 SLO는 별도 검증 범위에 남아 있다. [재현 검사기](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/recovery-drill/route_drill.py) · [스트리밍·캐시 검사기](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/recovery-drill/drill_client.py)

## 초당 1회 메트릭 조회의 비용

A100 6번의 파생 이미지 `1fc444b1840e`에서 하나의 Qwen2.5-7B-Instruct 프로세스를 유지했다. API 40건을 먼저 통과한 뒤 초당 평균 8건의 같은 90초 합성 혼합 부하를 기준 → 추가 조회 → 추가 조회 → 기준 순서로 실행했다. 기준에도 부하 검사기의 기본 메트릭 조회가 있었고 두 비교 구간에는 `/metrics`를 초당 한 번씩 추가했다. 각 구간에서 일반·스키마·도구·logprobs·장문 등 687건이 정상 완료됐고 30건을 의도적으로 연결 종료했다. 예상 외 오류는 0건이었다. 추가 조회는 구간당 95건씩 성공했고 응답 시간 중앙값은 7.47–7.78ms였다. [실행 순서](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/campaign.sh) · [관측값과 체크섬](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/verified-summary.json)

| 지표 | 기준 먼저 | 추가 조회 먼저 |
|---|---:|---:|
| 프로세스 CPU 초 / 경과 초 변화 | +0.01% | +0.02% |
| GPU 스텝 실행 시간 합 변화 | +0.40% | −0.01% |
| 후반 TTFT p95 변화 | +0.35% | +1.31% |
| 후반 전체 완료 시간 p95 변화 | +0.12% | +0.09% |

정밀 RSS의 구간 시작 대비 drain 변화는 실행 순서대로 +34,784KiB, −16KiB, +60KiB, +88KiB였다. 가장 큰 변화가 첫 기준 구간에 있어 이 대조에서 추가 조회와 구간 순서의 RSS 영향을 분리하기 어렵다. 정밀 RSS를 구간 시작 전과 drain 뒤 동일 프로세스에서 읽었고, CPU 델타는 `/proc`의 user·system ticks로 계산했다. 종료 시 요청·KV·device·pinned 할당은 0이었고, GPU 6·7의 기존 20GiB 예약을 확인했다. 측정 범위는 이 이미지와 1Hz 추가 조회의 단일 교차 실행이다. [부하·관측 코드](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/observer_client.py) · [이미지와 결과 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/verify_observer.py)

### 워밍업 후 역순 교차 대조

같은 이미지와 실행 인자에서 API 40건을 확인한 뒤, 추가 조회 없이 같은 90초 혼합 부하로 워밍업했다. 워밍업에서 정밀 RSS는 34,780KiB, 익명 RSS는 28,636KiB 늘었다. 앞선 실행에서 첫 기준 구간의 정밀 RSS 증가량은 34,784KiB였다. 이어 한 프로세스에서 추가 조회 → 기준 → 기준 → 추가 조회 순서로 측정했다. 워밍업과 각 측정 구간에서 687건이 정상 완료됐고 30건을 의도적으로 연결 종료했다. 추가 조회는 두 구간에서 각각 95건 성공했다. [워밍업 포함 실행 순서](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/warmed-replay/campaign.sh) · [관측값과 체크섬](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/warmed-replay/verified-summary.json)

| 추가 조회 구간과 기준 구간의 차이 | 추가 조회 먼저 | 기준 먼저 |
|---|---:|---:|
| 프로세스 CPU 초 / 경과 초 | +0.046% | +0.022% |
| GPU 스텝 실행 시간 합 | −0.008% | −0.006% |
| 후반 TTFT p95 | +1.62% | +0.56% |
| 후반 전체 완료 시간 p95 | +0.07% | +0.06% |
| 정밀 RSS 변화량 차이 | −280KiB | +260KiB |

워밍업 뒤 각 구간의 정밀 RSS 변화량은 실행 순서대로 −88KiB, +192KiB, −108KiB, +152KiB였다. 큰 RSS 증가는 두 실행 모두 추가 조회가 없는 첫 혼합 부하에 나타났다. 워밍업 뒤 추가 조회의 RSS 차이는 순서에 따라 부호가 달랐다. 결과는 같은 Qwen2.5-7B-Instruct 합성 부하와 1Hz 추가 조회에서 관측한 값이다. 종료 영수증과 두 GPU 홀더의 20GiB 예약 복귀를 확인했다. [워밍업과 4개 구간 검산](../../benchmarks/evidence/20261120-w5-jemalloc/image-follow-up/metrics-observer/warmed-replay/verify_warmed.py)
