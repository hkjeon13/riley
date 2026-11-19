# W5-B A100 혼합 부하와 종료 검증

2026-11-19 KST에 `aws-a100`의 물리 GPU 7에서 Qwen2.5-7B-Instruct와 `native-bf16-varlen-v1`을 실행했다. 모델의 문맥 상한은 8,192, 출력 상한은 512토큰이다. CUDA 바이너리는 W4의 `f7c956d3c` 빌드와 SHA-256이 같다. [W5 계획](05-documentation-and-operations.md) · [W4 CUDA 빌드](04-structured-output-result.md) · [실행 인자와 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/smoke/receipt.json)

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

Docker가 보고한 호스트 PID와 컨테이너 ID를 각 표본 전후로 대조하고, 컨테이너 사용자로 `/proc/1/status`와 `/proc/1/smaps_rollup`을 읽는다. 기존 부하 검사기가 요청 처리, 취소, 지연 변동, RSS 증가 0을 판정한다. 정밀 표본 누락과 RSS 증가는 실패로 기록한다. 정상 종료와 자원 회수 검사가 끝나면 이미지 ID, 바이너리 해시, 혼합 부하 원본 및 정리 지표의 해시를 보존한다. CPU 테스트에서 프로세스 변경 검출과 RSS 실패 후 종료 경로를 확인했다. 실제 컨테이너 결과는 이미지 빌드 뒤 수집한다. [Linux 부하 검사기](../../benchmarks/live/soak.py) · [이미지 검사 방법](01-validation-result.md#배포-이미지-gpu-검사)

첫 cache-off 시도에서는 검사기가 `--prefix-cache-blocks 0`을 넘겼고 서버가 준비 단계에서 거부했다. 부하 측정은 시작되지 않았고 GPU는 동일 홀더의 20GiB 예약으로 복귀했다. 인자 생략과 128개 지정이 각각 유지되는 회귀 검사를 추가했다. 실행 중인 cache-on이 종료되면 같은 이미지와 체크포인트에서 수정한 cache-off 검사만 재실행하도록 등록했다. [첫 시도와 재실행 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json)

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

장치 메모리 갱신 실패를 가정한 CPU 단위 테스트는 tentative KV 블록 회수와 sidecar 무효화를 확인했다. 실제 운영 트래픽 분포, GPU 장치 장애, 배포 이미지 추론은 추가 검증 항목이다. [장치 실패 시 KV 회수 테스트](../../crates/riley-runtime/src/paged_kv.rs#L2186)

기존 테스트는 할당 생성·해제의 결과가 불명확한 경우와 복사 제출·완료 오류를 독립 자식 프로세스로 검사한다. 첫 시도는 C ABI 헤더에서 cuBLAS probe 전용 타입의 기능 조건이 누락돼 종료 코드 101을 기록했다. 헤더 수정 이후 실제 빌드는 Rust 바인딩의 기능 조건 여섯 곳에서 종료 코드 101을 기록했다. 두 수정 사항을 적용한 재시험은 정밀 RSS 대조가 끝난 뒤 GPU 6에서 실행했다. [실패 주입 테스트](../../crates/riley-cuda/tests/memory_fault_injection_gpu.rs) · [C ABI 실패 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/initial-build/SHA256SUMS) · [Rust 실패 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/rust-build/SHA256SUMS) · [재시험 순서](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/campaign.sh)

CUDA 12.4의 SM80 빌드에서 소스·패치 입력 체크섬 144개를 확인한 뒤 메모리 오류 주입 테스트를 실행했다. 생성 rollback, 명시적 해제, 지연 제출, 완료 복구의 네 사례가 서로 다른 자식 프로세스에서 종료 코드 0을 반환했다. 로그 검사기가 자식 네 건과 부모 한 건의 통과, 순차 격리, 총 16개 단계 표지를 검증했다. 물리 GPU 6은 같은 홀더의 20GiB 예약으로 돌아갔다. [실행과 체크섬](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/SHA256SUMS) · [오류 주입 판정](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/verification.json)
