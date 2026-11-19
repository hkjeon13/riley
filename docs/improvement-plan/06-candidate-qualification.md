# Riley 0.1.0 후보 검증 근거와 릴리스 판정

2026-11-19에 `main` `68788bee9`의 배포 입력을 확인했다. 현재 판정은 **릴리스 보류**다. A100에서 확인한 네이티브 실행 파일과 배포 이미지의 실행 파일은 각자의 SHA-256으로 추적한다. [후보별 원본 목록과 해시](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

## 실행 파일과 소스 연결

| 실행 대상 | 소스 리비전 | 확인한 범위 |
|---|---|---|
| A100 네이티브 빌드 | `f7c956d3c` | CUDA 12.4, SM80. API, 장문 정확성, Qwen2.5-14B 성능, Qwen2.5-7B 운영 검증 |
| Linux AMD64 배포 이미지 | `d3d8fe6e5` | CUDA 12.4, SM80·SM86·SM89. 이미지 빌드, 비루트 사용자, CLI, 동적 링크, 런타임 파일 무결성 |
| A100에서 재빌드한 배포 이미지 | `d3d8fe6e5` | 이미지 ID `0a907408ec73`. A100 6번의 CUDA 드라이버 주입, Qwen2.5-7B-Instruct API 40건, 장문 출력 대조 232건, GPU 자원 회수 |

초기 감사에서 `f7c956d3c` 네이티브 빌드와 `96c04dbe4` 배포 이미지의 프로그램 입력 Git 객체를 비교했다. 그 뒤 헤더·Rust 바인딩·서버 런타임을 보완해 배포 이미지의 Riley 실행 파일 해시가 갱신됐다. 네이티브 시험은 각 실행 파일로 계속 추적한다. [초기 네이티브 빌드](04-structured-output-result.md) · [이미지 빌드 이력](01-validation-result.md#현행-배포-이미지)

현행 배포 검증 이미지 `d3d8fe6e5`의 이미지 ID·Riley 실행 파일 해시를 보존했다. 이 이미지와 감사 시점 `68788bee9`의 배포 입력 Git 객체 14개가 모두 일치한다. 동일 Rust·CUDA 프로그램 입력인 `77c367003`에서는 CUDA 12.8 네이티브 빌드·C ABI·Clippy도 통과했다. [CUDA 12.4 이미지 원본](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/receipt.json) · [CUDA 12.8 네이티브 CI](https://github.com/hkjeon13/riley/actions/runs/44410425140)


## 현재 판정

| 조건 | 확인된 근거 | 남은 확인 |
|---|---|---|
| W1 Linux·과거 계약 | hosted Linux CPU·CUDA 검사와 고정 Python의 RC3 격리 테스트 34건 통과. A100에서 재빌드한 이미지의 NVIDIA 드라이버 주입과 실제 모델 API 40건 확인 | W5의 최종 운영 기준 충족 |
| W2·W3 성능 | 이벤트 발행·스케줄 네 조합, 14B 두 모델의 동일 KV·커널 계측, Qwen2.5-14B의 prefill 대기와 vLLM 한도 대조, Qwen3-4B 길이별 대조. 스케줄 후보 세 개는 지연 회귀로 미채택. Qwen2.5-14B의 1,024행 QKV 후보는 네 실행의 출력과 장문 토큰 ID가 일치했고 decode·짧은 요청 기준을 통과했다. RAG 첫 토큰 p95 개선이 일관되지 않아 기본값 적용을 보류했다. | 채택 후보의 RAG p95 재현성과 영향 모델의 출력·성능 기준 확인 |
| W4 장문·정확성 | 고정 입력 18개와 holdout 11개의 엔진 비교, 두 사례의 첫 BF16 차이 및 28개 레이어 residual·최종 norm 재계산, 네 모델의 동일 입력 GSM8K 품질 기준 통과. A100 재빌드 이미지의 장문 대조 232건 일치. 두 사례의 첫 레이어 KV와 attention 출력을 대조했고 마지막 입력 위치의 K·V 차이를 bias·반올림 순서로 재현. HF에서 bias 순서와 prefill 분할을 교차한 네 조건에서 사례 3의 첫 ID는 모두 5257. 같은 BF16 입력의 MLP 출력 대조도 기록 | 다른 위치와 레이어 및 해당 차이가 최종 점수까지 전파되는 정도 |
| W4 스키마 | 문서화한 부분집합에서 GPU 일반 API 40건과 schema 계약 11건을 통과했다. A100 재빌드 이미지의 대표 모델 API 40건도 통과했다. | W5의 최종 운영 기준 충족 |
| W5 운영 | 네이티브와 배포 이미지에서 각각 캐시 off·on 30분 부하와 종료 회수 확인. 배포 이미지의 익명 RSS는 cache-on +616KiB, cache-off +700KiB. 별도 진단에서 증가한 익명 private 매핑 32개가 glibc 반환 호출 뒤 감소했다. 느린 SSE·취소와 GPU 메모리 오류 주입 네 사례 확인 | 할당 지점을 좁히고 두 캐시 설정에서 RSS 증가 0 조건을 충족한다. 실제 운영 분포 확인 |

기존 이벤트 발행 정책과 서빙 설정을 기준선으로 삼았다. 이전 빌드에서 관찰된 성능 실험은 입력·장비·바이너리에 연결하며, 새로운 수치 기준이나 기본값을 채택할 때는 종합 계획의 출력 불변성과 회귀 기준을 적용한다. [W2 결과](02-async-step-result.md) · [W3 결과](03-long-context-result.md) · [W4 정확성](04-long-accuracy-result.md) · [W5 운영 결과](05-mixed-soak-result.md)

W3의 QKV 추적 416건은 해당 비계측 요청·응답과 일치했다. 50ms 간격 부하의 동일 장문 요청에서 늘어난 첫 토큰 지연 528–541ms 중 381–392ms는 prefill 조각 사이 대기였다. RAG 첫 토큰 p95 개선이 일관되게 재현될 때까지 후보는 격리 시험에만 둔다. [두 순서의 단계별 결과](03-long-context-result.md#50ms-간격-유입의-추가-지연-위치)

W5의 30분 실행에서 종료 뒤 GPU·pinned 메모리와 활성 KV는 0이었다. Cache-off 정밀 실행에서 일반 RSS의 변화는 0이었고 `smaps_rollup`은 익명 RSS +700KiB를 기록했다. 따라서 RSS 증가 0 조건은 미충족이다. 정밀 판정기를 적용한 다음 실행에서 +812KiB를 확인했고 매핑별 합계에서 전부 익명 private에 속했다. 요청·지연·종료 회수 검사는 통과했고 RSS 검사만 실패했다. GPU 메모리 오류 주입은 A100 6번의 CUDA 12.4 SM80 테스트 바이너리에서 네 독립 사례가 통과했다. [운영 품질 기준](../goal.md#33-운영-품질) · [정밀 RSS 결과](05-mixed-soak-result.md#정밀-rss-재실행-결과) · [오류 주입 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/verification.json)

같은 A100 바이너리의 후속 대조에서도 RSS가 기본값에서 +932KiB, `MALLOC_ARENA_MAX=2`에서 +3,648KiB 증가했다. 두 조건 모두 RSS 기준만 실패했고 arena 제한은 채택하지 않았다. [비교 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/result/paired-comparison.json)

초기 자동 승인 검토 뒤 `96c04dbe4`와 `d3d8fe6e5` 아카이브의 전송을 각각 승인받았다. `aws-a100`의 전용 경로에서 두 아카이브의 SHA-256이 원래 CI에 기록된 값과 일치함을 확인했다. GPU 7의 RSS 비교 이후 `d3d8fe6e5`를 빌드했다. AWS 이미지 ID는 `0a907408ec73`, GitHub CI 이미지 ID는 `8cb92e573ef1`이다. Riley 바이너리를 포함한 패키지 런타임 파일 21개의 SHA-256은 모두 같았다. AWS 이미지에서 GPU 6의 체크포인트 무결성, API 40건, 정상 종료와 20GiB 예약 복귀를 확인했다. [A100 스테이징 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-staged-sources.json) · [A100 패키지·GPU 증거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-package-validation.json) · [종합 완료 조건](README.md#8-종합-완료-체크리스트)

같은 이미지에서 장문 29개에 대한 232건을 확인했다. 초기 GPU 확인의 이미지 ID, CPU 보고서, Riley 바이너리 및 체크포인트 해시와 일치했다. Cache-on과 cache-off의 30분 혼합 부하는 각각 정상 완료 13,749건과 의도적 연결 종료 700건을 기록했다. 두 서버 모두 정상 종료하고 자원을 회수했다. 후반 익명 RSS는 cache-on에서 616KiB, cache-off에서 700KiB 증가해 두 설정 모두 RSS 증가 0 기준을 충족하지 못했다. [장문 대조](04-long-accuracy-result.md) · [배포 이미지의 혼합 부하](05-mixed-soak-result.md#배포-이미지의-혼합-부하)

주기적으로 glibc가 메모리를 반환하도록 한 진단군의 cache-off 30분 실행은 익명 RSS가 60KiB 증가해 증가 0 조건을 충족하지 못했다. 같은 이미지의 비계측 기준은 700KiB 증가했다. Cache-on에서는 기준 +616KiB에서 주기적 반환 적용 시 −108KiB로 바뀌어 모든 검사를 통과했다. 두 설정 모두 API 40건과 지연·종료 검사를 통과했다. Cache-off가 남아 릴리스 판정은 보류다. [주기적 반환 결과](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

31초 주기의 별도 cache-off 표본 진단에서도 익명 RSS가 500KiB 증가했다. 중간과 drain의 반환 이후 경과 시간을 확인했고, 증가 0 기준은 계속 미충족이다. [표본 시각과 매핑 대조](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

7초 주기의 첫 시도는 라이브러리 적재 실패로 후보 판정에서 제외했다. 같은 이미지에서 서비스 계정의 적재를 먼저 검증한 재실행에서 cache-off 익명 RSS는 44KiB, cache-on은 292KiB 증가했고 두 모드 모두 기존 RSS 증가 0 항목만 실패했다. Drain 표본은 각각 직전 반환 완료 후 0.19–0.59초와 4.70–5.05초에 읽혔다. 릴리스 보류 상태이며 동일한 7초 주기에 스레드별 할당 캐시 비활성화 대조를 등록했다. [재실행 결과](05-mixed-soak-result.md#배포-이미지의-주기적-메모리-반환-후보)

7초 주기 반환과 스레드별 할당 캐시 비활성화를 함께 적용한 동일 이미지에서 `cache-off` 정밀 익명 RSS는 144KiB 감소했고 `cache-on`은 88KiB 증가했다. 두 시험 모두 API 40건과 요청·지연·자원 회수 항목을 통과했다. 증가 0 기준을 `cache-on`에서 충족하지 못해 릴리스 판정은 보류다. [두 설정의 검산과 원본 해시](../../benchmarks/evidence/20261119-w5-mixed-soak/periodic-allocator-trim/tcache-control/paired-verified.json)

같은 배포 이미지에서 스레드별 할당 캐시 비활성화만으로 두 RSS 기준을 충족하는지 별도 대조를 등록했다. 이 시험은 기존 이미지 실행 옵션으로 재현하며 7초 주기 메모리 반환은 활성화하지 않는다. [대조 조건과 실행 순서](05-mixed-soak-result.md#스레드별-할당-캐시-설정-단독-대조)
