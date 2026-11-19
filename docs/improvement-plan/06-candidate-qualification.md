# Riley 0.1.0 후보 검증 근거와 릴리스 판정

2026-11-19에 `main` `27e36810b`의 배포 입력을 확인했다. 현재 판정은 **릴리스 보류**다. A100에서 확인한 네이티브 실행 파일과 배포 이미지의 실행 파일은 각자의 SHA-256으로 추적한다. [후보별 원본 목록과 해시](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

## 실행 파일과 소스 연결

| 실행 대상 | 소스 리비전 | 확인한 범위 |
|---|---|---|
| A100 네이티브 빌드 | `f7c956d3c` | CUDA 12.4, SM80. API, 장문 정확성, Qwen2.5-14B 성능, Qwen2.5-7B 운영 검증 |
| Linux AMD64 배포 이미지 | `d3d8fe6e5` | CUDA 12.4, SM80·SM86·SM89. 이미지 빌드, 비루트 사용자, CLI, 동적 링크, 런타임 파일 무결성 |

초기 감사에서 `f7c956d3c` 네이티브 빌드와 `96c04dbe4` 배포 이미지의 프로그램 입력 Git 객체를 비교했다. 그 뒤 헤더·Rust 바인딩·서버 런타임을 보완해 배포 이미지의 Riley 실행 파일 해시가 갱신됐다. 네이티브 시험은 각 실행 파일로 계속 추적한다. [초기 네이티브 빌드](04-structured-output-result.md) · [이미지 빌드 이력](01-validation-result.md#현행-배포-이미지)

현행 배포 검증 이미지 `d3d8fe6e5`의 이미지 ID·Riley 실행 파일 해시를 보존했다. 이 이미지와 감사 시점 `27e36810b`의 배포 입력 Git 객체가 모두 일치한다. 동일 Rust·CUDA 프로그램 입력인 `77c367003`에서는 CUDA 12.8 네이티브 빌드·C ABI·Clippy도 통과했다. [CUDA 12.4 이미지 원본](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/receipt.json) · [CUDA 12.8 네이티브 CI](https://github.com/hkjeon13/riley/actions/runs/44410425140)


## 현재 판정

| 조건 | 확인된 근거 | 남은 확인 |
|---|---|---|
| W1 Linux·과거 계약 | hosted Linux CPU·CUDA 검사와 고정 Python의 RC3 격리 테스트 34건 통과 | 실제 배포 이미지의 NVIDIA 드라이버 주입과 추론 |
| W2·W3 성능 | 이벤트 발행·스케줄 네 조합, 두 14B 모델의 동일 KV·커널 계측, Qwen2.5-14B 요청별 prefill 대기와 vLLM 한도 대조, Qwen3-4B 길이별 대조. 진행 중인 prefill 우선 후보는 decode TPOT 회귀로 미채택. 완료 가능한 prefill 우선 후보는 두 번째 RAG seed의 TTFT와 두 동시 유입 조건의 decode 지연 증가로 미채택 | 채택할 후보의 재현성과 영향을 받는 모델의 출력·회귀 판정. 동시 유입 조건의 decode 지연 포함 |
| W4 장문·정확성 | 고정 입력 18개와 holdout 11개의 엔진 비교, 두 사례의 첫 BF16 차이 및 28개 레이어 residual·최종 norm 재계산, native layer 0의 KV·causal 재계산, 네 모델의 동일 입력 GSM8K에서 품질 기준 통과 | Riley/HF의 내부 Q/K/V 차이 원인과 최종 배포 후보 재확인 |
| W4 스키마 | 문서화한 부분집합에서 GPU 일반 API 40건과 schema 계약 11건 통과 | 배포 이미지의 대표 모델 API 재확인 |
| W5 운영 | 캐시 off·on 30분 부하·종료 회수, 느린 SSE와 취소, GPU 메모리 오류 주입 네 사례 | 재현된 익명 RSS 증가의 할당 원인과 증가 0 조건, 실제 운영 분포, 최종 배포 이미지의 동일 바이너리 검증 |

기존 이벤트 발행 정책과 서빙 설정을 기준선으로 삼았다. 이전 빌드에서 관찰된 성능 실험은 입력·장비·바이너리에 연결하며, 새로운 수치 기준이나 기본값을 채택할 때는 종합 계획의 출력 불변성과 회귀 기준을 적용한다. [W2 결과](02-async-step-result.md) · [W3 결과](03-long-context-result.md) · [W4 정확성](04-long-accuracy-result.md) · [W5 운영 결과](05-mixed-soak-result.md)

W5의 30분 실행에서 종료 뒤 GPU·pinned 메모리와 활성 KV는 0이었다. Cache-off 정밀 실행에서 일반 RSS의 변화는 0이었고 `smaps_rollup`은 익명 RSS +700KiB를 기록했다. 따라서 RSS 증가 0 조건은 미충족이다. 정밀 판정기를 적용한 다음 실행에서 +812KiB를 확인했고 매핑별 합계에서 전부 익명 private에 속했다. 요청·지연·종료 회수 검사는 통과했고 RSS 검사만 실패했다. GPU 메모리 오류 주입은 A100 6번의 CUDA 12.4 SM80 테스트 바이너리에서 네 독립 사례가 통과했다. [운영 품질 기준](../goal.md#33-운영-품질) · [정밀 RSS 결과](05-mixed-soak-result.md#정밀-rss-재실행-결과) · [오류 주입 원본](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/result/verification.json)

같은 A100 바이너리의 후속 대조에서도 RSS가 기본값에서 +932KiB, `MALLOC_ARENA_MAX=2`에서 +3,648KiB 증가했다. 두 조건 모두 RSS 기준만 실패했고 arena 제한은 채택하지 않았다. [비교 결과](../../benchmarks/evidence/20261119-w5-mixed-soak/glibc-arena-control/result/paired-comparison.json)

초기 자동 승인 검토 뒤 `96c04dbe4`와 `d3d8fe6e5` 아카이브의 전송을 각각 승인받았다. `aws-a100`의 전용 경로에서 두 아카이브의 SHA-256이 원래 CI에 기록된 값과 일치함을 확인했다. GPU 7의 RSS 비교 이후 `d3d8fe6e5`를 빌드하고 새 이미지 ID와 바이너리 해시를 고정해 GPU 6에서 대표 모델 API와 종료를 검증하는 작업을 시작했다. 실제 추론과 운영 근거가 확인되면 릴리스 판정을 갱신한다. [A100 스테이징 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/serving-lint-fix/a100-staged-sources.json) · [이미지 검증 범위](01-validation-result.md#현행-배포-이미지) · [종합 완료 조건](README.md#8-종합-완료-체크리스트)

같은 이미지에서 장문 29개와 cache-off·on 30분 혼합 부하를 확인하는 후속 작업도 등록했다. 두 작업은 초기 GPU API 확인 결과의 이미지 ID, CPU 보고서 및 체크포인트 해시를 대조한 뒤 GPU 6에서 순서대로 실행한다. [장문 대조](04-long-accuracy-result.md) · [배포 이미지의 혼합 부하](05-mixed-soak-result.md#배포-이미지의-혼합-부하)
