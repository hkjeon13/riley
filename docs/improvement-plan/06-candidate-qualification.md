# Riley 0.1.0 후보 검증 근거와 릴리스 판정

2026-11-19 기준 `main`의 `f1078c677` 소스에 대한 검증 판정은 **릴리스 보류**다. A100에서 확인한 네이티브 실행 파일과 배포 이미지의 실행 파일은 각자의 SHA-256으로 추적한다. [후보별 원본 목록과 해시](../../benchmarks/evidence/20261119-improvement-qualification/manifest.json)

## 실행 파일과 소스 연결

| 실행 대상 | 소스 리비전 | 확인한 범위 |
|---|---|---|
| A100 네이티브 빌드 | `f7c956d3c` | CUDA 12.4, SM80. API, 장문 정확성, Qwen2.5-14B 성능, Qwen2.5-7B 운영 검증 |
| Linux AMD64 배포 이미지 | `96c04dbe4` | CUDA 12.4, SM80·SM86·SM89. 이미지 빌드, 비루트 사용자, CLI, 동적 링크, 런타임 파일 무결성 |

두 소스와 감사 시점 `main`에서 Rust 소스, CUDA 커널, vendored 코드, Cargo 입력, Rust toolchain, 체크포인트 도구의 Git 객체가 같다. 배포 이미지의 entrypoint에는 `serve --help` 보완이 포함됐다. 두 바이너리는 같은 Cargo 기능으로 빌드됐고 CUDA 아키텍처 목록과 바이너리 해시가 각각 다르다. 각 결과의 적용 범위는 해당 바이너리다. [네이티브 빌드 기록](04-structured-output-result.md) · [이미지 빌드 기록](01-validation-result.md#현행-배포-이미지)

감사 이후 `main`의 `a21829d0e`에서 CUDA 헤더의 두 진단 선언에 기능 조건을 적용했다. Production CPU와 C ABI 네 조합이 통과했다. 헤더 수정이 포함된 `120e984c9`의 이미지 빌드와 CPU 검증을 마쳤다. 새 이미지의 런타임 파일 해시는 위 표의 이미지와 모두 일치했다. A100 오류 주입 재시험과 배포 이미지의 실제 GPU 실행은 남아 있다. [후속 빌드와 CI 기록](01-validation-result.md#현행-배포-이미지) · [새 이미지 검증 근거](../../benchmarks/evidence/20261119-cu124-distribution-image/header-fix/receipt.json)

Rust 바인딩을 보완한 `9ea8012c1` 배포 이미지도 CPU 검증을 통과했다. 이 이미지에서 Riley 실행 파일의 해시가 바뀌었다. 별도 hosted 네이티브 CI는 일반 CUDA 서버와 오류 주입 바이너리를 링크한 뒤 CUDA Clippy 단계에서 실패했다. [새 이미지와 바이너리](../../benchmarks/evidence/20261119-cu124-distribution-image/ffi-fix/receipt.json) · [네이티브 CI 단계별 근거](../../benchmarks/evidence/20261119-w5-mixed-soak/memory-faults/ffi-retry/hosted-native-ci.json)

## 현재 판정

| 조건 | 확인된 근거 | 남은 확인 |
|---|---|---|
| W1 Linux·과거 계약 | hosted Linux 검사와 정확한 Python을 쓴 RC3 격리 테스트 34건 통과 | 실제 배포 이미지의 NVIDIA 드라이버 주입과 추론 |
| W2·W3 성능 | 이벤트 발행·스케줄 네 조합, 두 14B 모델의 동일 KV·커널 계측, Qwen2.5-14B 요청별 prefill 대기와 vLLM 한도 대조, Qwen3-4B 길이별 대조. 진행 중인 prefill 우선 후보는 decode TPOT 회귀로 미채택 | 채택할 후보의 재현성과 영향을 받는 모델의 출력·회귀 판정 |
| W4 장문·정확성 | 고정 입력 18개와 holdout 11개의 엔진 비교, 두 사례 첫 BF16 차이 추적과 native layer 0의 KV·causal 재계산, 네 모델의 동일 입력 GSM8K에서 품질 기준 통과 | Riley/HF의 내부 Q/K/V 차이 원인과 최종 배포 후보 재확인 |
| W4 스키마 | 문서화한 부분집합에서 GPU 일반 API 40건과 schema 계약 11건 통과 | 배포 이미지의 대표 모델 API 재확인 |
| W5 운영 | 캐시 off·on 30분 시험, 캐시 재사용·퇴거, 포화 후 회복, 느린 SSE 소비자·취소·종료 회수 검증 | 실제 운영 분포와 장치 장애, 최종 배포 후보의 동일 바이너리 근거 |

기존 이벤트 발행 정책과 서빙 설정을 기준선으로 삼았다. 이전 빌드에서 관찰된 성능 실험은 입력·장비·바이너리에 연결하며, 새로운 수치 기준이나 기본값을 채택할 때는 종합 계획의 출력 불변성과 회귀 기준을 적용한다. [W2 결과](02-async-step-result.md) · [W3 결과](03-long-context-result.md) · [W4 정확성](04-long-accuracy-result.md) · [W5 운영 결과](05-mixed-soak-result.md)

W5의 30분 실행에서 종료 뒤 GPU·pinned 메모리와 활성 KV는 0이었다. Cache on과 allocator를 계측한 cache off의 후반 RSS는 0.00MiB였고 같은 GPU·설정의 비계측 cache-off 두 실행에서 +0.75MiB를 기록했다. 같은 호스트의 CPU 대조에서 `status` RSS가 768KiB 단위로 갱신되는 사례를 재현했다. 상주 페이지 증가 여부와 최종 후보의 메모리 조건은 동일 부하의 `smaps_rollup` 근거가 남아 있다. [운영 품질 기준](../goal.md#33-운영-품질) · [RSS 원본 대조](05-mixed-soak-result.md#cache-off-rss-재현)

배포 이미지 재빌드를 위한 소스 아카이브 전송은 자동 승인 검토에서 거절됐다. 승인된 검증 경로에서 이미지 ID와 실행 파일 해시를 고정한 뒤 대표 모델의 API·정확성·운영 근거를 연결해야 릴리스 판정을 갱신할 수 있다. [이미지 검증 범위](01-validation-result.md#현행-배포-이미지) · [종합 완료 조건](README.md#8-종합-완료-체크리스트)
