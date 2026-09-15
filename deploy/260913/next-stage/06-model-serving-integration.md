# PR-N06 — 선택한 native BF16 후보의 모델·serving 통합

## 공통 적용 조건

[공통 계약과 실행 순서](README.md)를 따른다. 현재는 계획이며 구현·측정 완료를 의미하지 않는다. strict 기본 경로는 보존하고 새 실행은 opt-in으로 둔다. 1차 RTX4090 전체 GPU peak 20,000,000,000 bytes, BF16 weight/KV, Rust→native ABI→CUDA를 유지한다. runtime Python은 사용하지 않는다. Blender 종료 상태를 유지하며 복구하지 않는다.

## 목적과 선행 조건

N01/N02 완료, N03~N05 중 통과 후보 또는 native 대조군 자체의 전체 실행 평가 가치가 문서화돼야 한다. 모든 후보를 구현해야 시작할 수 있는 구조가 아니다. 이 단계부터 전체 모델에 필요한 변경을 수행한다.

## 변경 묶음

1. 측정 경로의 135M 전용 buffer/shape를 선정 모델 descriptor로 확장하고 checkpoint/tokenizer/RoPE/bias/head partition을 모델 계약에 연결한다. 기존 loader와 execution owner를 재사용한다.
2. SmolLM2-1.7B 연결 확인 후 Qwen2.5-3B의 prefill/decode를 같은 native 수치 profile로 연결한다. 후보는 flag로 개별 분리하며 strict와 native를 불투명하게 섞지 않는다.
3. resident-token admission, KV pool, workspace/packing 및 graph bucket을 전체 20GB 예산으로 관리한다. 로딩 중 peak와 cache 잔류도 검증한다.
4. cancellation/stop/EOS/COW/회수/재사용 및 실제 HTTP streaming 경로를 검증하고 fallback 이유를 manifest에 기록한다.

## 검증과 첫 serving milestone

순서: 135M strict 회귀 → 1.7B total context≤8192 연결 → 3B 수치/품질·memory → serving. 품질 실패 중 성능 수치만 통과시키지 않는다.

우선 3B 입력/출력/active=(512,128,1→8),(2048,256,8),(8192,512,1→4), 긴 공유/비공유를 비교한다. 이후 예산과 지원이 확인되면 active32, 입력16384/출력1024/active1→4로 확장한다. 후보 matrix이지 모든 조합 실행 약속이 아니다. 7B는 3B 완료 후 별도 판단한다.

첫 비교표는 동일 조건의 Riley native control / 선택 후보 / 고정 안정 버전 vLLM으로 작성한다. 기존 Riley가 모델을 지원하지 않으면 N/A로 표기하고 135M 수치와 섞지 않는다. 지원되는 SGLang/TRT-LLM은 동일 조건 비교군으로 추가한다. 설치 버전·실제 backend·KV dtype·graph·prefix 정책·EOS·raw를 기록한다.

## 완료·중단·롤백

수치·메모리·serving 완료를 별개로 판정한다. rollout default 변경은 자동 완료 조건이 아니다. 순이득이 없으면 통합된 native 기준의 지원 가치와 성능 후보 실패를 구분해 기록한다. 후보 flag를 끄고 native control 또는 기존 strict 경로로 복귀한다. 공개 배포·commit/push는 이 계획 요청에 포함하지 않는다.
