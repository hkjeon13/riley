# W2 착수 범위·GPU 준비 기록

확인일: 2026-11-18 (Asia/Seoul)\
관련: [W2 계획](02-async-step.md) · [W1 결과](01-validation-result.md) · [종합 계획](README.md)

## 현재 판단

W2의 코드 조사·소유권 설계·CPU 고장 시뮬레이션 준비는 바로 진행할 수 있다. 실제 CUDA 제출/완료 분리의 검증, GPU 유휴 감소와 성능 이득 확인에는 NVIDIA GPU 서버가 필요하다. 이번에는 소스 조사와 착수 조건 정리까지 수행했으며 비동기 실행 코드를 구현하거나 GPU에 접속하지 않았다.

사전 조사 당시에는 `codex/restore-validation-release-contracts`에서 W1 변경을 미커밋 상태로 보관하고 있었다. 조사 시작 시 W1 source manifest의 1,354개 파일이 모두 일치했다. 이후 사용자 요청에 따라 `main`으로 전환해 W1을 커밋했다. 사전 조사 이후 실행은 [W2 GPU 실행 결과](02-async-step-result.md)에 정리했다.

## 먼저 재현할 가설

[목표 문서](../goal.md)의 SmolLM2-1.7B reason-c32 추가 기록은 nsys 아래 약 476µs의 스텝 경계 유휴와, nsys 없이 host trace에서 약 100µs/5.8ms의 직렬 구간을 구분한다. 후자를 전부 숨겨도 이상적 상한은 약 +1.7%다. 기존 처리량 비율 0.968을 약 0.985로 올리는 정도의 추정이며, 새 측정이나 보장된 효과가 아니다.

따라서 큰 구조 변경에 앞서 같은 후보·모델·워크로드에서 다음 세 실행을 분리한다.

1. trace 환경변수를 unset한 정상 부하: 처리량·TTFT·TPOT 기준선.
2. host trace만 켠 부하: plan/pack/submit/download/sample/commit/events 구간과 관측 비용.
3. nsys CUDA graph node trace: 실제 스텝 경계 GPU 공백과 graph/eager 실행 확인.

프로파일러를 켠 숫자를 정상 부하 개선 가능 시간으로 그대로 대입하지 않는다. 유효한 독립 후보 두 개의 개선이 각각 2% 미만일 때 종료한다는 기존 규칙도 유지한다.

## 현재 코드에서 확인한 경계

| 현재 위치 | 확인 내용 | W2 설계 조건 |
|---|---|---|
| [engine `step`](../../crates/riley-server/src/engine.rs) | plan 이후 graph 또는 일반 executor 경로를 실행하고, 결과 다운로드·sampling·scheduler commit 순서로 진행 | 양쪽 경로를 구분하고 commit 전 토큰 발행 금지 유지 |
| [scheduler 실행 어댑터](../../crates/riley-scheduler/src/execution.rs) | 일반 경로는 output mode에 따라 logits/greedy 결과를 다운로드하고 stream 동기화 뒤 `DownloadedLlamaIteration` 반환 | 제출 상태와 완료 상태를 구분하고 미완료 버퍼 lease 유지 |
| 같은 어댑터의 `classify_runtime_failure` | 동기화 성공 때만 안전한 abort 정보를 제공하고, 완료를 확인하지 못하면 abort 정보를 제공하지 않음 | unknown completion에 재사용·해제·eager 재시도 금지 |
| 같은 어댑터의 downloaded result/workspace 회수 | output slot·iteration ID와 greedy token workspace를 완료 데이터에 결속 | slot 라우팅과 한 번만 회수하는 소유권 계약 보존 |
| [host trace](../../crates/riley-runtime/src/host_trace.rs) | 12개 호스트 구간의 합계를 128스텝마다 stderr로 기록; 환경변수의 존재 여부를 첫 확인 때 캐시 | OFF 실행은 프로세스 시작 전 unset. 실행 중 unset으로 끌 수 있다고 가정하지 않음 |

host trace는 프로세스 전역 합계이며 개별 스텝과 GPU 이벤트의 일대일 인과관계를 제공하지 않는다. 기존 로그만으로 비동기 상태 전이가 올바르다고 판정하지 않는다.

## GPU 없이 진행할 구현 준비

첫 설계는 동시에 실행 중인 GPU iteration 하나를 유지하고, 제출/완료 경계를 분리하더라도 기존 위치에서 기다리는 단계로 제한한다. CUDA event·stream·pinned buffer·KV 참조·iteration ID·output slot의 소유자를 명시한 뒤 실제 타입/API 변경을 정한다. 상태 이름만 있는 시뮬레이터를 추가해 GPU 수명이 보장된 것처럼 취급하지 않는다.

CPU 검증에서는 기존 scheduler/라우팅 테스트를 재사용하면서 새 경계에 필요한 취소·EOS·slot 재정렬·완료 불명·중복 회수 시나리오를 정한다. scheduler/KV를 변경하지 않는 독립 호스트 작업만 중첩 후보로 분류한다. 다음 decode 입력은 이전 토큰·EOS·grammar 상태에 의존하므로 N+1 iteration을 무조건 먼저 enqueue하지 않는다.

실제 GPU 기준선이 재현되기 전에는 성능 이득을 이유로 새로운 큐·이중 버퍼·기본 실행 경로를 도입하지 않는다. W4의 정확성 기준과 W1의 Linux CI 실행 확인도 최종 채택 전에 필요하다.

## 필요한 서버와 전달 정보

첫 SmolLM2-1.7B 조사에는 **GPU 한 장**으로 시작할 수 있다. 기존 성능 근거와 직접 비교하려면 프로젝트 주 대상인 **A100 40GB**와 승인된 모델·CUDA 환경을 재사용한다. 40GB는 기존 기준 장비 사양이며 1.7B 모델의 최소 메모리 요구량을 새로 산정한 수치는 아니다. 다른 GPU로 기능 검증할 수 있으나 A100 성능 근거와 혼합하지 않는다.

서버를 정할 때 필요한 정보는 SSH 별칭, 배정 GPU 번호·독점 사용 가능 시간, 승인된 모델/manifest 위치, CUDA 빌드 환경, 기존 Riley·vLLM 실행 환경과 nsys 사용 가능 여부다. 기존 목표 문서의 GPU 6·7 비교 규칙은 당시 서버에 관한 것이므로 새 서버의 장치 번호로 그대로 옮기지 않는다. 서버 가용성이나 현재 설치 버전은 이번에 확인하지 않았다.

TP=2 검증은 단일 GPU 후보가 유효한 뒤 두 장으로 확장한다. CUDA 컴파일·링크만 확인할 때는 GPU 장치가 없어도 지원 Linux/CUDA 빌드 환경에서 가능하지만, macOS CPU 테스트만으로 CUDA 경로나 이벤트·메모리 수명을 검증할 수는 없다.

## 다음 완료 지점

- CPU 준비 완료: 소유권 설계와 기존 테스트 재사용 범위 확정.
- GPU 조사 완료: 같은 세션의 trace OFF/host trace/nsys 기준선과 관측 비용 기록.
- 구현 착수 판단: 숨길 수 있는 비용·독립 작업·예상 이득을 근거로 최소 후보 선택 또는 보류.
- 채택 판단: 동일 출력·취소·실패·자원 수명 검사, 반복 성능과 회귀 기준 통과 후 결정.

이번 결과는 W2 사전 조사이며 W2 구현·성능 개선 완료를 뜻하지 않는다.
