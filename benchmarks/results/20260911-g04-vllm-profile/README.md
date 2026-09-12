# G04 — SmolLM2 측정 전 구현·검증 완료

**코드와 정합성 준비는 완료했다. 성능 측정은 0회다. 현재 남은 실행 조건은 독점 GPU preflight다.**

vLLM과 9번째 토큰부터 달라지던 문제를 해결했다. attention의 BF16 tensor-core softmax, FP32 residual 수명, RMSNorm 합산/FMA, prefill GEMM의 BF16 split-K 부분합이 원인이었다. 관측값이나 외부 KV를 주입하지 않는 native prefill·decode에서 입력 128개와 출력 32개를 검증했다.

## 검증 결과

- 실제 vLLM serving과 prefill·decode 159개 위치, 30개 layer의 Q/K/V/context **19,080개 비교: 불일치 0개**.
- 새 명시적 profile: 요청 4개, prefill 23토큰 취소, 출력 23토큰 후 취소, 서로 다른 physical block mapping, **491회 replay**. 생성 32토큰 일치, 각 replay 전후 CUDA live allocation 통계 동일, 종료 후 할당 0.
- 기존 graph의 Riley eager 전체 logits 일치 테스트도 통과했다. 기존 기본 수치 계약을 변경하지 않았다.
- 실제 release HTTP: CPU/GPU greedy, streaming/non-streaming, 길이 밖 요청의 HTTP 400, 연결 종료 후 재요청 모두 통과.
- CUDA CPU 테스트 84개, runtime CPU 260개, 서버 CLI 22개, profile CLI 10개, checker 25개, 실제 CUDA graph 계약 2개 통과.
- `riley-profile --prepare-only true`: 모델·graph·trial 준비와 명시적 종료 완료, 성능 trial 0회, cleanup allocation 0.

## 고정 후보와 사용 범위

- 원격 source: `/tmp/riley-g04-vllm-profile-source-260911`
- clean commit: `59f02242a2993b698d5f1c6b970b96bad66fb0a4`
- release: `/tmp/riley-g04-vllm-profile-target/release/riley`
- release SHA-256: `4753c83893f9e69de6b70e9083e5c9903e006cb8c17cab5a2880ee364266f8ae`
- `riley-profile` SHA-256: `bcc0230a559e8bf4069f9da89927b5fd8a5706413dde08c5dc456c9c86a13c43`
- 검증한 19개 변경 파일을 로컬에도 반영하고 모든 파일의 SHA-256 일치를 확인했다. 로컬의 기존 다른 변경과 `riley-model` 파일은 유지했다. 로컬 commit/push는 하지 않았다.

서버는 `--execution-graph-policy require --graph-numerics vllm-smol-p128-v1`로 명시적으로 선택한다. SmolLM2-135M BF16, SM89, CUDA runtime 13.0, cuBLASLt 13.1.1, c1/p128/o32가 기록된 비교 범위다. 이 결과로 다른 모델·길이·동시성이나 M4/M5 성능 우위를 주장하지 않는다.

새 수치 정책은 기존 Riley eager와 bitwise 동일한 E0 변경이 아니다. 기존 E0 schema/checker는 그대로 두고, `VLLM_REFERENCE` 및 `riley.vllm-profile-run.v1` 결과와 별도 checker를 연결했다. checker는 고정된 provenance·환경·workload와 각 요청의 입력/출력 token hash를 확인한다. GPU event timing은 계속 unmeasured다.

## 지금 남은 조건

`live-preflight.json`: 다른 Blender compute PID 3개가 남아 있고 GPU 사용량 743 MiB가 기준 256 MiB를 넘어서 exit 2다. 해당 프로세스를 종료하거나 기준을 낮추지 않았다.

GPU가 비워지면 `measurement-plan.json`의 고정 입력·후보로 live preflight를 다시 통과한 뒤 engine-only와 HTTP streaming 성능 측정을 실행할 수 있다. AB/BA 5개 process pair, 각 5회 warmup 제외 및 30회 측정 계획을 준비했고, runner의 기본 실행은 검증만 수행한다. 이번에는 `--measure`를 실행하지 않았다.

주요 증거: `verification.json`, `native-all-final-validation.json`, `http-validation.json`, `native-preparation.json`, `source-manifest.json`, `measurement-plan.json`. 전체 source archive는 원격 결과 폴더의 `source.tar.gz`에 보존했다. 첫 구현에는 prefill 수치 보정을 위한 중복 GEMM 비용이 있어, 실제 성능 개선 폭은 측정으로 판단해야 한다.

추가 확인: 실제 측정용 vLLM adapter도 준비·public engine API 검증·동일 토큰 기능 요청·종료를 통과했다(`vllm-preparation.json`). 설치된 의존성 196개의 전체 이름/버전 집합이 고정 lock과 일치한다. 양쪽 engine 결과의 모든 token hash를 검사하도록 runner를 연결했고, 정상 결과 및 불일치·누락·실패 거부 4조건을 확인했다.

증거 범위: 19,080개 내부 텐서 대조는 관측 코드를 넣은 native 통합 진단 빌드에서 수행했다. 관측 코드를 제거한 최종 명시적 profile은 별도로 491회 replay와 32토큰, 실제 HTTP 응답으로 검증했다. 두 종류의 증거를 동일한 바이너리에서 얻었다고 주장하지 않는다.

## 후속 성능 측정 완료

GUI 유지 별도 조건으로 엔진·HTTP 각 5쌍, 총 600회 측정을 완료했다. 결과와 조건 한계는 [후속 보고서](../20260911-g04-gui-measurement/README.md)에 있다. 기존 256MiB 조건은 통과하지 않았으며, 위의 미측정 상태는 준비 시점의 기록이다. 이번 후보는 vLLM보다 요청 완료 시간이 엔진 약 14.25배, HTTP 약 14.36배 길다.
