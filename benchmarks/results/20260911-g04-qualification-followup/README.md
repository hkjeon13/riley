# G04 후속 검증 — CUDA 통일, 환경 식별, 출력 차이 진단

**측정 준비는 아직 미완료다. 실제 성능 측정은 0회다.**

## 이번에 해결한 항목

- SmolLM2 c1/M=1 배치 owner가 실행하지 않는 dense prefill 계획 때문에
  CUDA 12.8 전용 HF cuBLASLt 검사에 걸리던 준비 의존성을 제거했다.
  실제 실행하는 paged attention과 기존 HF RMSNorm 반올림은 유지했다.
- CUDA 13 nvcc의 host stub이 요구하는 Linux C++ 런타임을 빌드에 연결했다.
  최종 release는 별도 RUSTFLAGS 없이 빌드했다. 실제 연결은 CUDA runtime
  13.0, cuBLASLt 13.1.1이며 nvcc는 13.3.73이다. 드라이버 표시 버전을
  실행 런타임으로 대신 기록하지 않았다.
- CUDA 13과 12.8에서 각각 graph 1회 생성, 341회 재사용, 모든 eager logits
  일치, 마지막 device/pinned 할당 0을 확인했다. 두 환경의 출력 32개도 같다.
- CUDA 13 release 서버에서 Require/Auto/Disabled의 M=1 요청 및 미지원
  Require M=2의 시작 거부를 확인했다. CPU/GPU greedy HTTP lifecycle 2개도
  스트리밍, 취소, 재사용, 종료, 할당 0 조건을 통과했다.
- 현재 RAM 67185594368 bytes는 새 exact 환경 ID
  `rtx4090-ubuntu22-driver580-20260911-v2`로 등록했다. RAM 차이 4096 bytes의
  부팅 원인은 확정하지 않았다. 기존 v1을 바꾸거나 RAM 허용 오차를 넣지 않았다.
  GPU 독점, idle 256 MiB, 온도, 드라이버, 디스크 등 검사 기준도 유지한다.
- CPU runtime 259개, preflight 3개, competitive 검사 45개가 통과했다.

CUDA 13 일반 dense HF prefill이나 Auto M=2 fallback 전체를 검증했다는 뜻은
아니다. 위 GPU 자격 범위는 SmolLM2 c1/M=1이다. 기존 CUDA 12.8 HF prefill
allowlist는 그대로다.

## 출력 불일치: 확인한 사실과 미해결 범위

가중치, tokenizer, `Hello` 128회 입력과 출력 길이 32를 유지했다.

1. 정식 Riley 후보는 CUDA 12.8/13에서 같은 출력을 낸다. HF eager의
   128-token prefill 출력과도 32개 토큰이 같다. CUDA 버전 통일로 기존
   vLLM 기본 실행과의 9번째 출력 차이가 없어지지는 않았다.
2. 같은 첫 embedding 입력에 대한 HF RMSNorm과 vLLM native RMSNorm은
   576개 값 중 149개가 다르다. 최대 절댓값 차이는 0.00390625다.
3. 별도 소스 사본에서 HF RMSNorm의 중간 BF16 반올림만 제거하면
   **vLLM eager와 32개 토큰이 모두 일치**한다. 이 사본은 진단 전용이다.
   원래 HF 연산 계약을 바꾸므로 해당 수정은 정식 후보에 반영하지 않았다.
4. 기본 vLLM compiled 실행과 vLLM eager 자체도 출력 뒤쪽에서 갈라진다.
   따라서 반올림 한 곳의 수정으로 기본 vLLM과의 전체 정합성까지 해결됐다고
   주장할 수 없다. compiled/eager 사이의 최초 연산 차이는 아직 특정하지 못했다.

원본 배열과 첫 불일치 위치는 `diagnosis.json`에 있다. 입력을 바꾸거나
vLLM의 최적화를 꺼서 기본 경쟁 기준을 통과한 것으로 표시하지 않았다.
`norm-only-experiment.log`의 `full_logits_exact`는 수정 사본의 graph/eager
사이 일치이며, 정식 후보나 기본 vLLM과의 일치를 뜻하지 않는다.

**남은 개발 항목은 기본 vLLM compiled와의 수치 경로 차이를 추적하고,
기존 HF 계약과 구분되는 구현이 필요한지 판단한 뒤 교차 검증을 통과하는 것이다.**
이는 현재 해결되지 않았으며 새 비교 기준이나 예외를 적용하지 않았다.

## 실행 환경의 실제 차단

새 v2 사전 검사는 RAM 확인을 지나 GPU idle memory 743 MiB > 256 MiB에서
멈춘다. 다른 프로젝트의 Blender PID 4007728, 201778, 501891이 존재한다.
이 작업들을 종료하거나 GPU를 초기화하지 않았다. 이 세션들의 GPU 사용이
끝나야 독점 preflight를 통과할 수 있다. `runtime-host.json`과
`preflight-v2.stderr`에 현재 원본을 남겼다.

## 고정 및 재검증

- 원격 후보: `/tmp/riley-g04-followup-source-260911`
- commit: `9b53ffa14cea7c066fda8eff65b977c860b12afb`
- source archive/manifest: `source.tar.gz`, `source-manifest.json` (969개 파일)
- release: `/tmp/riley-g04-followup-target13/release/riley`, `riley-profile`
- 원격 증거: `/tmp/riley-g04-followup-260911`
- 원래 작업 트리에는 commit/push를 하지 않았고 기존 모델 파일 6개는 보존했다.

새 `measurement-plan.json`은 CUDA 13 경로와 v2 환경을 고정한다.
`python3 measure.py`는 소스/바이너리/입력 해시 확인만 통과했고
`measurement_started=false`를 기록했다. 출력 정합성과 GPU 독점이 미통과인
동안 `--measure` 실행은 거부한다. M4/M5 또는 성능 우위의 증거는 아니다.

진단 스크립트는 correctness-only이며 시간·처리량 표본을 수집하지 않는다.
초기 빌드 캐시 경로 충돌 및 bin feature 누락은 수정 후 실제 바이너리와
HTTP 실행으로 확인했다. 무실행 cargo 성공을 빌드 증거로 사용하지 않았다.
