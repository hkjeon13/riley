# G04 — 서버 통합·측정 준비 검증 결과

**상태: 구현 검증 통과, 경쟁 비교 자격 미통과. 성능 측정은 실행하지 않았다.**

사용자가 승인한 남은 작업을 서버 연결, 실제 128/32 조건 검증, release 후보 고정,
vLLM 환경 확인과 비교 계획 생성까지 연속 진행했다. 작은 구현 단계마다 승인을
다시 요구하지 않았다. 아래의 실제 교차 검증 실패 때문에 `측정 준비 완료`로
표시하지 않는다. 비교 기준을 완화하는 결정은 적용하지 않았다.

## 완료한 구현과 검증

- 실제 executor의 weight/plan/scratch/KV 부모를 보유하는 persistent graph owner를
  추가했다. Rust 참조를 강제로 연장하거나 새로운 unsafe/Send 구현을 추가하지 않았다.
  graph 생성·실행·해제는 동일 서버 worker에서 수행한다.
- scheduler가 발급한 실제 block mapping을 매번 새 metadata로 전달한다.
  single-token prefill, decode, CPU sampling, GPU greedy, 취소, 완료와 KV 회수는
  기존 요청·scheduler commit 경계를 유지한다.
- exact-owner registry의 준비된 슬롯을 재사용한다. 잘못된 새 입력은 이전 결과의
  유효성을 폐기하며, 실행 실패는 owner를 poison 처리한다. completion 불명 상태에서는
  eager 재시도나 scheduler KV 회수의 근거를 만들지 않는다.
- `riley serve --execution-graph-policy disabled|auto|require`를 연결했다.
  graph 선택 시 검증된 grouped attention을 명시적으로 구성한다.
  기본값은 disabled이며, 실제 release 바이너리에서 Require, Auto fallback,
  Disabled, 미지원 Require의 시작 거부까지 확인했다.
- 동일 graph 실행을 `NativeBenchmarkExecutor`와 `riley-profile`에 연결했다.
  `execution_graph_policy=disabled|require`의 c1 설정과 정확한 qualification ID를
  검증한다. graph의 GPU event timing은 unmeasured/null로 표현하며 host 시간을
  GPU 시간으로 표시하지 않는다. native host의 컨테이너 식별자는 null이다.
- 실제 SmolLM2, `Hello` × 128로 토큰 ID `[19556]` × 128을 만들었다.
  128개 입력/32개 출력, 중간 취소에 해당하는 23개 prefix, 서로 다른 비연속 physical
  block mapping을 하나의 graph로 341회 실행했다. 모든 logits가 fresh eager owner와
  정확히 일치했고 마지막 CUDA device/pinned 할당은 0이었다.
- CPU/GPU greedy HTTP 경로에서 128/32, streaming/non-streaming 일치, prefill/decode
  연결 종료, 대기 후 재요청, shutdown 및 zero-allocation을 확인했다.
- CPU library 343개, server CLI 21개, profile CLI 9개, profile checker 22개가 통과했다.
  HTTP 테스트는 직렬 실행했고, C02 권한 테스트는 `umask 077`로 부모 디렉터리까지
  private하게 만든 조건에서 통과했다. 초기 병렬 timeout/umask 조건 실패를 성능
  개선 결과로 해석하지 않는다.
- engine-only qualification은 prepare/close까지만 실행했다. `run_trial`의 측정
  캠페인은 실행하지 않았으며 vLLM도 기능 확인 요청 1개만 실행했다.

## 고정 후보

- 원격 source snapshot: `/tmp/riley-g04-candidate-260911`
- snapshot commit: `2cfd76c27ad1f25f5f41cfea93c2d3f5420e8a9b`
- release binaries: `/tmp/riley-g04-candidate-target/release/riley`, `riley-profile`
- 준비·실패 증거: `/tmp/riley-g04-readiness-260911`
- 정확한 SHA-256: `candidate.json`, `source-bundle.json`, `source-manifest.json`
- 969개 파일의 소스 사본과 `source.tar.gz`를 보존했다. 원래 checkout에는 commit/push를
  하지 않았고, 기존 `crates/riley-model` 6개 변경 파일은 이전 해시와 모두 동일하다.

## 실제 미통과 조건

### 1. vLLM 교차 토큰 정합성

vLLM 0.27.1 / Python 3.13.15 / torch 2.13.0을 실제 import하고 모델을 실행했다.
가중치와 tokenizer.json은 두 엔진에서 동일한 SHA-256이다. 같은 입력 128개와
출력 길이 32개를 사용했지만 **9번째 출력(0-based index 8)**에서 Riley `2341`,
vLLM `1443`으로 처음 갈라졌다. 원본 token arrays와 텍스트는 `parity.json`에 있다.

이는 graph가 기존 Riley 경로에서 바꾼 출력이 아니다. 이 입력의 모든 graph logits는
기존 Riley eager와 정확히 같다. 다만 어떤 연산의 수치 차이가 교차 불일치를
처음 발생시켰는지까지 입증하지는 않았다. 기존 competitive 계약의
`exact-generated-token-hash-when-comparable`, `token_mismatch_count_max=0`을 통과했다고
주장할 수 없다. 입력을 바꿔 이 실패를 숨기거나 gate를 완화하지 않았다.

### 2. CUDA 환경 정합성

Riley 고정 후보는 CUDA Runtime 12.8 / cuBLASLt 12.8.4이고 vLLM의 torch 빌드는
CUDA 13.0이다. 기존 계약의 same-campaign CUDA runtime 조건과 다르다.
기존 vLLM wheel에서 별도 CUDA 13 검증용 toolkit 사본을 만들고 libstdc++를 연결한
빌드까지 수행했지만, 실행 시 기존 HF prefill 경로가 다음 이유로 거부했다.

```
CUDA environment is outside the reviewed cc89/runtime-12080/cuBLASLt-120804 contract
```

이 별도 빌드는 qualified 후보가 아니다. 기존 vLLM package 파일을 변경하지 않았고
기존 CUDA 12.8 후보도 유지했다. 실패 로그는 `riley-g04-cuda13-parity.log`이다.

### 3. 실행 환경 preflight

기존 `benchmarks/scripts/preflight.sh`는 RAM을 `67185598464` bytes로 고정한다.
현재 `/proc/meminfo` 값은 `67185594368` bytes로 4096 bytes 다르므로 exit 2였다.
또한 확인 당시 다른 Blender compute PID 3개가 남아 있었고 GPU 총 사용량은
743 MiB였다. 해당 작업은 종료하지 않았다. 측정 시에는 독점 GPU와 원래 preflight
조건을 충족해야 하며, 준비 과정에서 threshold를 낮추지 않았다.

## 준비한 실행 계획

`measurement-plan.json`에는 실제 source/binary/model/tokenizer/요청 해시와 다음을
명시했다.

- engine-only와 HTTP streaming을 별도 실행
- c1/p128/o32, 각 프로세스 5회 warmup 제외 + 30회 측정
- 5개 독립 fresh-process pair, AB/BA 교대
- engine-only ignore-EOS, HTTP natural-EOS 및 32토큰 완료 확인
- cache-off, greedy, 정확한 CLI 인자와 환경 변수
- SSE 이벤트 시간을 per-token ITL로 잘못 표시하지 않는 raw event 기록

`python3 measure.py`의 준비 검증은 artifact hash/clean source 확인을 통과했고,
`measurement_started=false`와 자격 미통과 사유를 출력했다. `--measure`는 자격
미통과 항목이 있으면 실행을 거부한다. formal competitive executable-lane/preflight
receipt나 M4/M5 성과로 간주하면 안 된다. 측정 성능 수치와 vLLM 대비 우위 주장은 없다.

후속 결정은 실행 승인 재요청이 아니라 **기존 정합성 계약을 유지하며 차이를 해소할지,
출력·환경 차이를 명시한 진단용 비교를 별도 기준으로 허용할지**이다.
