# Phase 2 재개: FlashInfer 서빙 불변성 수정

2026-10-24 KST. Qwen3-4B BF16 greedy, aws-a100 GPU7. 정식 qualification은 진행 중이며, 아래 서버 수치는 스크린이다.

## 확인한 기존 상태

- `main`의 기존 커밋은 원격과 같았다. Phase 0/1 후속 코드와 자료를 `f1656d560`으로 커밋·푸시하고 README에 native/vLLM 비교표를 추가했다.
- Phase 0의 native는 fixed C1/C8/C16/C32에서 vLLM의 1.066/0.998/1.004/1.009배, natural C8/C32에서는 0.988/0.914배였다. HF-literal 경로와 구분해야 한다.
- Phase 1의 fused/segment/direct/hybrid attention 및 버스트 정책은 사전 채택 기준을 충족하지 못했다. 기본 경로는 유지한다.
- Phase 2는 고정 query tile의 FlashInfer 바인딩과 GPU planner까지 구현돼 있었다. 단위 테스트는 통과했지만 서빙의 `natural-398` 출력이 배치에 따라 달랐다. 마지막 always-plan 진단은 셸 구문 오류로 실행되지 않았다.

## 이번 변경과 진단

기존 p2e 바이너리에서 GPU planner를 매 레이어 재실행하는 arm과 재사용하는 arm을 모두 실행했다. 둘 다 출력 불변성이 실패했다. 측정 192개 중 canonical 일치 수는 always-plan C8/C32 128/134, reuse-plan C8/C32 138/157이었다. 이 진단 중 별도 CPU 빌드도 수행했으므로 처리량은 채택 판단에 쓰지 않는다.

FlashInfer translation unit에만 `--fmad=false`를 적용했다. online softmax의 `old_max * scale - new_max * scale`는 두 max가 같을 때 정확히 0이어야 한다. 곱셈·뺄셈을 FMA로 합치면 반올림 잔차가 남을 수 있고, causal mask 뒤의 추가 iteration 수가 query run에 따라 달라질 때 출력이 달라질 수 있다. Tensor Core MMA 자체는 이 옵션으로 비활성화되지 않는다.

기존 third-party 헤더는 변경하지 않았다. FlashInfer 버전은 0.6.16.post3이고, 포함한 파일의 원래 해시와 라이선스를 `third_party/flashinfer/`에 보존한다.

**계획과 구현의 차이:** 현재 후보는 query tile 64를 고정하고 각 행의 전체 context를 KV split 없이 처리한다. 원래 계획의 split 256 구현이 아니다. 작은 구현으로 서버 스크린의 충분한 이득을 확인했으므로 먼저 이를 qualification한다. 장문·다른 형상에서 추가 병렬성이 필요하면 고정 split 구현은 다음 단계로 남는다.

서버 벤치 러너도 수정했다. 이전 `complete.json`은 native의 eager 일치를 실제 assert하지 않으면서 `all_native_eager_reference_assertions_passed=true`라고 기록했다. 이제 수집 완료, strict assert 여부와 native 불변성을 따로 기록하고, `--require-native-invariance`에서는 canonical 불일치가 하나라도 있으면 비정상 종료한다. 과거 원자료는 수정하지 않는다.

## 검증

후보 바이너리: `a5058be586ec2443661be239dfabe4dbbf27eea194de6a6907c2b6015a6eb819` (`riley-p2g-no-fma`).

- GPU: 8개 head 구성에서 20/60페이지 periodic 입력 및 60페이지 nonperiodic 입력을 검사했다. 순서·타일 정렬·이웃·청크·step 종류에 따른 행별 바이트 일치, 패딩 0, 두 레이어를 캡처한 Graph의 2회 replay와 eager 일치를 통과했다. 페이지 경로와 최대 차이는 nonperiodic 입력에서 0.0078125였다.
- 기존 prompt-direct 및 fused-decode GPU 회귀도 통과했다. 빌드/테스트 로그는 아래 근거 디렉터리에 있다.
- 별도 초기 고진폭 입력(Q/K 범위 약 ±6)에서는 페이지 경로와 차이가 0.075927734로 테스트의 cross-mode 허용치 0.03125를 넘었다. 이 실패 로그를 `random-before.log`로 보존한다. 최종 nonperiodic fixture는 범위 약 ±2이며, 고진폭 조건까지 정확성을 통과했다고 주장하지 않는다. 두 수치 모드의 출력 바이트가 같다는 계약도 아니다.
- 로컬 CPU: CLI 38개, scheduler 단위 32개, burst 통합 4개, dense-row × sequence bound 1개가 통과했다. 기존 CLI 테스트가 CPU 빌드에서 사용할 수 없는 CUDA profile의 성공을 요구하던 부분을 feature별 기대값으로 수정했다. 로컬 macOS에는 nvcc가 없으므로 CUDA 빌드는 A100 호스트에서 수행했다.

GPU 하네스(µs/레이어, 진단용):

| 경로 | decode32 | mixed 31 + prompt 224 |
|---|---:|---:|
| 기존 page 경로 | 59.7 | 184.4 |
| 후보 FlashInfer | 57.6 | 69.1 |

서빙 스크린(각 셀 1개 프로세스, warm 64 + 측정 192):

| Natural | 후보 tok/s | canonical 일치 | HF correctness |
|---|---:|---:|---:|
| C8 | 848.277 | 192/192 | 36/45 |
| C32 | 2370.252 | 192/192 | 36/45 |

두 동시성의 correctness canonical 불일치는 0개다. 같은 캠페인의 vLLM 대조를 아직 마치지 않았으므로 qualified vLLM 초과 결과로 보고하지 않는다. 기본 활성화도 하지 않았다.

## Qualification

사전 등록: [qualification-preregistration.json](../benchmarks/evidence/20261024-phase2-resume/qualification-preregistration.json). 측정 시작 전에 원격에 해시를 기록했다.

6셀 × 5 fresh-process 반복 × native/candidate/stock vLLM = 90 arms다. fixed C1/C8/C16은 GPU6, fixed C32와 natural C8/C32는 GPU7에서 실행한다. 셀 내 모든 엔진은 같은 GPU를 순차 사용하며, 양쪽 원래 holder PID 2027914/2027915와 CUDA context를 유지한다. 외부 GPU 작업은 변경하지 않는다.

G1 GPU 검사, G2 모든 arm 192/192 및 교차 동시성 불변성, G3 correctness HF ≥11/15(또는 33/45), G4 natural C32가 같은 캠페인의 vLLM 이상이면서 native 대비 ≥5%, G5 셀별 처리량 ≥−1.3%·TPOT p50 ≤+1.5%·TPOT p99 ≤+10%를 모두 만족해야 채택한다.

근거: [`benchmarks/evidence/20261024-phase2-resume/`](../benchmarks/evidence/20261024-phase2-resume/). 원격 raw 요청과 실행 산출물은 `/mnt/efs/nlp/psyche/riley-phase2-resume-20261024/`에 보존한다.
