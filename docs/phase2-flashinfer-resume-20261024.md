# Phase 2 재개: FlashInfer 서빙 불변성 수정

2026-10-24 KST. Qwen3-4B BF16 greedy, aws-a100 GPU6/7. 정식 90-arm qualification을 완료해 G1~G5를 모두 통과했다. 아래 초기 스크린과 최종 qualification 결과를 구분한다. 새 수치 경로는 opt-in 프로파일로 제공한다.

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

두 동시성의 correctness canonical 불일치는 0개다. 이 표는 정식 대조 전의 스크린이며, 최종 성능 판정은 아래 90-arm 결과에 따른다.

## Qualification

사전 등록: [qualification-preregistration.json](../benchmarks/evidence/20261024-phase2-resume/qualification-preregistration.json). 측정 시작 전에 원격에 해시를 기록했다.

6셀 × 5 fresh-process 반복 × native/candidate/stock vLLM = 90 arms를 모두 완료했다. 중단·제외·재측정은 0개다. fixed C1/C8/C16은 GPU6, fixed C32와 natural C8/C32는 GPU7에서 실행했다. 셀 내 모든 엔진은 같은 GPU를 순차 사용했고, 양쪽 원래 holder PID 2027914/2027915 및 start ticks `6083215119`/`6083215121`를 유지했다. 외부 GPU 작업은 변경하지 않았다.

G1 GPU 검사, G2 모든 arm 192/192 및 교차 동시성 불변성, G3 correctness HF ≥11/15(또는 33/45), G4 natural C32가 같은 캠페인의 vLLM 이상이면서 native 대비 ≥5%, G5 셀별 처리량 ≥−1.3%·TPOT p50 ≤+1.5%·TPOT p99 ≤+10%를 모두 만족해야 채택한다.

근거: [`benchmarks/evidence/20261024-phase2-resume/`](../benchmarks/evidence/20261024-phase2-resume/). 원격 raw 요청과 실행 산출물은 `/mnt/efs/nlp/psyche/riley-phase2-resume-20261024/`에 보존한다.

## 최종 성능 (5회 중앙값)

각 셀은 warm 64 + 측정 192이며, native/page와 후보의 token budget은 256, stock vLLM은 576이다. context 1024, KV block 수 C×64, greedy, prefix cache off다. vLLM의 기본 compile/CUDA Graph 경로를 유지했다. 변화율은 baseline/candidate의 반복별 비율 5개의 중앙값이므로 중앙값끼리의 비율과 약간 다를 수 있다.

| 워크로드 | C | page-native tok/s | 후보 tok/s | vLLM tok/s | 후보/vLLM | paired 개선 |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 1 | 123.847 | 124.800 | 115.612 | 1.0795 | +0.56% |
| fixed | 8 | 716.888 | 767.245 | 710.461 | 1.0799 | +7.02% |
| fixed | 16 | 1155.668 | 1285.158 | 1129.782 | 1.1375 | +11.45% |
| fixed | 32 | 1696.083 | 1940.319 | 1686.906 | 1.1502 | +14.27% |
| natural | 8 | 817.904 | 847.670 | 830.515 | 1.0207 | +3.63% |
| natural | 32 | 2118.277 | 2370.867 | 2303.270 | 1.0293 | +11.98% |

| 워크로드/C | TPOT p50 변화 | TPOT p99 변화 | TTFT p95 변화 | ITL p99 변화 |
|---|---:|---:|---:|---:|
| fixed/1 | +0.38% | -0.06% | -9.79% | +0.42% |
| fixed/8 | -6.25% | -6.79% | -14.04% | -15.00% |
| fixed/16 | -10.79% | -13.95% | -10.84% | -14.52% |
| fixed/32 | -13.67% | -15.34% | -14.06% | -14.08% |
| natural/8 | -3.30% | -4.32% | -15.44% | -17.18% |
| natural/32 | -12.03% | -14.65% | -16.32% | -22.25% |

TPOT은 요청별 평균 inter-token 시간의 분위수다. 토큰별 도착 간격 ITL과 구분한다.

### 게이트 판정: 모두 통과

| 게이트 | 관측 | 판정 |
|---|---|---|
| G1 | 사전 GPU 검사 24개 head/context/입력 조합 + 순서/청크/이웃/Graph/패딩 검사, 기존 direct/fused 회귀 통과 | 통과 |
| G2 | 후보 30 arms × 192 = 5760/5760 canonical 일치. 모든 셀·캠페인 루트의 correctness canonical 불일치 0 | 통과 |
| G3 | C1 12/15, C>1 36/45 (하한 11/15, 33/45) | 통과 |
| G4 | natural C32: 같은 캠페인 vLLM의 1.0293배, page-native 대비 +11.98% | 통과 |
| G5 | 최저 처리량 변화 +0.56%, 최대 TPOT p50 변화 +0.38%, 최대 TPOT p99 변화 −0.06% | 통과 |

**남은 지연 격차:** fixed C32 TTFT p95는 후보 226.180ms / stock 162.848ms이고, natural C32는 248.007ms / 223.330ms다. 기존 native보다 각각 약 14%/16% 줄었지만 stock보다 느리다. 처리량 승리를 모든 지연 지표의 승리로 해석하지 않는다.

**수치 계약과 HF 일치:** 후보는 자기 출력의 배치 불변성을 만족하는 새 수치 모드다. correctness 세트의 HF 일치는 page-native와 같은 12/15지만, 반복 측정 workload에서는 fixed의 HF 토큰 완전 일치가 160/192→176/192, natural은 128/192→64/192로 달라졌다. stock은 이번 natural C8/C32에서 각각 88–99/95–103개였다. HF-literal 정확 일치가 필요한 경우에는 별도 strict 경로를 사용해야 한다. 이는 HF 완전 일치를 보존한 최적화라는 주장이 아니다.

### 고진폭 입력의 독립 참조 검사 (추가 검증)

초기 proximity 검사와 같은 고진폭 생성 규칙(Q/K 범위 약 ±6)을 Qwen3 head 형상(32/8, D128), 순열 페이지 및 prompt/decode 혼합 배치에 적용했다. 실제 BF16 Q/K/V 입력에 대해 별도로 구현한 CPU FP64 dot/softmax/weighted-sum과 40,960개 출력 원소를 비교했다. 후보 최대 절대 오차는 **0.0026596451**, 기존 page 경로는 **0.1148186409**였다. 후보는 이 입력의 이웃 유무에 따른 출력 불변성 검사도 통과했다.

따라서 이 fixture의 큰 cross-mode 차이는 후보가 page 결과에서 멀다는 사실을 뜻하며, FP64 기준 오차는 오히려 작았다. 초기 실패 로그를 그대로 유지하고, 사전 게이트의 normalized-input 조건을 사후 변경하지 않았다. 이 추가 검사는 일반적인 모든 입력에 대한 오차 상한 증명이 아니다.

### 실행 프로파일

`--inference-profile native-bf16-varlen-v1`이 `--native-attention-mode fixed-tile-unsplit-v1`과 benchmark 설정을 선택한다. context ≤1024, budget ≤256, active sequences ≤32를 검사하며 모든 레이어의 head dimension이 128이어야 한다. 기존 `native-bf16-v1`은 `page-v1`을 명시적으로 선택한다. 일반 실행의 기본값은 변경하지 않는다.

기존 native 프로파일에는 게시된 비교 argv와 차이가 있던 prompt attention, plateau decode bound, event publication, warm-up, staged upload 기본값을 맞췄다. 게시된 `native-extra.json`의 모든 옵션을 프로파일 확장 결과와 대조하는 검사를 추가했다. 로컬 CLI 40개 및 GPU 테스트 Rust typecheck가 통과했고, Linux CUDA 빌드에서 프로파일/게시 argv/owned-graph feature 검사도 통과했다.

프로파일 바이너리 SHA256: `e1d6483fe70b62dcdea49c38a4eb7a2cce61c72fe0aeaa96886378b95197f306`. 이 바이너리는 CLI 선택 기능을 추가한 별도 빌드이며, 90-arm 처리량 숫자는 앞서 동결한 `a5058be5…` 바이너리의 결과다.

새 프로파일을 환경 변수 없이 C1/C32에서 실제 실행했다. correctness 15/45개와 측정 요청 각 12개, 총 **84개 출력**이 qualification 후보의 canonical과 모두 같았다. 각 arm의 스트리밍 중단 2회 후 active/waiting/KV가 0으로 복구되고 재시도가 성공하는 기존 검사도 통과했다. 작은 프로파일 smoke의 처리량은 새로운 성능 결과로 사용하지 않는다.

```bash
./target/release/riley serve --model /path/to/verified/Qwen3-4B \
  --inference-profile native-bf16-varlen-v1 \
  --max-active-sequences 32 --max-output-tokens 128
```

### 증거와 남은 범위

- `qualification-analysis.json`을 원격과 로컬에서 같은 분석기로 집계해 G1~G5 판정을 재확인했다. `final-manifests.json`에 qualification/profile 소스 해시와 원격 raw 요청 파일 90개의 크기·해시를 보존했다. raw 요청은 EFS에 남기고 요약·canonical·실행 argv와 검증 로그를 Git에 포함한다.
- 최종 profile smoke 후 GPU6/7은 원래 holder PID/start ticks와 idle 20GiB 상태로 돌아왔다. 외부 GPU0의 tritonserver도 보존했다.
- 이번 채택 범위는 Qwen3-4B/A100의 위 6개 셀이다. 더 긴 context, 다른 모델/GPU, 고정 KV split 및 남은 TTFT 격차 개선은 후속 검증 범위다. 일반 서버 기본 경로나 HF-literal 모드를 대체한 결과가 아니다.
