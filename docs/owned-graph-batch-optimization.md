# Owned decode Graph와 수치 일치 배치 개선

최신 검증 후보는 RTX4090용 v6다. QKV와 Gate/Up의 요청 배치, decode Attention의 복사 및 softmax 저장을 개선했다. 기본 Graph/배치 옵션은 비활성이며, `--owned-decode-graph segments-v1 --owned-decode-batch lt-mlp-v2`로 조합을 선택한다. 배포·커밋·푸시는 수행하지 않았다.

## QKV Gate Up Attention 개선 20261017

Qwen2.5-3B-Instruct BF16을 같은 RTX4090에서 실행했다. 아래 개선 전후는 같은 소스 기준에서 이번 8개 파일 변경만 적용한 두 바이너리이며, 둘 다 기존 Graph와 Down 배치 옵션을 켰다. 직전 v3도 별도로 재측정했으며 원시 요약에 포함했다. 처리량은 전체 요청 합산 출력 tok/s다.

| 동시 요청 | 개선 전 Graph+배치 | 신규 배치만 | 신규 Graph+배치 | 동일 옵션 개선율 | stock vLLM 0.27.1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 116.23 | 114.15 | 118.25 | +1.73% | 130.16 |
| 8 | 141.40 | 150.96 | 153.25 | +8.38% | 701.79 |
| 16 | 153.87 | 167.06 | 167.78 | +9.04% | 1094.75 |
| 32 | 171.08 | 191.06 | 186.94 | +9.27% | 1499.53 |

C32에서는 신규 배치만 적용한 191.06 tok/s가 조합 186.94보다 높았다. 같은 옵션의 조합 개선율은 1.73~9.27%다. C32 배치만과 개선 전 조합의 차이는 11.68%다. 이번 측정에서 빠른 구성이 동시성에 따라 달라져 Graph를 기본 활성화하지 않았다.

### 측정 조건

모델·GPU·CPU affinity·token budget·KV 제한·greedy 설정은 아래 20261016 기록과 같다. GPU는 `GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`, native는 SM89 / CUDA12.8.1 / cuBLAS120804다. 설치된 stock vLLM은 Torch2.13+cu130에서 FlashAttention 2와 FULL_AND_PIECEWISE Graph를 선택했다. 외부 작업이 A100 GPU2/3을 계속 점유해 같은 RTX4090 cohort를 유지했다.

각 구성은 새 프로세스로 실행하고 고정 요청 64개를 워밍업한 뒤 48개 요청을 3구간 측정했다. 각 표 셀은 구간별 처리량의 중앙값이며, 3개 구간은 독립 재기동 실험이나 유의성 검정이 아니다. 본 비교 24개 구성 3,456개 요청과 동일 소스 대조 4개 구성 576개 요청, 합계 4,032개 측정 요청이다. 초기화·워밍업을 제외했고 측정 중 새 capture 비용은 포함했다. C16 신규 조합의 첫 구간과 C32 신규 조합의 두 번째 구간에는 각각 새 capture 72개가 있었다.

원시 runner의 `plan.scope`에는 이전 실험의 “five” 문구가 남았다. 실제 argv, `plan.windows=3`, 각 구간 기록과 요청 수가 실행 조건이다. [메타데이터 정정](../artifacts/qkv-attention-opt-20261017/measurement-metadata-amendment-v6.json)에 이 차이를 기록했으며 원본은 보존했다.

### 실제 변경과 수치 검증

QKV와 Gate/Up은 각 요청의 **M=1 행렬 형태와 reduction 알고리즘을 고정**하고 cuBLASLt의 batch count로 2~32개 singleton 요청을 묶는다. Q/K/V 세 projection과 Gate/Up 두 projection은 각각 유지한다. vLLM처럼 채널을 합친 큰 GEMM으로 바꾸는 방식과 구분된다. descriptor와 heuristic 선택은 cold 준비 때 끝내며, SM89/cuBLAS120804 및 검증된 Qwen geometry에서 활성화한다. A100의 기존 SM80/cuBLAS120405 recipe는 별도로 유지한다.

독립 M=1 Lt heuristic의 opaque algorithm을 보존하고 ID13, tile0, stages0, splitK1, reduction0, workspace0 및 모든 배치 크기의 AlgoCheck를 확인한다. 같은 ID를 AlgoInit로 재구성하는 방식은 이 heuristic-only algorithm에서 지원되지 않아 제외했다. 일반 strided/pointer GEMM과 큰 M으로의 병합은 BF16 출력이 달라져 채택하지 않았다.

Attention은 M=1에서 Q와 context의 항등 transpose 복사 두 개를 생략한다. T≤2048에서는 기존 F32 softmax 계산을 유지하면서 최종 BF16 출력을 직접 저장해 별도 scatter kernel을 없앴다. QK·PV geometry, BF16 반올림 경계, prefill 계산과 긴 문맥 softmax는 유지했다. Attention의 요청별 호출 구조는 남아 있다.

첫 모델 검증에서 한 요청의 16개 prefill 토큰을 16개 singleton 요청으로 잘못 선택하는 조건을 발견했다. 이 때문에 `natural-16`의 선택 logits 3행이 달랐다. v6는 첫 요청 그룹이 정확히 1행일 때만 배치 계획을 선택한다. 실패 원본을 보존했고, 다중 토큰 그룹은 원래 M으로 실행한다.

- CPU 회귀: 480 passed, 40 ignored.
- native QKV 93조건과 Gate/Up 385조건: 독립 행 연산 대비 BF16 바이트 일치. 입력 갱신, capture/replay 및 자원 회수 포함.
- Attention: D64/D128, T=1~9216의 62개 경계 조건을 개선 전 같은 GPU 출력과 바이트 비교. 과거 A100 HF 값은 진단 자료로 분리했다.
- 전체 모델: C1 eager, C8 배치, C7 Graph+배치에서 선택한 전체 vocabulary logits 256행이 같은 GPU의 개선 전 C1과 바이트 일치했다. 잘못 처리했던 16토큰 prefill을 포함한다.
- HTTP/SSE: 본 비교와 동일 소스 대조의 저장된 Riley 응답 4,428개에서 token IDs / text / prompt IDs / finish / usage가 개선 전과 같았다. 같은 GPU의 HF 생성 결과 15개 프롬프트와 token IDs / text / prompt IDs도 일치했다. baseline 15개를 3회 반복한 45개 응답도 동일했다.
- 연결 중단 48회, 재시도 72회 및 allocation 안정성 검사 통과. 28개 성능 구성에서 외부 compute PID는 0개였고 각 측정 뒤 active/waiting requests와 사용 KV 블록은 0이었다.

과거 A100 HF teacher logits와 현재 SM89 logits는 동일 환경의 바이트 정답으로 취급하지 않는다. 이번 엄격한 수치 보존 기준은 현재 GPU의 개선 전 출력이다. 같은 GPU HF 생성 일치는 별도로 검사했으며 HF의 모든 logits와 전 입력의 동등성을 주장하지 않는다.

### CUDA trace와 남은 병목

C8에서 동일한 고정 요청 12개, 생성 240 tokens를 Graph 없이 profile했다. 기존 Down 배치를 이미 켠 같은 소스 대조군과 비교하면, 이번 변경의 전체 kernel 실행은 155,638회에서 98,794회로 36.52% 줄었다. GEMV 계열 kernel은 61,620회에서 29,400회로 줄었다. eager부터 전체 개선 경로를 비교하면 162,339회에서 98,794회로 39.14% 감소하며, 이 수치에는 이전 Down 배치 효과도 포함한다. Q gather와 context transpose는 각각 8,640회에서 432회로, softmax scatter는 8,712회에서 504회로 줄었고 남은 호출은 prefill이다. 측정 구간의 CUDA runtime allocation 호출은 0이었다. profile 처리량은 성능 표에 넣지 않았다.

신규 trace에서도 GEMV 계열이 GPU kernel 시간 합계의 약 85.6%를 차지한다. 이 분류에는 linear projection과 Attention의 QK/PV가 함께 포함된다. vLLM의 `QKVParallelLinear`, `MergedColumnParallelLinear`는 요청을 큰 행렬로 처리하고, FlashAttention 2는 `query_start_loc`와 block table을 받아 여러 요청의 Attention을 처리한다. Riley는 검증된 M=1 reduction을 유지하는 배치와 요청별 Attention을 사용하므로, 커널 호출 감소를 vLLM과 같은 Tensor Core 계산 구조로 해석하지 않는다. C8/C16/C32에서 stock vLLM은 신규 조합보다 각각 4.58/6.52/8.02배 높은 처리량을 보였다.

### 지연 시간과 생성 일치

각 셀은 `TTFT p95 / 요청 전체 시간 p95`(ms)이며 구간별 p95의 중앙값이다.

| 동시 요청 | 개선 전 조합 | 신규 조합 | stock vLLM |
|---:|---:|---:|---:|
| 1 | 14.24 / 272.04 | 14.26 / 267.05 | 12.97 / 242.83 |
| 8 | 100.19 / 1772.40 | 89.75 / 1618.29 | 46.29 / 316.84 |
| 16 | 254.34 / 3121.15 | 240.96 / 2840.58 | 92.88 / 388.84 |
| 32 | 828.08 / 5139.82 | 741.38 / 4770.97 | 191.14 / 486.32 |

vLLM은 같은 GPU·모델·요청 조건으로 비교했으며 Riley 개선 전후의 바이트 일치 기준에 포함하지 않았다. 고정 요청의 전체 생성 token 열이 같은 GPU HF와 일치한 비율은 다음과 같고 품질 점수로 해석하지 않는다.

| 동시 요청 | Riley 각 구성 고정 측정 | stock vLLM 고정 측정 |
|---:|---:|---:|
| 1 | 144 / 144 | 96 / 144 |
| 8 | 144 / 144 | 102 / 144 |
| 16 | 144 / 144 | 104 / 144 |
| 32 | 144 / 144 | 101 / 144 |

### 재현 자료

- [원시 검증·실패 로그·CUDA trace·소스 증거](../artifacts/qkv-attention-opt-20261017/final-evidence-v6.tar.gz), [archive SHA256](../artifacts/qkv-attention-opt-20261017/final-evidence-v6.sha256).
- [최종 수치 요약](../artifacts/qkv-attention-opt-20261017/final-summary-v6.json), [이번 변경 CUDA trace 요약](../artifacts/qkv-attention-opt-20261017/profile-incremental-summary-v6.json), [자원·소스 감사](../artifacts/qkv-attention-opt-20261017/resource-source-audit-all-v6.json).
- [전체 변경 patch](../artifacts/qkv-attention-opt-20261017/qkv-attention-v6-full.patch), [932개 파일 재구성 검증](../artifacts/qkv-attention-opt-20261017/full-patch-replay-v6.json). 최적화 8개 파일 외에 기존 owned MLP helper의 API 인자 누락 3곳을 mode0으로 보완해 원래 행별 계산을 유지했다.
- 기준 binary SHA: `2531e79cde58db124c79ab997d9f7aafe05581415ced42f64eed21758cc802c2`.
- 후보 binary SHA: `a03120a6cb61edddf439f446238386f914364b80f17676da4b7f3c8898aee3ed`.
- 원격 원본: `ai-assistant:/data/riley-qkv-attention-opt-20261017`. 실패 결과와 동시 작업의 원본 소스 및 기존 호스트 자료를 유지했다.
- 아래 20261016 표와 이번 표를 서로 다른 GPU 실험처럼 섞거나 구간 수를 합산하지 않는다. A100 수치와도 분리한다.

## RTX4090 이전 비교 20261016

`ai-assistant`의 RTX4090 한 장에서 Qwen2.5-3B-Instruct 고정 체크포인트를 BF16으로 실행했다. 아래는 전체 요청 합산 출력 tok/s이며, 구성별 5개 연속 측정 구간의 중앙값이다.

| 동시 요청 | 개선 전 | Graph만 | 배치만 | Graph + 배치 | 조합 개선율 | stock vLLM 0.27.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 112.63 | 116.18 | 112.66 | 116.19 | +3.16% | 129.80 |
| 8 | 136.78 | 138.24 | 140.14 | 141.75 | +3.63% | 706.35 |
| 16 | 147.60 | 149.00 | 151.93 | 152.55 | +3.35% | 1,096.75 |
| 32 | 163.35 | 165.25 | 170.55 | 167.99 | +2.84% | 1,505.33 |

C32에서는 배치만 적용한 170.55 tok/s(+4.41%)가 조합 167.99보다 높았다. 5개 구간은 같은 프로세스의 연속 표본이므로 독립 재기동 실험이나 통계적 유의성으로 해석하지 않는다. C32 조합의 구간별 범위는 164.73~171.11, 개선 전은 162.12~168.45 tok/s다. 이 결과만으로 Graph를 모든 동시성에 기본 적용하지 않는다.

### 측정 조건과 경계

- 모델: `Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1`; checkpoint manifest SHA `f85648630f4aef16ddc62b04848ce082b577d08d6e14915c2ccd402fdbd377f8`.
- Native: Rust 1.85, SM89, CUDA 12.8.1 / cuBLAS 12.8.4 (`120804`). vLLM은 설치된 0.27.1 / Torch 2.13.0+cu130 환경과 기본 Graph/compile 설정을 사용했다. event-spin 등 수정한 vLLM은 최종 표에 포함하지 않았다.
- 동일한 8개 P core (`0,2,4,6,8,10,12,14`), batch token budget 512, sequence bound 1024, KV block 16 tokens, KV blocks `동시성×64`, prefix caching off. Native budget은 checksum으로 고정한 CLI 소스 기본값 512와 저장한 argv로 확인했다. `/v1/config`는 config_unavailable을 반환해 런타임 설정 증거로 사용하지 않았다.
- 고정 12개 프롬프트(입력 약 115~130 tokens), 출력 상한 8/16/24/32 tokens. 원래 EOS, greedy, repetition penalty 1.0, top-p 1.0, top-k off, frequency/presence penalty 0을 유지했다.
- 20개 새 프로세스(4개 동시성×5개 구성), 프로세스마다 64개 워밍업 요청 후 48개 요청을 5구간 측정했다. 총 측정 요청 4,800개. 초기화·워밍업은 처리량에서 제외했으며 측정 도중 발생한 capture 비용은 포함했다.
- C16 조합의 두 번째 구간에서 새 Graph capture 72개가 발생했다. 이를 지우거나 0-capture 구간만 골라 계산하지 않았다.
- 완료된 각 구성 뒤 A100 GPU2/3 점유와 실제 OS 소유자를 재확인했다. 계속 `wckim` vLLM 작업이 사용 중이어서 동일한 RTX4090 cohort를 유지했다. 다른 작업은 중단하지 않았다.

### 수치 및 자원 검증

CPU 475 passed / 40 ignored. SM89 Down의 독립 probe는 행 수 1~32에서 기존 M=1과 BF16 바이트 일치했다. native GPU 제어 35조건은 혼합 prefill/decode 그룹, 고정 주소 capture/replay, 입력 갱신, 스트림 접근 및 allocation 회수를 포함한다. 전체 모델의 Qwen teacher 검사는 C8 배치·C8 조합·C7 조합에서 각각 128·128·112개 선택 logits 행을 같은 GPU의 C1 eager와 바이트 단위로 비교했다.

HTTP 원시 응답 4,488개(측정·정확성 제어·재시도)는 개선 전 eager와 token IDs / text / prompt IDs / finish reason / usage가 전부 같았다. 같은 GPU에서 별도로 생성한 HF와도 token IDs / text / prompt IDs가 전부 일치했다. 연결 중단 32회와 재시도 48회를 검사했고, 각 검사와 측정 후 active/waiting requests 및 KV 사용 블록이 0으로 회수됐다. allocation 안정성 검사와 20개 구성의 GPU 프로세스 감사가 통과했으며 관측된 외부 GPU PID는 0개였다. 최종 GPU는 compute process 없이 사용 메모리 257MiB, 사용률 0%로 돌아왔다.

기존 A100 HF 정답을 RTX4090에 그대로 적용한 첫 pilot은 `planning`에서 실패해 중단했고 원본과 실패 로그를 보존했다. 현재 GPU와 설치된 Torch/Transformers로 HF를 다시 생성하니 개선 전 native가 15/15 프롬프트에서 일치했다. 모든 15개를 3회 반복한 45개 응답도 동일했다. 새 HF와 과거 A100 HF 사이에서는 5개 프롬프트의 생성 토큰열이 달랐고 prompt IDs는 같았다. GPU·라이브러리 환경을 섞은 정답 비교를 수치 회귀로 판정하지 않았다. 허용 오차는 늘리지 않았다.

아래의 일치율은 고정 프롬프트 요청의 **전체 생성 토큰열이 HF와 같은 비율**이며 품질 점수가 아니다. vLLM 처리량은 이 수치 차이를 함께 고려해 해석한다.

| 동시 요청 | Riley 각 구성, 고정 측정 | vLLM 고정 측정 | vLLM 전체 프롬프트 제어 |
|---:|---:|---:|---:|
| 1 | 240 / 240 | 160 / 240 (66.7%) | 8 / 15 |
| 8 | 240 / 240 | 173 / 240 (72.1%) | 28 / 45 |
| 16 | 240 / 240 | 171 / 240 (71.2%) | 27 / 45 |
| 32 | 240 / 240 | 173 / 240 (72.1%) | 27 / 45 |

### 지연 시간

각 셀은 `TTFT p95 / 요청 전체 시간 p95`(ms)다. 5개 구간별 p95의 중앙값이며 전체 요청을 합친 p95가 아니다. TPOT는 응답당 평균 토큰 간 시간으로, 개별 토큰 ITL과 구분한다.

| 동시 요청 | 개선 전 | Graph + 배치 | stock vLLM |
|---:|---:|---:|---:|
| 1 | 14.24 / 281.22 | 14.29 / 271.95 | 13.85 / 243.24 |
| 8 | 95.53 / 1836.11 | 92.61 / 1757.81 | 47.66 / 313.25 |
| 16 | 254.27 / 3263.51 | 248.09 / 3185.00 | 91.88 / 383.58 |
| 32 | 862.75 / 5297.86 | 833.32 / 5203.37 | 190.62 / 483.89 |

### 적용 범위와 남은 격차

Graph는 레이어별 pre/post-attention segment를 재사용한다. Attention, embedding, final norm/LM head는 기존 eager 경로다. 고정 버퍼와 warm/capture/replay 분리 및 적격하지 않은 배치의 eager fallback은 vLLM의 dispatcher/PIECEWISE 구성을 참고했다.

RTX4090 배치는 SM89 / cuBLAS120804 / BF16 / N=2048,K=11008 / 연속 singleton 2~32행의 Down projection만 strided GEMV로 묶는다. 다른 geometry와 다중 토큰 prefill 그룹은 기존 계산을 보존한다. Nsight에서 기존 Down의 32개 GEMV launch가 같은 계열 GEMV 배치 launch 하나로 줄었고 출력 바이트가 일치했다. Gate/Up은 M=1 GEMV에서 배치 WMMA로 커널이 바뀌어 값이 달랐으므로 그대로 유지한다. A100 Gate/Up Lt recipe는 기존 SM80/cuBLAS120405 조건에 남아 있고 4090에 무조건 적용하지 않았다.

이 때문에 실제 배치 GEMM을 광범위하게 사용하는 vLLM과의 격차가 남는다. QKV·Gate/Up의 행별 연산, eager attention, 부분 Graph라는 제한이 있어 C8/C16/C32에서 조합 대비 vLLM이 각각 약 4.98/7.19/8.96배 높은 처리량을 보였다. 이번 결과를 전체 연산의 배치화 완료나 vLLM 동급 성능으로 제시하지 않는다.

### 재현 자료

- [원시 측정·GPU 검증·소스 manifest](../artifacts/graph-batch-opt-20261016/final-sm89-evidence-v4.tar.gz), [HF/eager 정답 및 pilot 원본](../artifacts/graph-batch-opt-20261016/sm89-reference-pilot-evidence-v1.tar.gz).
- [최종 수치 요약](../artifacts/graph-batch-opt-20261016/final-sm89-summary-v4.json), [baseline 대비 10개 파일 patch](../artifacts/graph-batch-opt-20261016/owned-graph-batch-sm89-v3.patch).
- 기준 binary SHA: `1d59d4dfa318850284a1304eb11bee619564697e84e55057882fcad1d2fca27f`.
- 후보 binary SHA: `762e5eeb6de13be3118d82a283df4c4c669808475d6f55b1df1ef0e4bcea0acc`.
- 원격 증거: `ai-assistant:/data/riley-graph-batch-opt-20261016-rtx4090`.
- A100의 이전 중간 성능 구간과 RTX4090 최종 수치를 합산하지 않는다.

## 이전 A100 실험 기록

아래는 별도 A100 cohort의 v1 검증과 v2 준비 기록이다. A100 v2 실행은 점유 때문에 보류됐으며 위 RTX4090 v3 최종 표와 구분한다.

## 수치 불일치의 원인과 수정

대상은 A100, CUDA 12.4 / cuBLAS 12.4.5, 고정 Qwen2.5-3B BF16 체크포인트다.
실제 활성값과 weight를 checksum으로 고정한 독립 CUDA probe에서 11개
bias-free projection × 5개 행 수 × 7개 수치 정책, 385개 조건을 비교했다.
첫 행의 M=1 결과가 HF와 바이트 일치함을 매번 검사했다. 추가 행은 원본
활성값의 열을 회전해 서로 다른 입력을 만들었다.

| 원인 분리 조건 | 기존 M=1과 다른 BF16 값 |
|---|---:|
| Qwen Down, 기본 32행 결합 | 26,285 / 65,536 |
| 같은 결합, 중간 축약 정밀도 저하 금지 | 127 / 65,536 |
| 같은 결합, FP32 출력 후 BF16 변환 | 127 / 65,536 |
| Gate, 기본 8행 결합 | 55 / 88,064 |
| Up, 기본 8행 결합 | 86 / 88,064 |
| Gate/Up, 고정 cuBLASLt recipe, M=1..32 | 64개 조건 모두 0 |

Nsight NVTX trace에서 Down 32행 기본 경로는 BF16 중간값으로 split-K 결과를
합쳤다. `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 적용 후에는
FP32 중간값을 사용했다. 남은 차이는 기존 M=1의 GEMV/축약 방식과 결합 GEMM의
누적 방식이 다르기 때문이다. 정밀도 상향만으로 기존 결과와 같아지지는 않았다.

Gate/Up의 기본 M=2·8은 `sliced1x2` 커널, M=32는 다른 nonsliced 커널을 선택했다.
cuBLASLt algorithm ID 6 / tile ID 18 / stages ID 16 / split-K 1 / reduction NONE을
고정한 후보는 M=1..32의 64개 조건에서 기존 행별 결과와 바이트 일치했다.
일부 FP64 내적 진단 좌표에서는 기존 결과가, 다른 좌표에서는 변경 결과가 더
가까웠다. 기존 결과의 비트 일치와 수학적 정확도를 구분하고, 채택 기준은
기존 결과와의 비트 일치로 유지했다.

일반 strided-batch와 무제한 행 결합의 실패 소스/로그도 보존했다. Output/Down 등
일치가 회복되지 않은 projection은 기존 M=1 GEMM을 유지한다. 허용 오차를
늘리거나 실패한 결과를 정상으로 간주하지 않았다.

## vLLM에서 참고한 Graph 구성

설치된 vLLM 0.27.1의 `compilation/cuda_graph.py`, `v1/cudagraph_dispatcher.py`,
`forward_context.py`와 [공식 설계](https://docs.vllm.ai/en/v0.27.1/design/cuda_graphs/)를
확인했다. 원본은 `artifacts/graph-batch-opt-20261016/vllm-0.27.1-graph-reference.tar.gz`다.

- 배치 descriptor로 FULL / PIECEWISE / NONE을 선택한다.
- shape별 Graph와 주소가 고정된 입력 버퍼를 재사용한다.
- 호출자는 현재 토큰과 메타데이터를 같은 버퍼에 갱신한다.
- warmup과 capture를 분리하고, 캡처하기 어려운 Attention을 분리할 수 있다.

Riley owned Attention은 KV 길이를 cuBLAS GEMM 크기에 직접 사용한다.
`--owned-decode-graph segments-v1`은 Attention 앞뒤의 고정 연산을 레이어별
piecewise Graph로 재사용한다. Embedding, Attention, 최종 norm/LM head는 기존
경로를 유지한다. 순수 decode, 실제 행 수와 버킷 일치, 연속 슬롯, 모든 행의
출력 존재, 짝수 레이어 조건에서 사용하고 나머지는 동일 수치의 eager 경로다.
첫 적격 실행은 library warmup, 두 번째는 capture 후 실행, 이후에는 replay다.
Graph를 참조 plan·buffer보다 먼저 파괴하며 실패 시 기존 poison 규칙을 따른다.

## 배치 구현과 적용 범위

`grouped-v1`은 같은 M=1 cuBLAS 호출들을 native validation/use-lease 하나에서
제출한다. GPU 행렬을 하나로 합치는 연산은 아니다.

GPU 검증을 통과한 `lt-mlp-v1`은 SM80, cuBLAS 120405, K=2048 / N=11008,
M=2..32의 Gate/Up만 실제로 결합한다. 알고리즘과 descriptor는 cold prepare하고
추가 workspace를 요구하지 않는다. 순수 decode이며 실제 행 수가 prepared
shape와 같을 때 사용한다. 이 조건 때문에 부분·혼합 배치에서 이득이 줄었다.

최신 `lt-mlp-v2`는 실제 2..32행 각각의 descriptor를 cold prepare한다.
부분 배치도 준비한 큰 버킷과 독립적으로 선택한다. 혼합 배치에서는 연속된
singleton 요청 행만 최대 32행씩 묶고, 여러 토큰인 prefill 그룹의 원래 GEMM
범위와 연산은 유지한다. 새 GPU 검증은 대기 중이다. 실행 중 descriptor나
workspace를 할당하지 않는다. 두 최적화 옵션 모두 기본값은 disabled다.

## 완료된 v1 검증

- 11개 projection × M=1..32, 352개 component 조건: 바이트 일치.
- Graph 입력 갱신 후 재사용, cross-stream 접근 차단, 명시적 자원 반환 통과.
- 독립 HF 실제 활성값 component oracle 44개 조건, bias descriptor 검증 통과.
- C8 기준선/배치/Graph 결합: 경로별 full-vocabulary 로짓 208행,
  46,759,936바이트가 C1 기준과 일치. 각 요청 복제본도 검사했다.
- CUDA/runtime/server 회귀: 475 passed, 40 ignored. 최초 C02 4개 실패는
  기본 umask의 writable 임시 디렉터리 때문이었다. 기존 계약에 맞는 umask 077로
  재실행해 통과했고 원래 실패 로그도 보존했다.

v1 binary SHA256:
`7a6227b6693162948e6a6ceca3281191dc72072d6133e792b88711be9bf2686a`.

## v1 성능 실험의 완료 구간

동일 GPU 3, Qwen2.5-3B BF16, 입력 115~130토큰, 출력 8/16/24/32토큰의
12개 고정 프롬프트를 사용했다. 새 프로세스 5회, 각 64회 워밍업 제외 후
48요청 / 출력 960토큰을 측정한 중앙값이다. 아래는 **중간 후보 v1**의 결과다.

| 동시 요청 | 개선 전 | Graph만 | 배치만 | 결합 | vLLM 기본 | vLLM 수정 진단 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 97.13 | 103.23 | 96.45 | 103.40 | 19.88 | 134.83 |
| 8 | 112.92 | 117.99 | 133.24 | 140.39 | 156.47 | 747.92 |

단위는 전체 요청 합산 출력 tok/s다. C16/C32의 완료된 최종 비교는 없다.
C8 결합 한 실행은 측정 중 새 shape의 Graph 72개를 캡처했고 그 비용도 포함했다.
vLLM 수정 진단은 process-local `torch.cuda.Event(blocking=False)` 변경이다.
설치된 vLLM을 수정하지 않았고 기본 vLLM 수치와 구분한다. 이 호스트의 C1
기본/수정 진단 차이가 크므로 기본 실행만으로 일반적인 엔진 우열을 판정하지 않는다.

v1의 보존된 Riley 응답 2,568개에서 HF 토큰열/텍스트 일치 검사를 통과했다.
vLLM은 HF와 다른 생성 결과도 관측했다. C1 입력 토큰열은 동일했고, 기본과
수정 진단은 같은 5개 검사 프롬프트의 같은 위치에서 갈라졌다. 이는 정답률
평가가 아니며, vLLM 중간 모델 연산의 첫 차이까지 특정한 결과는 아니다.

## Nsight 증거의 범위

별도 Graph 진단 후보 `5d1a502...`의 C1 3요청 / 출력 48토큰에서 직접 launch
API가 50,340 → 16,320회로 줄고 Graph launch 3,240회가 추가됐다.
합계 19,560회, 약 61.1% 감소다. GPU 산술 자체를 줄였다는 뜻은 아니다.
`--cuda-graph-trace=graph`이므로 Graph 내부 kernel 개수/시간을 eager와 직접
비교하지 않는다. C8 node trace의 Nsight 2023.4 importer 오류와 C1 batch-only의
GPU activity 누락은 보존하고 해당 GPU 통계에서 제외했다.

## v2 검증·최종 비교 준비

v2 A100 binary SHA256:
`98b47061c0a1a18fb10c07c1d77ff32b70b3cfb21d5d8e48dc7138850946fa00`.
GPU 실행 전 admission에서 다른 사용자 점유가 확인됐다. v2 GPU 테스트가
실행되거나 실패한 수치 결과가 나온 상태가 아니다.

준비된 검증은 큰 버킷 안의 실제 M=1..32 및 혼합 그룹 3종, 11개 projection의
385개 조건과 7/8개 요청의 full-vocabulary 로짓 비교다. 모든 바이트 일치가
확인돼야 최종 성능 측정을 시작한다.

최종 측정 프로토콜은 설정별 새 프로세스 1회에서 워밍업 64요청 후 48요청
측정 구간 5개를 실행한다. 같은 상주 프로세스의 5개 구간이며 5회 독립 재기동이
아니다. C8/C32는 엔진 순서를 뒤집고, 캡처 비용과 모든 응답/지연시간을 보존한다.
이전 v1의 재기동 실험과 최종 표의 표본을 합치지 않는다.

## 재현 자료

`artifacts/graph-batch-opt-20261016/`:

- `numeric-cause-result-v1.json`, `numeric-kernel-summary-v1.json`,
  `rounding-cause-v1.json`, `numeric-audit-v1.json`: 불일치 원인 분리.
- `lt-candidate-source-v1.{json,tar.gz}`, `lt-implementation.patch`: 검증된 v1 소스.
- `lt-candidate-source-v2.{json,tar.gz}`, `lt-v2-expansion.patch`, `lt-v2-implementation.patch`: 미검증 확장 후보.
- `validate_lt2.py`, `measurement-plan-v2.json`, `measure_graph_batch_windows.py`,
  `run_windows_v2.py`, `analyze_windows.py`: 준비된 검증과 최종 비교 절차.

빌드는 고정 소스 스냅샷에 이 작업의 변경만 적용했다. 공유 checkout에서
동시에 진행된 관련 없는 수정은 유지했고, 측정 소스와 섞지 않았다.
