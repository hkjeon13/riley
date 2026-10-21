# Qwen3-4B packed prefill projection v12와 vLLM 재비교

2026-10-20. Q/K/V/O prefill 배치화를 구현·검증했다. TTFT p95는 약5.7–5.8% 줄었지만, 처리량은 C16 -0.12%, C32 +1.95%에 그쳤다. vLLM 대비 큰 처리량 격차는 남아 있다. C16의 한 반복에서는 TPOT tail 악화도 관찰됐다.

## 적용 우선순위와 라이브러리 조사

이번 작업은 직전 v11의 미세 tile 조정 대신, 요청별 projection 실행을 총 토큰 단위로 묶는 구조 변경을 먼저 적용한다. 기준 바이너리는 `/mnt/efs/nlp/psyche/riley-batched-v11e-20261020/riley-batched-v11e`이며 SHA256은 `2fb223ba679556fc3797fdc745ea5597744352c9de48f71396cf9a324c4e44a4`다. 과거 v10과 비교한 개선율을 이번 개선율로 사용하지 않는다.

| 우선순위 | 기성 구현과 이유 | Riley 상태와 이번 결정 |
|---|---|---|
| 1 | vLLM Qwen2의 선형 층은 요청별 루프 대신 `hidden_states`의 전체 토큰을 처리한다. QKV와 Gate/Up도 합친 projection을 사용한다. | Gate/Up·Down의 일부 prefill 묶음은 이미 있었다. 남은 Q/K/V/O prefill을 총 토큰 단위로 묶되, QKV의 N 차원 합치기는 알고리즘 선택과 누적 순서를 바꿀 수 있어 이번에는 분리된 가중치를 유지한다. |
| 2 | vLLM은 uniform decode의 full Graph와 prefill/mixed의 piecewise Graph를 구분한다. | 직전 v11의 pure decode body Graph를 유지한다. mixed/prefill 전체 Graph는 동적 요청 경계와 attention이 남아 있어 별도 설계가 필요하다. 이번 packed projection은 그 전제인 선형 실행 단위 정리를 수행한다. |
| 3 | vLLM MRV2는 pinned staging과 persistent CPU state를 분리하고 변경분을 전송하며 GPU에서 입력 metadata를 준비한다. SGLang·TensorRT-LLM은 CPU scheduling과 GPU 실행을 겹친다. | 기존 staged-batch 업로드는 유지한다. 새 overlap scheduler는 아직 적용하지 않는다. embedding 오류 후 KV mutation 여부, 취소, EOS, pinned buffer 재사용 시점과 이중 버퍼 소유권을 함께 바꿔야 한다. 특히 TensorRT-LLM 문서의 한 단계 추가 decode를 그대로 가져오지 않는다. |
| 4 | FlashAttention은 attention score 중간값의 메모리 왕복을 줄이는 구현이다. | drop-in 교체를 이번에 선택하지 않는다. 공식 검증도 수치 허용 오차를 사용하며, Riley의 기존 native/HF literal BF16 equality를 보장하는 계약과 다르다. 원래 probability BF16 경계와 reduction tree를 유지하는 별도 exact 구현이 필요하다. |

출처: [vLLM Qwen2 구현](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py), [vLLM CUDA Graph 설계](https://docs.vllm.ai/en/latest/design/cuda_graphs/), [vLLM MRV2 설계](https://github.com/vllm-project/vllm/blob/main/docs/design/model_runner_v2.md), [SGLang overlap scheduler](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [TensorRT-LLM overlap scheduler](https://nvidia.github.io/TensorRT-LLM/1.3.0rc20/torch/features/overlap_scheduler.html), [FlashAttention 검증 기준](https://github.com/Dao-AILab/flash-attention#tests).

위 링크의 최신/main 설계는 참고 자료다. 실제 비교 대상인 설치된 vLLM `0.27.1+cu129`가 최신 MRV2를 사용한다는 뜻은 아니다. 설치된 Qwen3/Qwen2/linear/activation 소스는 수정 없이 별도로 해시를 기록했다. 외부 라이브러리의 공개 성능 수치를 Riley의 예상 개선율로 사용하지 않았다.

## 구현과 적용 범위

Q/K/V/O의 요청별 prefill 행들을 인접한 구간에서 합친다. attention은 원래 요청 경계를 사용한다. 각 projection의 native M128 누적 방식과 같은 cuBLASLt 알고리즘을 cold prepare에서 고정하고, 합치는 요청의 원래 크기와 합친 크기 모두 검증 범위 안에 있을 때만 실행한다. 단독 prefill이나 범위 밖 입력은 기존 경로로 처리한다.

| Projection | N × K | 원래 요청 M 범위 | 합친 M 범위 | Lt id/tile/stage/splitK/reduction/swizzle/custom |
|---|---:|---:|---:|---|
| Q | 4096 × 2560 | 65–130 | 257–576 | 6/18/12/0/0/1/0 |
| K, V | 1024 × 2560 | 94–128 | 161–512 | 6/18/12/4/4/1/0 |
| O | 2560 × 4096 | 65–128 | 161–576 | 6/20/11/5/4/0/0 |

FP32 accumulation, BF16 입출력, 원래 split-K/reduction 방식과 기존 private mode11 및 buffer-use bookkeeping을 사용한다. 새 hot allocation은 없다. plan당 workspace는 기존 상한 16MiB 안에서 `AlgoCheck`가 요구한 크기만 cold allocation한다. cuBLASLt 120405, SM80/108SM, unmodified workspace/SM 환경 가드를 유지한다.

수정 파일은 `crates/riley-cuda/src/gemm.rs`, `crates/riley-runtime/src/llama/forward.rs`, `kernels/src/cublas_gemm_probe.cu`다. Qwen의 Q는 일반 hidden plan이 아닌 `hf_separate_query`를 사용하므로 그 plan을 준비한다. 첫 후보에서 이 준비 경로가 누락된 기록은 별도 EFS root에 보존하고 최종 후보와 섞지 않는다.

## 독립 component 검증

모델 checkpoint layer0의 실제 Q/K/V/O BF16 가중치를 사용했다. 입력은 기존 seed를 사용하며 O의 K4096 입력만 seed bytes를 반복·절단해 만들었다. 각 파일의 tensor metadata와 SHA256을 보존했다. 실제 serving activation 전체를 사용한 검사는 아니며, 후속 whole-model gate로 보완한다.

6종 finite input state로 원래 M64–130을 검사해 M128과 동일한 구간을 구했다. Q64, K/V64–93 및129–130, O64 및129–130은 차이가 있어 제외했다. 선택한 merged 전체 정수 M 범위의 6종 입력은 차이가 0이었다. 큰 행 수에서 cuBLAS 기본 heuristic이 바뀌는 문제를 막기 위해 recipe를 고정한다.

CUDA Graph에 동일 연산 64개를 담고 8회 replay해, 기존 요청별 M128 실행과 merged 실행을 비교했다. 3회 교차 순서 측정의 중앙값이며 단위는 µs다. 서버 처리량 개선율로 해석하면 안 된다.

| Projection, merged M512 | 기존 4 × M128 | merged M512 | GPU 시간 감소 |
|---|---:|---:|---:|
| Q | 83.3564 | 64.8104 | 22.25% |
| K | 44.2439 | 25.5879 | 42.17% |
| V | 44.2156 | 25.5782 | 42.15% |
| O | 82.8393 | 67.8891 | 18.05% |

Q의 M256은 41.37→44.11µs로 악화돼 dispatch에서 제외했다. K/V M520 이상도 전체 검증 범위에 포함하지 않아 제외했다.

## 통합 경로와 남은 병목

별도 64 warm / 64 keep NVTX·Nsight 진단에서 C16/C32 모두 Q/K/V/O의 merged Lt 호출을 확인했다. 예를 들어 Q는 M258–511, K/V와 O는 M241–507 등의 실제 호출이 기록됐다. 같은 진단의 v11 기준선에는 이 세 shape의 merged Lt 호출이 없었다. 두 경로 모두 decode Graph가 실행됐다. 소스에 분기가 있다는 사실만으로 fast path가 적용됐다고 판단하지 않았다.

| 진단 항목 | C16 v11→v12 | C32 v11→v12 |
|---|---:|---:|
| 식별된 Q/K/V/O prefill GPU 시간 합 | 184.679→161.723ms, -12.43% | 181.493→154.337ms, -14.96% |
| 전체 kernel 수 | 109,315→104,529, -4.38% | 84,392→79,541, -5.75% |
| 전체 GPU kernel 시간 합 | 1772.638→1762.012ms, -0.60% | 1422.868→1418.910ms, -0.28% |

전체 시간 감소가 작은 이유를 숨기지 않는다. 새 경로는 prefill의 일부 projection을 줄이며, decode 연산과 attention, 남은 요청별 경로는 그대로 실행한다. closed-loop 요청 도착과 batch 구성도 바뀌므로, 대상 projection에서 절약한 시간을 전체 kernel 차이와 일대일로 동일시할 수 없다.

C32 후보 trace의 직접 식별된 Gate/Up은375.740ms(전체 kernel 합의 약26.5%), K/V는173.792ms(12.2%), QK/PV+softmax는163.987ms(11.6%)였다. Gate/Up 중 직접 식별된 prefill은165.406ms이며 decode 쪽도 남아 있다. Graph 안에서 실행된 일부 native GEMM은 NVTX 호출 상관관계가 없어 `other-unattributed`206.030ms로 남겼다. 따라서 O/Down/Q/head category는 완전한 분해가 아니고, 익명 GEMM을 임의로 Gate/Up이나 stock의 특정 연산에 배정하지 않았다.

C32 후보의 GPU kernel union / kernel span 비율은 약74.9%였다. 이것만으로 모든 빈 구간을 CPU 원인으로 확정할 수는 없지만, mixed batch의 launch·host 작업을 줄일 여지가 남아 있음을 보여 준다. `cudaStreamSynchronize` API 시간934.648ms는 GPU 완료 대기와 겹치므로 GPU 시간에 더하지 않는다. DRAM/L2 hardware counter를 수집하지 않았으므로 memory-bound라고 단정하지 않는다.

같은 GPU7에서 stock vLLM의 진단도 새로 수집했다. 같은64 warm / 64 keep 진단에서 stock은32,183 kernels / 846.769ms, Riley 후보는79,541 / 1418.910ms였다. stock의 GPU kernel union/span 비율은87.0%, Riley는74.9%였다. 서로 다른 batch 구성과 출력 경로를 포함한 합계 비교이며, 익명 stock GEMM의 종류를 추정하지 않았다. 정상 서버 처리량 표에는 이 profiler 표본을 섞지 않았다. 실제 stock 시작 로그에서 `FLASH_ATTN`/FlashAttention2 backend 및 `FULL_AND_PIECEWISE` Graph mode 선택도 확인했다.

큰 처리량 개선의 다음 우선순위는 다음과 같다. 아래는 이번에 구현 완료했다고 주장하는 항목이 아니다.

1. **mixed/prefill의 piecewise Graph**: 선형 구간의 shape class, 요청별 native arithmetic family, 출력 mode, stable buffer 주소를 key로 삼고, attention은 eager로 남긴다. 실제 prompt fixture별 Graph cache로 시험 결과만 최적화하지 않는다. cold warmup과 capture 상한을 먼저 정해 측정 중 capture로 TTFT를 악화시키지 않아야 한다. 기존 pure decode full Graph는 유지한다.
2. **decode Gate/Up의 native-family GEMM·epilogue 재설계**: 기존 두 FP32 누적 체인, Gate/Up 각각의 BF16 경계와 SiLU의 BF16 경계를 보존하는 MMA pipeline/epilogue가 필요하다. 표준 merged GEMM을 연결하는 것과 다르다. 실제 Gate/Up GPU 시간과 whole-model byte gate를 함께 비교한다.
3. **async-first executor와 GPU metadata**: persistent CPU state, pinned staging, in-flight GPU state를 분리하고 double buffer와 완료 event를 둔다. embedding failure-before-KV, EOS, 취소, epoch가 바뀐 요청과 KV 회수를 명시적으로 검증한 뒤 적용한다. 단순 동기화 삭제는 하지 않는다.

## 최종 서버 검증 결과

같은 물리 GPU에서 각 concurrency별로 **v11 / v12 / stock vLLM을 각각 5개의 독립 새 프로세스로 측정**했다. 반복별 실행 순서를 바꿨고, 각 arm은64 warm / 192 keep을 사용했다. 정상 성능 표본에는 profiler와 inspection preload가 없다. 모델은 Qwen3-4B BF16, batch token budget576, max sequence1024, KV blocks=C×64, GPU greedy다. stock은 prefix cache off이며 기본 compile/Graph 경로를 유지했다. 기준선에도 후보와 같은 owned-decode Graph 옵션을 넣었다.

각 metric의 5회 중앙값이며, 개별 요청을 모두 합쳐 계산한 percentile이 아니다.

| C | 항목 | v11 기준선 | v12 후보 | 변화 | stock vLLM |
|---:|---|---:|---:|---:|---:|
| 16 | 처리량 (tok/s) | 697.291 | 696.455 | -0.12% | 1125.464 |
| 16 | TTFT p95 (ms) | 94.331 | 88.822 | -5.84% | 83.578 |
| 16 | TTFT p99 (ms) | 191.426 | 180.885 | -5.51% | 110.784 |
| 16 | TPOT p95 (ms) | 21.839 | 21.535 | -1.39% | 13.151 |
| 16 | TPOT p99 (ms) | 23.310 | 22.982 | -1.41% | 13.570 |
| 32 | 처리량 (tok/s) | 897.190 | 914.686 | +1.95% | 1666.432 |
| 32 | TTFT p95 (ms) | 290.554 | 274.099 | -5.66% | 164.942 |
| 32 | TTFT p99 (ms) | 393.571 | 373.219 | -5.17% | 217.771 |
| 32 | TPOT p95 (ms) | 34.803 | 34.414 | -1.12% | 17.657 |
| 32 | TPOT p99 (ms) | 45.937 | 42.157 | -8.23% | 21.450 |

| C | v11 / stock 처리량 | v12 / stock 처리량 |
|---:|---:|---:|
| 16 | 61.96% | 61.88% |
| 32 | 53.84% | 54.89% |

C16에서는 처리량 개선이 확인되지 않았다. C32의 약1.95% 증가도 큰 격차를 해소한 결과로 해석할 수 없다. 서로 다른 시점에 측정한 이전 보고서의 stock 수치를 분모에 쓰지 않았다.

개별 paired 결과도 보존했다. TTFT p95는10개의 before/after pair에서 모두 감소했다. 처리량은 C16에서3/5 pair 증가, C32에서5/5 pair 증가했다. C16 repeat4의 TPOT p95는+2.556%, p99는+4.501% 악화돼 **모든 pair에서2% 이내 tail 비악화라는 기존 점검 기준은 충족하지 못했다**. 중앙값의 개선으로 이 기록을 지우지 않았다.

이 변경은 기존 native-prefill-sm80-v5 SM80 실험 경로에 적용된다. 기존 v11 E 바이너리는 그대로 보존했다. 이번 후보를 C16 처리량 개선이나 전면적인 tail 개선으로 승격하지 않는다. 초기 prefill TTFT 이점과 이 한계를 함께 확인할 수 있는 후보로 남긴다.

### 정확도와 실행 증거

- CPU 회귀519개와 승인된 private `/tmp`의 C02 fixture 재검사4개, 총523개 통과.40개 ignored는 GPU 검증을 대신하지 않으며, 별도 실제 GPU 검사를 수행했다. 네 임시 fixture tree는 제거됐다.
- eager와 Graph 각각104 HF full-vocabulary rows, 그리고 natural-128의 R1–R32 body 검사1024 rows. 총2464 BF16 파일이 원래 native 및 HF bytes와 동일했다. 전체104 case-row×32의 Cartesian 검사는 아니다.
- body Graph62 captures/62 first replays, 일반 all-R 검사51/50. Greedy warmup 뒤 logits mode 전환과 요청 수·context 변경을 포함한다.
- 정상 비교에서 full SSE6042개를 검증했다. Riley4032개는 원래 native 값과 완전히 동일했고, stock2010개는 자기 SSE 재구성 및 prompt count를 검증했다.
- 정상 성능30 arms(2 concurrency×5 repeats×3 engines)의 원시 SSE를 보존했다. Riley는 원래 native oracle의 token/text/finish_reason/usage/prompt IDs와 완전히 같아야 통과했다. stock은 stock SSE 값 자체의 재구성과 prompt token count를 검증했으며, 모든 stock 출력이 HF와 byte-identical이라고 주장하지 않는다.
- 독립 audit은 실제 argv의 binary/runner SHA256, token budget, KV block 수, 응답별 raw SSE, 처리량·분위수 재계산, 종료 후 active/waiting/KV=0, own+holder22GiB 상한 및 foreign GPU PID 부재를 검사했다.
- 첫 후보의 Q cold prepare 누락은 명시적인 NotSupported로 실패했고 GPU 자원은 회수됐다. 수정 후보의 소스·빌드·교사 검사는 별도 root에서 수행했다. vLLM IPC 폴더 누락으로 실패한 첫 normal 시도도 `rejected-ipc-attempt`에 보존했고, 그 시도의 표본은 위30 arms에 섞지 않았다.

### 바이너리 식별

후보 SHA256: `e4a4f4cafe1d005b80e4a0dde9aed249fca3b42f2d721408895e0ce4cdca32c9`.
935개 source file manifest SHA256: `5d12b875983207452cb7acc486922e77aa2f85dfbca357b5621ebf1f49d3a64e`.
소스·바이너리·정상 runner·profiling runner·inspection wrapper는 최종 비교 전에 고정했고, 이후 재확인했다. 패치는 clean Git HEAD가 아닌 immutable v11 E에 겹치는3-file layer다. 다른 dirty/untracked 작업을 보존했다.

## 원자료와 재현

독립 연구 root: `/mnt/efs/nlp/psyche/riley-packed-v12-20261020/research`.
최종 후보 root: `/mnt/efs/nlp/psyche/riley-packed-v12b-20261020`.
GPU는 물리 6/7만 사용하며, 원래 holder PID2027914/2027915 및 start ticks6083215119/6083215121의 CUDA context를 계속 유지한다. finite 작업 때만 예약 메모리를 줄이고 종료 후20GiB로 복구한다. context 중단·재시작·점유 해제는 하지 않는다.


최종 검증은 source935개, CPU523개, full-vocabulary BF16 파일2464개, 정상 성능30 arms, 진단을 포함한 raw SSE6599개를 확인했다. 정상 Riley 서버20개 모두 body Graph warmup이 listen 전에 끝났고, listen 뒤 추가 capture가 없었다. 모든 finite GPU 작업 종료 후3개의 fresh sample에서 원래 holder context와20GiB idle allocation, 다른 GPU PID 부재를 확인했다.

현재 결과는 이 checkpoint·하드웨어·고정 workload의 검증 범위다. 다른 모델이나 긴 context의 성능 향상을 일반화하지 않는다.


로컬 전달물: [v11 위에 적용하는3-file 패치](/Users/psyche/PycharmProjects/riley/benchmarks/candidates/sm80-packed-prefill-v12/candidate.patch), [범위·승격 제한](/Users/psyche/PycharmProjects/riley/benchmarks/candidates/sm80-packed-prefill-v12/README.md), [원시 SSE·audit·연구 증거 ZIP](/Users/psyche/PycharmProjects/riley/benchmarks/evidence/20261020-packed-v12.zip), [ZIP SHA256 receipt](/Users/psyche/PycharmProjects/riley/benchmarks/evidence/20261020-packed-v12-archive-receipt.json).
