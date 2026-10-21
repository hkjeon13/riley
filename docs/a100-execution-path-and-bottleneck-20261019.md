# A100 실행 경로·병목 확인 및 수치 보존 개선

작성일: 2026-10-19 (KST)

## 범위와 실행 위치

아래 2/3번은 최초 실행 기록이다. 현재 실행 대상은 사용자 지시에 따른 **6/7번**이며 최종 결과는 마지막 절을 기준으로 한다.

[조사·적용 계획](inference-library-research-and-plan-20261019.md)의 W0를 수행하고, 현재 경로에서 실제 실행되는 후보만 성능 비교한다. 기존 strict 수치 계약을 유지한다. 초기 GPU 2/3 기록과 사용자 변경 후 GPU 6/7의 최종 W0 측정을 함께 보존한다. 확대 release matrix와 vLLM 성능 동등성 검증은 완료하지 않았다.

- 호스트: `aws-a100` → 실제 `ldccai-a100-02`.
- 실행·모델·스크립트·도구·원시 결과: `/mnt/efs/nlp/psyche` 아래.
- 이번 실행 디렉터리: `/mnt/efs/nlp/psyche/riley-w0-a100-20261019`.
- 물리 GPU 2: `GPU-700c3d44-3302-1900-f8fc-e960b4fa0df0`, A100-SXM4-40GB, Qwen3-4B.
- 물리 GPU 3: `GPU-b7a1c292-6e3a-1bf2-6e6a-25307835853b`, A100-SXM4-40GB, Mistral-7B-Instruct-v0.3.
- 시작 admission에서 두 GPU는 각 4MiB, compute process 없음. 다른 GPU의 기존 작업은 유지했다. 각 arm도 선택 GPU admission·실행 PID·자원 회수를 별도로 확인한다.
- 모델은 기존 EFS checkpoint를 사용하며 새 다운로드나 GPU 대체를 수행하지 않았다.

## 현재 소스와 기준선의 일치

로컬 `164ac43a60bb388995bea3bdddc0a3ab4fcc235c`의 archive를 새 EFS source 디렉터리에 전송했다. 현재 소스 935개 SHA256은 기존 A100 기준선의 고정 manifest와 모두 일치했다.

- 기준 바이너리 SHA256: `8befa8ca193c136449981c24a11e4129a75417d67c4f068814bc9a222dd80a24`.
- 원본: `riley-qkv-gateup-v214-20261019/riley-v214-a100-before`.
- 이번 사본: `riley-w0-a100-20261019/riley-baseline`.
- receipt: 원격 `provenance.json`.

완료된 기존 빌드를 불필요하게 재실행하지 않고, 소스·빌드 기록·바이너리 hash를 결합해 기준선으로 재사용했다. 현재 서버에 설정된 결과가 아니라 이 소스로 직접 시작한 실험 프로세스의 실행 경로를 확인하는 작업이다.

## 확인한 실행 경로

Qwen3-4B의 실제 서버 로그는 다음을 선택했다.

```text
completion=IterationBatch, M=512, T=1024
qkv=owned-hf-separate-qkv-bitpack-v1
output=owned-hf-classic-request-rows-v1
mlp=owned-hf-classic-request-rows-v1
lm_head=owned-hf-classic-output-rows-v1
graph=false
```

여기서 `M=512`는 준비한 실행 용량이다. 모든 projection이 512개 실제 요청을 함께 계산했다는 뜻이 아니다. 실제 cuBLAS argument 계측에서는 decode의 Q/K/V가 각각 다음 shape였다.

| 연산 | 출력 너비 N | 입력 너비 K | 확인한 decode M |
|---|---:|---:|---:|
| Q | 4096 | 2560 | 1 |
| K/V | 1024 | 2560 | 1 |

`N=6144,K=2560`인 합친 QKV 호출은 확인한 C1/C8 경로에서 없었다. 소스도 `owned_hf_eager_bf16 && query_width != hidden`일 때 별도 query plan을 준비하고 Q/K/V를 분리 실행한다.

따라서 기존 v235 후보의 합친 QKV recipe를 통과시킨 component 실험은 현재 strict 경로의 Q/K/V 최적화 근거로 사용할 수 없다. 이전 actual-call 감사에서 해당 Lt/GemmEx 로그가 비어 있었던 현상은 실제 선택 경로와 감사 대상이 다르기 때문임을 뒷받침한다. 수치 불일치 때문에 로그가 비었다고 판정하지 않는다.

직접 근거는 원격 `qwen-qkv-dispatch-proof.json`, `runs/qwen-profile-baseline-v2/*/pid-*-geometries.jsonl` 및 실제 `server.log`다. API argument 관측과 아래 실패한 GPU timing trace의 유효성은 구분한다.

## 정확성 기준선

기존 A100 GPU 3에서 생성해 보존한 HF 참조 15개 프롬프트를 두 모델 모두 실제 HTTP/SSE로 실행했다. 두 모델 각각 생성 token IDs·text 15/15 일치, prompt IDs 일치를 확인했다. 새 C1 eager 응답의 token/text/prompt/finish/usage 전체를 이후 후보 비교의 기준으로 저장했다. Qwen의 이번 서버 실행은 물리 GPU 2이므로, 같은 물리 GPU 2에서 HF를 새로 생성하고 기존 참조와 대조하는 추가 provenance 검사를 준비했다. 이 검사는 아직 실행하지 않았으며 현재 결과를 동일 물리 GPU의 HF 검증 완료로 표현하지 않는다.

참조에는 고정 출력 budget 요청과 자연 EOS 요청이 함께 있다. 이 생성 검사는 모든 입력의 full-vocabulary logits 동등성 증명이 아니다. 기존 teacher/component byte gate를 별도로 유지한다.

원격 참조·결과:

- `runs/qwen-bootstrap-v2/native-reference.json`.
- `runs/mistral-bootstrap-v2/native-reference.json`.
- 각 arm의 `correctness.json`, `recovery.json`, 저장한 SSE 응답.
- 초기 bootstrap 준비 실패는 GPU 실행 전 참조 파일 hash 계산 순서 문제였다. 실패 경로를 보존하고 수정한 v2를 별도 실행했다.

## 프로파일 수집 문제와 처리

기존 CUDA 12.4에 포함된 Nsight Systems 2023.4.4는 C1/C8 데이터의 report 변환에서 `Wrong event order` 오류를 냈다. `.qdstrm`만 남고 `.nsys-rep`/SQLite 생성은 실패했다. 이 실행의 kernel 시간·GPU busy·병목 비율은 유효한 측정으로 채택하지 않는다. HTTP 생성·API geometry 관측은 별도 증거로 보존한다.

NVIDIA 공식 저장소에서 CLI 2026.5.1을 다운로드하고 공개 package SHA256을 검증한 뒤, 이번 EFS 디렉터리 아래에만 unpack했다. 시스템 CUDA/cuBLAS·공유 설치를 교체하지 않았다.

- package SHA256: `61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08`.
- 설치 receipt: `profiler-version.json`.
- 새 계측은 cuBLAS geometry의 NVTX 범위를 추가해 projection과 Attention kernel의 귀속을 구분한다.
- 새 runner는 report/SQLite export 성공을 필수로 검사한다. 계측 실행의 tok/s는 비계측 성능 비교에 사용하지 않는다.

최신 CLI의 독립 M1 Gate/Up 대조 실행은 native BF16 비교와 `.nsys-rep`→SQLite 변환을 통과했다. 전체 HTTP 경로의 프로파일 성공 여부는 별도로 확인한다. 임시 파일도 요청한 EFS 범위에 두어 profiler의 remote-filesystem 경고가 발생하며, 계측 wall time을 비계측 처리량으로 해석하지 않는다.

Nsight Compute의 metric 조회는 `ERR_NVGPUCTRPERM`으로 거부됐다(`ncu-capability.log`/`.exit`). 이 계정에서 DRAM bytes·L2 hit 등의 하드웨어 카운터 수치를 확보했다고 주장하지 않는다. 커널 시간·API argument·동기화·실제 before/after 성능을 우선 근거로 사용한다.

공식 자료: [Nsight Systems 다운로드](https://developer.nvidia.com/nsight-systems/get-started), [설치 가이드](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html).

## 다음 비교 후보와 진행 상태

현재 선택 경로에서 검증 가능한 **SM80 Gate/Up-only v225 후보**를 비교 대상으로 준비했다. 합친 QKV에만 적용되는 v235 확장은 이번 개선의 근거로 사용하지 않는다.

- 후보의 전체 소스 935개와 기준선 대비 7개 변경을 원격에서 다시 hash 검증했다.
- 후보 SHA256: `353d0ce2da8ec99ff15a3e39f16c83a8fd0debdd130b52dcf89f80dc25bd84b3`.
- 원격 `candidate-provenance.json`에 source root·전체 delta·hash를 보존했다.
- 기존 같은-A100 HF teacher 104 full-vocabulary rows/model 및 실제 1..32 Graph row key의 수치·호출 감사는 같은 후보 소스에 대한 기존 증거로 유지한다. 새 HTTP 결과나 새 성능 측정을 대신하지 않는다.
- 비계측 비교는 C1/C8, 두 fresh-process 반복, before/after 순서 교대, arm당 32 warmup+48 측정 요청이다. baseline은 `lt-mlp-v2`, 후보는 `bounded-gateup-sm80-v1`로 실행하며 최초 비교는 양쪽 Graph off다.
- stock vLLM 0.27.1+cu129는 별도 대조군이며 event-spin 수정은 적용하지 않는다. 엔진 결과의 token 일치와 처리량을 각각 기록한다.

작업의 PID·단계는 원격 `screen-state.json`과 `profile2026-state.json`에 기록된다. 상태 파일만으로 생존을 판단하지 않고 `/proc/<pid>` 및 실제 child 상태를 함께 확인한다. 성공·실패가 불명확한 작업을 자동 재시작하지 않는다.

초기 비교 준비 시점에는 후보를 runtime에 적용하지 않았다. 이후의 opt-in 통합 결정과 새 빌드 검증 상태는 마지막 절에 기록한다.

## 비계측 중간 결과와 현재 제약

| 모델/동시성 | 기준선 tok/s | Gate/Up 후보 tok/s | 해석 |
|---|---:|---:|---|
| Qwen3-4B C1 | 84.79 | 85.36 | 두 fresh-process 반복의 중앙값. 실질적으로 비슷하며 속도 개선 확정 아님 |
| Qwen3-4B C8 | 미완료 | 151.56 | 후보 한 번만 완료. 짝을 이루는 기준선이 없어 개선율 계산·채택 불가 |

C1 측정 응답은 baseline/candidate 각각 96개, C8 후보는 48개이며 모두 HF 생성 token열 및 저장한 eager 응답과 일치했다. C8 후보의 45개 별도 correctness 응답도 통과했다. 완료된 해당 arm들의 보존 NVML 샘플에는 외부 GPU PID가 없었다. Profiler를 켠 C8 기준선 수치를 이 표의 비계측 기준선 대신 사용하지 않는다.

2026-10-19 **18:24:44 KST**부터 다음 C8 기준선 시작 구간에 `wckim`의 vLLM TP4 작업이 관측됐다. 이 arm은 측정 요청을 실행하지 못했으며 비교에서 제외한다. 이후 GPU 2/3은 각각 약 37GiB를 사용했고 다른 작업은 중단하지 않았다. 자신의 실험 프로세스와 후속 controller는 terminal 상태임을 확인했다. 현재 전체 모델의 유효한 최신 profile, 완료된 C8 대조군, Mistral before/after 및 stock vLLM 대조군은 남아 있다.

이 과정에서 측정 harness가 admission 이후 등장한 외부 PID의 `/proc/<pid>/maps`를 읽으려다 권한 오류를 내는 문제가 드러났다. 후속 runner는 외부 PID를 먼저 식별하고, 관측 실패를 정상 측정으로 처리하지 않으며, 실행 중 점유 변경 시 확인된 자신의 프로세스만 종료하도록 강화했다. 실제 점유 상태에서 admission guard가 exit 75를 반환하고 지정한 시험 child를 실행하지 않은 것도 확인했다.

재개 준비물은 원격 `run-resume-w0.py`, `measure-w0-safe.py`, `measure-w0-profile2026-safe.py`, `guard-admission.py`다. 아직 재개 controller를 시작하지 않았다. 완료된 C1을 다시 돌리는 대신, 빈 검증 범위와 점유로 끊긴 C8 비교를 새 고유 실험 ID로 수행하도록 준비했다.

컴팩트 증거는 [`benchmarks/evidence/20261019-w0-a100/`](../benchmarks/evidence/20261019-w0-a100/), 원시 결과·실패 자료·스크립트는 원격 실행 디렉터리 및 로컬 ignored `artifacts/a100-w0-20261019/`에 보존한다. `checkpoint-summary.json`의 상태는 부분 진척이며 성능 개선 완료 또는 후보 채택이 아니다.

## 사용자 지정 변경: GPU 6/7 및 테스트 간 지속 홀드

이후 사용자 지시로 재개 대상은 GPU **6/7**로 변경했다. 위 GPU 2/3 결과는 이전 cohort 기록이다. 새 원격 root는 `/mnt/efs/nlp/psyche/riley-w0-a100-gpu67-20261019`이며 GPU UUID는 다음과 같다.

- 6: `GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7`.
- 7: `GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0`.

테스트 사이에 점유가 풀리지 않도록 각 GPU에 지속 CUDA holder를 시작했다. 대기 시 각 20GiB를 예약하며 테스트 중에는 예약만 64MiB로 줄이고 context/PID는 유지한다. 6번의 작은 `fresh-demand` 작업은 사용자 추가 지시로 공존을 허용했으며 건드리지 않았다. 각 GPU에 20GiB 예약을 완료했고, 두 GPU 모두 대기→테스트→대기 전환의 PID/start ticks 불변을 실제 확인했다. GPU compute spin은 없다. 명시적인 해제 지시 전까지 holder를 종료하지 않는다.

실행 코드·사용법: [`benchmarks/tools/gpu_hold/`](../benchmarks/tools/gpu_hold/README.md). 원격 `leases-v2/gpu6.status.json`, `gpu7.status.json`이 현재 heartbeat이며 holder PID는 각각 `2027914`, `2027915`다. 전환 증거는 `holder-transition-proof.json`에 보존했다. 이 확인은 홀드 전환 검증이며 새 GPU cohort의 모델 성능 측정 완료를 의미하지 않는다.

### GPU 6/7의 유효한 전체 서버 프로파일

새 물리 GPU에서 HF 참조를 각각 생성하고 pinned 모델 전체 파일 hash를 검증했다. Qwen3-4B/GPU6 및 Mistral7B/GPU7 모두 15개 생성 참조가 기존 GPU3 참조와 일치했다. 이어 각 장치의 새 eager 서버 응답 15개가 HF token/text/prompt와 일치했고, 이후 비교용 native token/text/prompt/finish/usage 참조를 저장했다.

Nsight Systems 2026.5.1로 두 모델의 C1/C8 전체 HTTP 구간을 수집하고 `.nsys-rep` 및 SQLite export exit 0을 확인했다. Graph off, iteration-batch completion, baseline `lt-mlp-v2`, 32 warmup 뒤 동일한 고정-budget 12개 요청(출력 240 tokens)의 **prefill+decode** 구간이다. 아래 수치는 계측용 커널 시간이며 비계측 tok/s가 아니다.

| 모델 | C | 커널 시간 합 ms | 커널 span ms | 커널 busy/span | Gate/Up ms (합 대비) | Down ms (합 대비) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 1 | 2348.12 | 2937.70 | 79.9% | 766.96 (32.7%) | 390.73 (16.6%) |
| Qwen3-4B | 8 | 2015.68 | 2282.96 | 88.3% | 754.23 (37.4%) | 389.26 (19.3%) |
| Mistral7B | 1 | 3564.77 | 4110.13 | 86.7% | 1586.83 (44.5%) | 730.21 (20.5%) |
| Mistral7B | 8 | 3295.37 | 3506.19 | 94.0% | 1588.77 (48.2%) | 730.29 (22.2%) |

NVTX로 cuBLAS 호출의 N/M/K를 감싸고 CUDA correlation ID로 kernel에 연결했다. Mistral의 Q와 O는 같은 geometry이므로 합산하고 추측으로 분리하지 않았다. C8 Attention QK/PV는 Qwen 약 82.70ms, Mistral 약 80.92ms였다. KV gather와 softmax를 고려해도 이 짧은 입력 workload에서는 Gate/Up, 다음 Down이 주요 개선 대상이다. DRAM/L2 카운터는 권한이 없어 메모리 대역폭 실측값으로 표현하지 않는다.

A100의 주요 kernel에는 `ampere_bf16_s16816gemm...sliced1x2`와 CUTLASS Tensor Core GEMM도 있다. 과거 RTX4090 자료의 GEMV 비율을 이번에 전용하지 않고, **요청마다 M=1 호출을 반복하는 단위와 배치 시 weight reuse 부족**을 확인한다. C1과 C8에서 같은 출력 240 tokens에 대한 Gate/Up 시간이 거의 같은 것이 bounded Gate/Up 후보를 우선하는 직접 근거다.

Qwen C8 기준선의 복사는 H2D 329회/191,547,546 bytes/17.70ms, D2H 98회/72,930,848 bytes/6.53ms였다. CPU의 CUDA API 대기 시간은 GPU kernel/copy 시간과 겹칠 수 있어 wall time 내역에 합산하지 않는다.

Qwen 후보의 첫 profile은 시작 중 다른 GPU PID가 나타나 중지했으며 측정 결과로 채택하지 않았다. 사용자가 공존을 허용한 작은 `fresh-demand` 작업은 동일 UID·고정 부모 PID/start ticks·최대 2GiB로 제한해 해당 workflow의 자식과 GPU helper를 후속 controller에서 인식한다. 공존이 관측된 arm은 생 NVML 기록에서 식별하고 깨끗한 성능 채택 근거와 분리한다. 원래 실패 자료와 종료 상태를 보존했고 완료된 기준선 profile은 재실행하지 않았다.

### Gate/Up 후보의 실제 커널 비용 감소

GPU7의 Mistral C8 후보에서 실제 `GemmEx N14336/K4096`의 M=2..8 호출을 확인했다. 같은 12개 요청/출력 240 tokens에서 Gate/Up 커널 시간은 **1588.77 → 465.38ms**, 전체 커널 합은 **3295.37 → 2170.37ms**, kernel span은 **3506.19 → 2364.69ms**로 감소했다. Down은 **730.29 → 731.06ms**로 사실상 그대로여서, Gate/Up 병합의 효과와 다음 병목을 구분할 수 있다. 후보의 45개 correctness 응답 및 12개 계측 응답 모두 저장한 eager/HF 생성 참조와 일치했다. 해당 baseline/candidate arm의 NVML에는 holder와 각 arm의 검증된 프로세스만 있었다.

GPU6 Qwen 후보에서도 Gate/Up **754.23 → 201.92ms**, 전체 커널 합 **2015.68 → 1464.40ms**가 관측됐다. 그러나 이 후보 arm에는 사용자가 허용한 작은 `fresh-demand` 작업이 공존했으므로 깨끗한 성능 채택 증거로 사용하지 않는다. 원시 샘플과 공존 감사 결과를 보존했다.

수치·커널 감소와 별도로 비계측 C1/C8 AB/BA 두 fresh-process 반복 및 stock vLLM 비교를 진행한다. 완료 전 처리량 개선율이나 최종 채택을 선언하지 않는다. Gate/Up-only 패치는 [별도 후보](../benchmarks/candidates/sm80-bounded-gateup-v225/README.md)로 분리했으며 `git apply --check`를 통과했다. 이후 아래 비계측 비교를 확인하고 opt-in으로 통합했다.


### 비계측 결과와 opt-in 통합

GPU7 Mistral C8의 AB/BA 두 fresh-process 비교는 각각 **1.5479배, 1.5524배**였다. 중앙값은 **70.38 → 109.10 tok/s (+55.0%)**이며, 각 arm 32 warmup+48 측정 요청이다. 두 반복 모두 TTFT·응답 평균 TPOT·E2E의 p50/p95/p99가 개선됐다. p99 중앙값은 다음과 같다.

| 항목 | 기준선 ms | Gate/Up 후보 ms |
|---|---:|---:|
| TTFT p99 | 359.54 | 314.81 |
| 응답 평균 TPOT p99 | 112.67 | 73.66 |
| E2E p99 | 3645.78 | 2359.55 |

저장 응답의 native token/text/prompt/finish/usage를 eager 참조와 독립적으로 재대조하고, tok/s를 원시 시작·종료 시각과 실제 output token 수로 재계산했다. 완료된 해당 arm의 NVML에서 외부 작업은 없었으며 22GiB 예산과 active/waiting/KV 회수를 통과했다. 작은 finite sample의 p99 기술통계이며 확대 release matrix를 대신하지 않는다.

같은 GPU7 stock vLLM 0.27.1+cu129는 C1 **79.00 tok/s**, C8 **436.32 tok/s**였다(각 2회). stock 기본 compile/Graph를 사용했으며 native Graph off 비교와 구분한다. 고정-budget 측정의 output tokens는 arm당 960으로 같았다. Mistral C8 후보/stock 처리량 비율은 약 **0.250**으로, 개선 뒤에도 격차가 크다. stock 생성 token 일치와 text 일치는 별도로 기록하며 Riley의 strict 수치 계약을 stock에 강제한 결과가 아니다. stock의 Graph capture 여부는 native 전용 로그 카운터로 입증하지 않는다.

GPU6 Qwen C8의 **110.33 → 152.19 tok/s**는 작은 외부 작업이 공존한 진단값으로 보존한다. 해당 cohort의 C1 약 3% 저하도 최종 GPU7 단독 결과와 구분한다. 새 독립 빌드로 GPU7에서 다시 수행한 Qwen AB/BA 두 반복은 C1 **84.92 → 84.71 tok/s**, C8 **109.81 → 152.35 tok/s (+38.7%)**였다. C8 paired 비율은 1.3906배와 1.3842배다. 모든 arm에서 native 전체 값과 HF 생성 token이 일치했고 외부 GPU PID가 없었다.

Qwen C8의 응답 평균 TPOT p99는 **71.22 → 50.84ms**, E2E p99는 **2311.60 → 1644.65ms**, TTFT p99는 **194.00 → 158.63ms**였다. 다만 **TTFT p95는 153.70 → 157.91ms로 약 2.7% 악화**했고 두 반복 모두 약 2.7~2.8% 저하했다. C1 tail도 소폭 악화했다. 따라서 처리량 개선과 strict 수치 통과를 확인한 **제한적 opt-in 후보**이며, 모든 latency 무회귀 gate 통과나 release 채택을 선언하지 않는다.

Gate/Up 전용 7개 파일을 **명시적 `bounded-gateup-sm80-v1` opt-in**으로 통합하고 기존 기본값은 유지했다. merged QKV recipe와 실패한 Qwen Down 배치는 포함하지 않았다. 원래 mode로 되돌리면 해당 후보를 사용하지 않는다.

새 독립 빌드의 SHA256은 `844dde2bac465c19f7540c6546575808a6389466ffed9813915c345be2cd4391`이며 manifest 935개 파일이 현재 작업 트리와 모두 일치한다. 새 CPU 회귀 검사는 EFS에서 514개가 통과했다. 기존 C02 audit fixture 4개는 `/mnt/efs/nlp`의 다른 UID 소유 경로 때문에 실패했고, 사용자가 허용한 4개 테스트의 임시파일만 `/tmp`로 옮겨 원래 빌드와 같은 `umask 077`로 재검사했다. 모두 통과해 총 518개 통과, 40개 기존 ignore다. 해당 임시폴더는 삭제됐고 원래 실패 로그도 보존했다. 새 빌드의 Mistral C8 serving은 109.11 tok/s와 전체 응답 일치를 확인했다. Qwen의 GPU7 단독 before/after 비교도 위 수치와 strict 응답 일치를 확인했다. stock vLLM 대조군도 완료했고 아래에 별도 집계했다.


### 다음 단계의 우선순위와 미완료 범위

1. **latency 무회귀 확인:** Qwen의 C8 TTFT p95 및 C1 tail 원인을 먼저 분리한다. 이번 2회·48요청 screen은 5회·192요청 C1/C8/C16/C32 확대 release matrix를 대신하지 않는다.
2. **Down projection:** Mistral 후보 profile에서 Down은 약 731ms로 남은 가장 큰 선형 연산이다. 기존 real-weight component eligibility 검사에서는 Mistral Down의 M=1..16만 byte-exact이고 M=17은 실패했다. 변경 시 16-row 상한부터 검토하되 full-vocabulary teacher·변경 입력·실제 Graph key·HTTP 검증을 새로 통과해야 한다. 이번 패치에는 포함하지 않았다.
3. **선택 경로 Q/O:** Qwen Q 및 Mistral Q/O는 기존 component 검사에서 M=1..32를 통과했지만, Qwen K는 M=2, V/O는 M=4, Down은 M=2에서 이미 달랐다. 합친 QKV의 별도 recipe가 있다는 이유로 현재 separate-QKV 경로를 바꾸지 않는다. component 성공은 전체 모델 채택 근거가 아니다.
4. **Attention/Graph/overlap:** 남은 실제 profile 비용에 따라 이어간다. 짧은 입력의 이번 결과를 장문 Attention workload로 일반화하지 않는다. strict BF16 경계와 reduction 순서는 유지한다.

하드웨어 카운터는 `RmProfilingAdminOnly=1`이며 비대화형 sudo 권한이 없어 확보하지 못했다. 전역 드라이버 설정을 변경하지 않았고 DRAM traffic/L2 hit 실측값은 없다. 이번 W0 근거는 유효한 Nsight Systems kernel geometry/time, 동일 작업의 비계측 응답, 전체 출력 값 대조다. vLLM과의 성능 동등성, 전체 latency gate 및 기본값 승격은 남은 작업이다.


### 최종 stock 대조군 및 종료 상태

| 모델 | C | 기준선 tok/s | Gate/Up 후보 tok/s | stock vLLM tok/s | 후보/stock |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B | 1 | 84.92 | 84.71 | 115.70 | 0.732 |
| Qwen3-4B | 8 | 109.81 | 152.35 | 656.92 | 0.232 |
| Mistral7B | 1 | 60.84 | 61.41 | 79.00 | 0.777 |
| Mistral7B | 8 | 70.38 | 109.10 | 436.32 | 0.250 |

각 값은 GPU7의 두 fresh-process arm 중앙값이며 32 warmup + 48 측정 요청/arm, 측정 output tokens는 모두 960이다. Qwen 후보는 새 독립 빌드이며 Mistral의 두 쌍은 동일 소스의 retained 후보 바이너리다. 새 빌드 Mistral의 별도 C8 serving 검사는 109.11 tok/s였다. native는 Graph off, stock은 기본 compile/Graph라서 같은 kernel 실행 정책 비교로 표현하지 않는다. stock Qwen도 HF 생성과 token이 다른 사례가 있으며 각 arm의 일치 수는 receipt에 별도로 보존한다. 수치 계약과 엔진 처리량 대조는 구분한다.

완료된 비계측 32개 arm의 저장 응답·처리량 원시 시각·자원 기록을 독립 재대조했다(여기에는 별도 표시한 GPU6 공존 진단 arm도 포함된다). 새 GPU7 단독 Qwen/Mistral 비교는 모두 외부 GPU PID 없이 22GiB 예산을 통과했다. 최종 `clean7-state.json`은 complete이며 GPU6/7에는 각각 기존 holder PID 2027914/2027915만 남아 대기 20GiB 예약을 유지한다. 원시 NVML 점유는 context 포함 각각 약 20,894MiB다. 자동 해제하지 않는다.

W0 실행 경로 확인·병목 계측·strict 수치를 유지한 제한적 개선 후보 구현·검증은 완료했다. Qwen tail 무회귀와 확대 release matrix 및 vLLM parity는 완료하지 않았다. 최종 compact receipt는 [GPU6/7 evidence](../benchmarks/evidence/20261019-w0-a100/gpu67/), 홀드 종료 상태는 그 안의 `final-hold-receipt.json`에 보존한다.


후속 Qwen TTFT p95 원인 조사와 같은 바이너리의 GPU greedy opt-in 비교는 [별도 보고서](qwen-ttft-p95-investigation-20261019.md)에 기록했다. 이전 48-request screen과 후속 96-request screen을 합산하지 않는다.
