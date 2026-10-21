# Qwen C16/C32 확대 검증과 native K/V 개선

2026-10-19 KST. Qwen3-4B, A100-SXM4-40GB 단일 GPU 기준. 기존 HF/native 수치 기준을 유지하면서 C16/C32로 검증을 확대하고, K/V projection의 native 연산 순서를 유지하는 mode8 `native-kv-sm80-v3`를 작업 트리에 통합했다. 기본값은 유지한다.

후속 결과: [C16 tail 회복 및 native-equivalent prefill mode10](qwen-c16-tail-and-attention-20261019.md). 아래 표는 이전 mode7→8 기록이며 새 결과와 합치지 않는다.

## 최종 추가 개선: mode7 대비 mode8

각 동시성에서 mode7/mode8/stock vLLM을 **같은 물리 GPU와 CPU affinity**로 대조한 3 fresh-process 반복의 중앙값이다. arm당64 warmup +192 측정 요청, 각3840출력tokens. C16은 GPU6/CPU24..31, C32는 GPU7/CPU32..39다. 서로 다른 GPU의 수치를 합치거나 새 mode의 C16→C32 scaling으로 해석하지 않는다.

| 동시성 / GPU | mode7 tok/s | mode8 tok/s | 추가 처리량 | stock vLLM tok/s | TTFT p95 ms: mode7 → mode8 |
|---|---:|---:|---:|---:|---:|
| C16 / GPU6 | 304.7 | **341.5** | **+12.1%** | 1138.8 | 139.1 → 140.9 |
| C32 / GPU7 | 318.6 | **363.9** | **+14.2%** | 1687.5 | 523.6 → 474.7 |

C16의 세 처리량 비율은1.1194/1.1200/1.1222배, C32는1.1455/1.1455/1.1400배였다. 응답 E2E p99 중앙값은 C16 1686.2→1505.9ms, C32 3482.1→3096.7ms였다.

**전체 지연시간 비회귀 gate를 통과했다고 선언하지 않는다.** C16 TTFT p95는 중앙값이139.1→140.9ms(+1.3%)로 늘었고, 두 반복에서 각각+1.3%/+1.7%였다. C32의 TTFT/응답 평균 TPOT/E2E p50/p95/p99는 세 반복 모두 개선됐다. C1/C8 별도2회/96요청 screen에서는 C8 처리량258.8→285.8tok/s(+10.5%), TTFT p95 97.9→95.8ms였다. C1 처리량86.9→85.7tok/s(-1.4%), C8 한 반복의 TTFT p99 +7.4%도 보존한다. mode8은 명시적 선택 기능이며 전역 기본값 승격이나 모든 tail의 무회귀를 주장하지 않는다. 여기의 TPOT는 응답별 평균 TPOT의 분포이고 token ITL이 아니다.

## 요청한 mode7 C16/C32 5회 확대 검증

기존 [mode7 보고서](qwen-sm80-projection-batching-v2-20261019.md)의 C1/C8 screen을 C16/C32 각각5회/192측정요청으로 확대했다. 이 cohort는 모두 GPU7/CPU32..39이며 mode6 Gate/Up, mode7 projection, stock vLLM을 비교한다. 위 mode8의 별도3회 cohort와 합산하지 않았다.

| 동시성 | mode6 tok/s | mode7 tok/s | stock vLLM tok/s | TTFT p95 ms: mode6 → mode7 |
|---|---:|---:|---:|---:|
| C16 | 171.4 | 302.4 | 1177.5 | 204.2 → 141.6 |
| C32 | 174.1 | 319.8 | 1698.6 | 724.8 → 524.9 |

mode7의 처리량 개선은 C16 약76.5%, C32 약83.7%였다. 이5회 확대 검증은 완료했고, 최종 mode8의3회 검증 및 C1/C8의2회 screen과 범위를 구분한다.

## vLLM과의 차이, 실제 줄인 비용

설치된 stock vLLM0.27.1+cu129의 실제 Qwen3 경로는 `QKVParallelLinear`, Qwen2MLP의 `MergedColumnParallelLinear`/`SiluAndMul`을 사용한다. 시작 로그에서 **FLASH_ATTN**, V1 engine, compile/Graph 경로를 확인했다. 원본 설치 파일 SHA와 시작 로그는 `installed-vllm-path-proof.json`에 보존했다. 일반 설계 참고는 [CUDA Graph](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/), [PagedAttention](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html)다. 실제 설치 버전의 선택 경로는 별도 로그를 기준으로 한다.

동일 GPU6의 이전64요청/1280출력token 프로파일에서 mode7 C32는422433kernels/3763.0ms kernel 합, stock vLLM은32362kernels/823.0ms였다. Riley의 요청별 K/V, Attention 및 작은 projection batch가 많은 호출로 남는다. Riley도 Tensor Core GEMM을 사용하며, 차이를 단순히 Tensor Core 사용 여부로 설명하지 않는다.

아래는 **GPU7**, 같은64요청/1280출력token에서 mode7과 최종 mode8을 직접 계측한 결과다. Nsight Systems2026.5.1 report/SQLite export는 모두 exit0이며, 계측 처리량은 비계측 성능 표와 합치지 않는다.

| GPU kernel 분류 | mode7 ms | mode8 ms |
|---|---:|---:|
| K/V projection | 691.3 | 228.4 |
| Down | 667.7 | 667.8 |
| Gate/Up | 583.7 | 586.9 |
| KV gather | 460.7 | 460.0 |
| Attention QK/PV | 444.1 | 445.8 |
| 전체 kernel 시간 합 | 3788.1 | 3334.6 |
| GPU kernel span | 4428.3 | 3831.2 |
| kernel 개수 | 422361 | 340073 |

K/V 시간을 약67% 줄였다. Down/Gate-Up과 KV gather/QK/PV는 거의 같고, 이 네 분류가 개선 후 kernel 시간의 약64.8%를 차지한다. 따라서 K/V만 더 최적화해서 남은 격차 전체를 해결하기는 어렵다. GPU6/7 사이의 profiler 시간을 합치거나 DRAM bytes/L2 hit 하드웨어 카운터를 확보한 것으로 표현하지 않는다.

최종 실제 serving에서 vLLM은 여전히 C16 약3.33배, C32 약4.64배 빠르다. stock vLLM의 전체 수치가 기존 HF/native oracle와 같은 것은 아니며 agreement를 따로 기록했다. 여기의 정확성은 기존 수치·생성 일치 계약이고 일반 과제 정답률 평가와 구분한다. Riley CUDA12.4/cuBLAS120405와 stock wheel cu129의 버전 차이도 있어 격차 전부를 하나의 요인으로 환산하지 않는다.

## 구현과 수치 검증

native K/V(N1024,K2560)의 SASS와 CUPTI runtime callback에서 K2560/N1024/lda2560/chunk64/stride32를 확인했다. lane별 `k=lane+32*j`의80 FFMA, descending32 F32 합산, 원래 BF16 반올림을 유지했다. 루프를16회 전개하고 실제 행2..4에는2행,5..8에는4행,9..32에는8행씩 weight를 재사용한다. singleton과 multi-token prefill extent는 원래 경로를 유지한다. 자체 device buffer/workspace를 추가하지 않고 기존 plan/stream/command-batch use ledger를 이용한다.

- 독립 component: unroll/reuse 변형 **2880비교/48660480 BF16값**이 native M=1과 일치했다. 실제 선택한 unroll16/reuse2·4·8은576비교/9732096값이며 모든 실제 행1..32와 실제·0·permutation/scale 입력을 포함한다.
- 전체 모델: **HF full-vocabulary104 teacher rows**, C1 eager와 실제 Graph key1..32에서 각각208개 BF16 원시 파일이 기존 native 및 HF와 byte-exact로 일치했다. Graph stderr에서2304capture/2304첫 replay를 독립 확인했다. 모든104행×모든shape의 Cartesian 전수 검사라는 뜻은 아니다.
- CPU 회귀 **519통과/기존40ignore**. 기본 EFS 실행의 원래 exit101 및 실패 로그를 보존한다. 기존 승인 C02 테스트4개의 fixture만 private `/tmp`로 옮겨 다시 통과했고 임시 트리를 삭제했다. 보안 검증 조건을 완화하지 않았다.
- 합계68개 accepted/control/diagnostic arm의 **13442개 raw SSE 값·timing·예산·KV회수**를 독립 재생했다. Riley9972개 값이 기존 token/text/prompt/finish/usage와 모두 같았고, 그중 최종 mode8 바이너리의 응답은1954개다. 전체 합계를 최종 바이너리만의 요청 수나 vLLM의 HF 전부 일치로 표현하지 않는다. stock의 prompt token 수와 agreement flag도 독립 확인했다.
- 매 arm에서 own+holder22GiB 예산, 외부 GPU PID 없음, 종료 요청 후 active/wait/KV0, 취소 후 retry의 값·allocation 회수를 확인했다. 원래 EOS와 neutral greedy 설정을 유지했다.

지원 범위는 Qwen3-4B geometry(2560,32,8,128,9728), SM80/108SM, CUDA12040/cuBLAS-Lt120405, 변경 없는 workspace/MPS 설정, 최대32sequences다. mode0..7과 기본값의 산술을 유지한다. Mistral 및 다른 GPU의 mode8 지원을 선언하지 않는다.

## 실패 후보와 증거 경계

contiguous80/128 누적 후보는 변형 입력에서 BF16 차이를 검출해 제외했다. 첫 strided32/loop1 후보는 전체 수치 검증을 통과했지만 C8이 약260.8→250.9tok/s로 느려져 채택하지 않았다. 해당 소스·바이너리·실패/성공 원시는 EFS에 보존하고 최종 unroll16 후보와 분리했다.

최초 확대 matrix의 vLLM startup은 IPC 디렉터리 부재로 실패했다. 완료한 C16/r0 Riley arm을 보존하고 EFS의 ipc6/ipc7을 준비한 뒤 실패 arm부터 새 root로 이어갔다. 완료 arm을 반복하거나 실패 로그를 성공으로 덮지 않았다. 단일 native-call probe의 종료 시점 감시 실패도 별도 실패 receipt로 남기고 채택 근거에서 제외했다.

owned-HF `/v1/config`는503 `config_unavailable`을 반환했다. 이를 성공한 설정 조회로 표현하지 않고 실제 argv, 실행파일 SHA, resolved startup 로그, GPU kernel/NVTX를 결합해 경로를 확인했다. 실제 생성 API와 `/metrics`는 별도로 검증했다.

## 다음 개선의 우선순위와 통과 조건

1. **Native 순서를 보존한 paged GQA**: 현재 `hf_kv_gather`와 요청별 QK/PV 제출을 줄인다. SM80의 실제 T별 QK/PV family를 component로 확인하고, 원래 BF16 score/probability 반올림과 softmax 순서를 유지한다. 기존 SM89 전용 wide-chunk guard를 그대로 SM80으로 넓히지 않는다. 미검증 shape는 기존 경로를 사용한다.
2. **Down/Gate-Up의 더 큰 실제 row 재사용**: C32에서도 Down≤4, Gate/Up≤16으로 나눠 제출하는 비용이 남는다. native M=1 reduction과 일치하는 recipe를 먼저 검증한다. 빠른 GEMM을 쓰기 위해 tolerance를 늘리거나 임의 padding을 도입하지 않는다.
3. **Prefill/Graph 확장**: 원래 prefill extent를 지키는 request grouping을 검토한다. Attention의 context 의존 host 제출을 device metadata로 바꾼 뒤 whole-step Graph를 검토한다. 현재 layer segment Graph와 같은 것으로 간주하지 않는다.

각 후보는 component byte equality → HF104/full vocabulary와 실제 Graph1..32 → 원래 EOS/취소/KV회수/22GiB 검사 → 동일 GPU C1/C8/C16/C32 확대 비교 순으로 진행한다. 기본값 승격에는 전체 tail 비회귀 기준을 별도로 만족해야 한다. 예상 speedup을 검증 결과처럼 제시하지 않는다.

## 소스·실행 위치·사용

최종 바이너리 SHA256: `cc521448e19978e148d1e04d1c00b7a1eec524683316fce8c3ebc0bbab1cf277`. 현재935개 소스 파일이 최종 manifest와 모두 일치한다. [mode8 patch/사용 조건](../benchmarks/candidates/sm80-native-kv-v3/README.md), [compact evidence](../benchmarks/evidence/20261019-c16-c32-native-kv/)를 보존했다. commit/push는 수행하지 않았다.

- 원래5회 확대 및 폐기 후보: `/mnt/efs/nlp/psyche/riley-expand-c32-20261019`.
- 최종 source/target/바이너리/teacher/benchmark/profiler: `/mnt/efs/nlp/psyche/riley-native-kv-unroll-20261019`.
- 기준 mode7: `/mnt/efs/nlp/psyche/riley-sm80-projections-v2-20261019/riley-projections-v2-qualified`, SHA `d339fe5d39b04983f09cbb36dce32b22c4ef45d77db9b2a8604ab1a38e22f884`.
- 최초 native oracle SHA `8befa8ca193c136449981c24a11e4129a75417d67c4f068814bc9a222dd80a24`와 mode6 serving SHA `844dde2bac465c19f7540c6546575808a6389466ffed9813915c345be2cd4391`을 구분한다.

기존 qualified owned-HF 실행 인자에 아래를 명시한다. 실제 GPU 사용은 기존 holder `control.py run`을 통해 수행한다.

```text
--execution-completion iteration-batch
--owned-decode-batch native-kv-sm80-v3
--sampling-backend gpu-greedy
--max-active-sequences 32
```

rollback 옵션은 `--owned-decode-batch bounded-projections-sm80-v2`다. 모델·스크립트·benchmark·원시 자료는 EFS에 두고 무거운 SSE/SQLite/Nsight/BF16/바이너리는 EFS에 남겼다.

모든 자신의 GPU 실행을 끝낸 뒤 물리 GPU6/7의 **원래 holder PID2027914/2027915와 CUDA context를 유지하며 각20GiB 대기 예약으로 복귀**했다. 최종 NVML에는 해당 두 GPU에 holder만 남았다. 외부 작업을 종료하지 않았고 명시적인 해제 지시 전까지 홀드를 유지한다. `native-kv/final-hold-receipt.json`이 완료 시점의 증거다.
