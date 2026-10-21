# Qwen prefill attention·host 실행 구조 v15C

2026-10-21 KST. **최종 후보를 채택했다.** 기존 native/HF literal full-vocabulary BF16 기준을 유지하면서 prefill attention의 데이터 이동·softmax 실행 구조와 host resource 등록·iteration 완료 경계를 변경했다. C16/C32 각각 before/after/stock 5회 새 프로세스 비교에서 처리량은 **+4.14% / +5.16%**, TTFT p95는 **-6.83% / -9.66%**였다. 이번 10개 before/after pair의 TTFT·TPOT·E2E p95/p99에서2% 초과 악화 플래그는0이다. 계획의 첫 묶음 +10% 목표나 vLLM parity에는 미달한다.

## 효과가 작았던 원인과 확인한 병목

[v14](qwen-mixed-attention-structural-v14-20261021.md)는 처리량 +0.76%/-0.27%였고 tail 기준도 실패했다. 작은 단독 연산의 이득을 mixed pipeline 전체로 확대해석할 수 없었다. two-pass mixed 실행은 추가 완료 경계를 만들었고, R24~32 attention 후보의 구성요소 이득도 실제 batch 분포에서 유지되지 않았다. v14 변경은 이번 코드에 합치지 않았다.

Graph node를 실제 layer·residual·PV·SiLU 경계와 함께 재분류한 기존 v13 진단에서 Gate/Up+Down은 GPU kernel 시간 합의 약43.0%(C16)/41.0%(C32)였다. prefill attention에는32개 query head로 확장한 KV staging, Q gather/output scatter, 별도 FP32 mask·softmax 임시 tensor와 BF16 cast가 남아 있었다. host에는 operand마다1024-entry lease ledger를 재검색하는 작업과, 이미 CPU에서 검증한 embedding token의 device report를 layer0 전에 기다리는 경계가 있었다.

이번에는 이 구조를 변경했다. 다만 MLP GEMM과 decode KV/attention은 여전히 큰 부분을 차지하며, prefill/host만으로 전체 차이를 해소하지 못했다. CPU API 대기에는 GPU queue 대기가 겹치므로 순수 CPU 시간으로 합산하지 않는다. DRAM/L2/SM hardware counter 없이 memory-bound로 확정하지 않는다.

## 채택한 구현과 적용 범위

- SM80에서 cold-qualified attention plan, M>1/T<=2048/QH32/KVH8/D128이고 필요한 span이16-byte 정렬됐을 때8개 KV head만 compact staging한다. 기존 K scratch의 남는 부분에 device pointer table을 만들고, Q/context는 row-major 상태로 pointer-batched QK/PV GEMM을 실행한다. per-head GEMM M/N/K와 native BF16 경계를 보존한다. Q gather/output scatter와 별도 FP32 scale-mask/cast pass를 제거하고 같은 warp reduction의 causal softmax로 융합했다. 새 device allocation은 없다.
- BF16 query/output의2-byte offset view를 포함해 부적격 geometry·정렬은 기존 경로를 유지한다. [cuBLAS12.4 BF16 batched alignment 계약](https://docs.nvidia.com/cuda/archive/12.4.0/cublas/index.html#cublasgemmbatchedex)에 맞춘 호환성 조건이다. 서버 내부 정렬된 span의 성능 결과를 범용 view 전체에 확대하지 않는다.
- command-batch resource 등록에는2048-slot 고정 hash index를 추가한다. 원래1024-entry ledger가 acquisition/release·capacity·실패 처리의 권위를 유지한다. confirmed GPU completion과 모든 release 후에만 index를 비운다. stream당 cold host16KiB, hot allocation0이다. foreign thread/stream 거부·capacity 초과 시 미획득·4세대 reuse·full-table 종료를 native CPU helper와 실제 GPU 계약 검사로 확인했다.
- mode10 staged lane의 CPU token admission을 유지하고 embedding device report를 iteration 완료 시 검사한다. malformed report는 출력 publish·scheduler commit 전에 실패한다. EOS·취소·KV reservation·pinned staging 재사용 계약은 유지한다. 여러 iteration을 동시에 in-flight로 만드는 변경은 아니다.

새 attention 경로는 `owned-hf-eager-bf16-v1`, `native-prefill-sm80-v5`, `segments-v1`의 기존 명시적 설정에서 적격할 때 사용한다. 모든 모델·기본 실행 옵션을 이 경로로 전환한 결과가 아니다. R2~16 Gate/Up은 기존 native fused kernel이다.

## 정상5회 비교: 개선 전·후·stock

동일 물리 A100 GPU7에서 before/after/stock 순서를 교대하고 매번 새 서버를 시작했다. Qwen3-4B BF16, greedy, budget576, max sequence1024, KV blocks=C×64, prefix cache off. 각 arm은64 warm/192 keep,12 fixed prompts115~130 tokens, balanced8/16/24/32 출력 상한으로 실제3,840 output tokens다. 정상30 arms에는 profiler·inspection preload·HOSTTRACE가 없다. 지표별5회 중앙값이며 percentile을 pooling하지 않는다. 호스트 전체의 CPU/IO까지 전용 격리한 실험은 아니다.

| 지표 | C16 v13B | C16 v15C | C16 stock | C32 v13B | C32 v15C | C32 stock |
|---|---:|---:|---:|---:|---:|---:|
| output tok/s | 717.829 | 747.513 | 1138.307 | 932.064 | 980.137 | 1683.754 |
| TTFT p95 ms | 88.446 | 82.409 | 83.158 | 271.712 | 245.456 | 162.982 |
| TTFT p99 ms | 178.047 | 164.515 | 110.557 | 368.821 | 335.009 | 216.597 |
| TPOT p95 ms | 20.916 | 20.028 | 12.706 | 33.342 | 31.542 | 17.574 |
| TPOT p99 ms | 22.625 | 21.469 | 13.315 | 42.028 | 38.458 | 22.718 |
| E2E p95 ms | 689.518 | 660.452 | 424.361 | 1066.124 | 1008.210 | 554.416 |
| E2E p99 ms | 733.468 | 714.224 | 446.437 | 1235.197 | 1171.555 | 675.360 |

stock 대비 처리량 비율은 C16 **63.06%→65.67%**, C32 **55.36%→58.21%**다. stock은 vLLM0.27.1+cu129/Torch2.13.0+cu129, 실제10개 startup log의 FA2/FULL_AND_PIECEWISE를 확인했다. stock 수치 경로는 Riley의 correctness reference가 아니다.

| 반복 | C16 처리량 변화 | C32 처리량 변화 | 2% 초과 tail 플래그 |
|---:|---:|---:|---|
| 0 | +5.188% | +4.183% | 없음 |
| 1 | +3.886% | +6.784% | 없음 |
| 2 | +4.135% | +5.472% | 없음 |
| 3 | +1.896% | +4.565% | 없음 |
| 4 | +6.100% | +4.997% | 없음 |

TPOT는 `(마지막 토큰 시각-첫 토큰 시각)/(output tokens-1)`의 요청 평균이며 token ITL percentile과 다르다. 정상 runner는 모든 토큰 arrival timestamp를 파일로 저장하지 않았으므로 per-token ITL 개선을 주장하지 않는다. TTFT를 인위적으로 지연시키거나 출력 조건을 바꾸지 않았다.

[v15B](qwen-prefill-host-v15b-20261021.md)의 C32 반복2 TPOT p99+2.3977%는 삭제하거나 C 결과로 덮어쓰지 않는다. 당시 percentile bracket과 request completion 순서가 달랐고 동일 요청의 TTFT/E2E는 개선됐지만 평균 TPOT는 악화했다. 단일 kernel 저하나 TTFT만을 원인으로 확정하지 않았다. C의 추가 fresh5회는 별도 검증이며, alignment guard가 그 tail을 해결했다는 인과 주장은 하지 않는다.

## 실제 경로 검증과 host 근거

정상 측정 뒤 정확히 같은 최종 C binary를 별도 C32/64-keep Nsight로 검사했다. compact prefill gather **2,304=64×36 layers**, pointer-batched GEMM **4,608** NVTX ranges를 확인했다. Q gather/output scatter/scale-mask/BF16 scatter는0이었다. 실제 M/N/K·leading dimension은 [dispatch proof](../benchmarks/evidence/20261021-prefill-host-v15c/actual-prefill-dispatch-proof.json)에 있다. 이 trace를 정상 성능 수치에 섞지 않는다.

B의 별도192-keep 진단은 before→after kernel 수가 C16 291,306→259,301, C32 213,753→184,813이었다. C16 launch152,679→120,167, stream sync810→540이었다. C 최종 normal과 다른 진단이므로 이 수치를 C의 동일 campaign 측정치로 표기하지 않는다. generic context 복원 변경은 관측된 context API 비용이 작아 우선순위에서 제외했다. 자세한 attribution과 resource-index native micro 측정은 B 보고서에 보존한다. 외부 라이브러리의 workspace 재사용·장치 metadata·graph-friendly staging 원리를 적용했지만, 별도 수치 경로의 FlashAttention을 strict native byte gate 검증 없이 대체하지 않았다.

## 정확도·호환성·자원 검증

- 독립 source/빈 target 빌드: **CPU524 pass/40 ignored**. ordinary520과 이미 허용된4개 C02 private `/tmp` fixture 재검사다. EFS 상위 UID 때문에 실패한 최초 검사와 재검사를 구분하며 승인된 임시 트리는 제거했다.
- 원래 native/HF literal full-vocabulary BF16: 기본2,464 files(104 C1+104 all-R shrink+1,024 body rows), mixed1,024+992 files, 합계 **4,480 files**가 일치했다. mixed 역순 slots·prefill output slot 없음·invalid token preflight·hot allocation 불변을 확인했다. dtype/tolerance/토큰·text·usage·finish/EOS 조건을 완화하지 않았다. positive early-EOS 사례를 새로 증명한 검사는 아니다.
- 실제 GPU 계약5개 test functions: memory2, owned command batch1, host runtime1, 새 alignment1. alignment 검사는4개(M,T)×3 finite 상태×5 query/output offset pair×2 실행 방식= **120 cases**다. literal output bytes·view 밖 sentinel 보존·hot CUDA allocation 불변을 통과했다. 오래된 execute_row_groups test 호출에는 현재 grouping selector0을 명시했다.
- C 최초 observer binding 실패는 resource 성공으로 처리하지 않았다. 초기 C1 child는 끝났지만 전체 resource 관측이 없었고 첫 정상 queue는 성능 arm 실행 전에 중단됐다. 실패 상태와 로그를 보존했다. bounded binding retry·cleanup을 보완한 observer v18로 새 이름의5개 teacher 검사와 GPU 계약 검사를 반복했다. 실제 exe/UID/environment marker/start ticks/thread group을 확인하고 전체 장치 메모리를 포함했다. 미확인 live process를 own으로 간주하거나 종료하지 않았다.

정상30 arms+별도 C32 profile1개의 **6,154 full SSE**를 독립 감사했다. Riley는 원래 native/self-SSE·HF tokens/text·usage·finish·회수 기준을 통과했다. stock은 자체 실제 SSE와 지표를 감사했으며 native와 동일 수치 경로라고 주장하지 않는다. 정상·진단의 장치 전체 최고 **18,267MiB**, teacher/GPU test의 최고 **20,347MiB**로22GiB cap 이내였다.

GPU6의 허용된 작은 외부 작업은 보존했고 정상 성능 비교는 깨끗한 GPU7에서만 순차 수행했다. 종료 후3회 fresh 관측에서 원래 holder PID2027914/2027915, start ticks6083215119/6083215121, 각각 idle20GiB allocation과 heartbeat를 확인했다. context 종료·재시작·점유 해제는 하지 않았다. GPU 홀드는 계속 유지한다.

## 재현과 다음 우선순위

- 최종 binary SHA256: `233c3a6a81bad76d65577bacd5837d59e2ebe1dd52a8fde73f3cd07a38cf937a`.
- qualified source937: v13B935 files+resource index header+alignment GPU test. 현재 소스와 모든 manifest hash를 비교한다. observer·CPU ledger helper는 별도 도구이며 server source 집합에 포함되지 않는다.
- [압축 증거](../benchmarks/evidence/20261021-prefill-host-v15c.zip), [감사·통계 index](../benchmarks/evidence/20261021-prefill-host-v15c/README.md). 원본 BF16·ELF·Nsight는 EFS heavy SHA index로 연결한다. 실패한 최초 시도도 포함한다.
- EFS root: `/mnt/efs/nlp/psyche/riley-prefill-host-v15c-20261021`.
- 앞선 큰 실험 checkpoint: A `301c9bf93`, B `eee686456`를 각각 commit·push했다. 최종 C는 검증된 runtime·README·보고서·증거를 함께 commit·push한다.

남은 큰 우선순위는 실제 mixed batch 분포에서의 Gate/Up+Down과 decode KV/attention GPU 작업이다. 다음 변경은 full pipeline 노출 비중과 동일 native BF16 중간 경계를 보존하는 GEMM family부터 다시 측정해야 한다. host의 여러 iteration overlap은 EOS·취소·reservation commit·buffer 수명 계약을 먼저 증명해야 한다. 이번의 제한된 동기화 제거를 완전한 asynchronous scheduler 구현이나 +10% 목표 달성으로 간주하지 않는다.
