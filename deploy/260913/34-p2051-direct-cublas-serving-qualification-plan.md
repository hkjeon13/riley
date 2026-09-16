# P2051 direct cuBLAS 후보의 품질·serving qualification 계획

상태: **계획만**. P10 r3까지의 raw-Q correctness work은 완료됐다. 다음 batch는
`cuda-cublas-gemm-probe` test-only surface를 바로 serving selector로 승격하지 않는다.
먼저 raw Q/K/V와 full-forward의 사전 선언한 quality gate를 통과시킨 뒤에만 opt-in
integration과 vLLM AB/BA 측정으로 간다.

현재 근거는 [P10 결과와 비교표](33-cublaslt-bias-epilogue-qualification.md)의 P10 r3이다.
RTX 4090/SM89에서 direct `cublasGemmEx` default math가 P7 default raw-Q와
`0 / 4,200,448` BF16 byte-exact였고, 반복 실행·allocation accounting·close도 통과했다.
이는 단일 no-bias Q endpoint의 arithmetic correspondence이며 serving 성능 증거가 아니다.

## 유지할 경계

- Rust serving hot path에 Python을 넣지 않는다. Python/HF는 immutable offline oracle과 artifact
  작성에만 사용한다.
- strict default selector와 현재 CUDA Graph, command batch, scheduler, HTTP contract는 P13
  full-forward quality pass 전까지 바꾸지 않는다.
- direct cuBLAS probe의 graph/batch rejection, same-context/span/non-overlap guards, plan poison,
  close fail-closed semantics은 production candidate에도 유지한다.
- shared-host PSI는 before/after 공변량으로 보존한다. 낮은 PSI를 기다리거나, sample을 제거·가중·보정하거나,
  빠른 결과만 재실행해 선택하지 않는다.
- 4090에서 지원하지 않는 Hopper/Blackwell/multi-GPU 기능은 compile/dispatch contract를 검증하고
  runtime test는 `unsupported-on-SM89`로 기록한다. 이를 4090 fallback으로 숨기지 않는다.

## 중간 vLLM 비교표 계약

각 의미 있는 batch 종료 뒤 [P10 비교표](33-cublaslt-bias-epilogue-qualification.md#serving-비교표-보고-계약)를
갱신한다. quality-only batch는 `vLLM throughput`, `TTFT`, `TPOT`, `P95/P99`를 **미실행**으로 남기고,
통과 범위와 다음 gate를 적는다. 실제 숫자는 아래 조건을 모두 만족하는 serving batch에서만 넣는다.

| 필수 고정값 | 기록할 값 |
|---|---|
| 모델 | model revision, checkpoint SHA, tokenizer, dtype, numerical profile |
| 장비 | GPU/SM, driver, CUDA/cuBLAS, GPU memory cap, active GPU process |
| 워크로드 | prompt/output distribution, EOS/sampling, cache/prefix policy, concurrency, warmup |
| 엔진 | Riley baseline/candidate revision과 selector receipt, vLLM version/actual backend/options; 지원되면 SGLang/TRT-LLM도 동일 열 |
| 반복 | 사전 선언한 AB/BA 순서, 독립 process 수, raw request rows, median + IQR 또는 fixed-seed bootstrap interval |
| 안정성 | success/error/reject/OOM, E2E·TTFT·TPOT P50/P95/P99, GPU/CPU memory와 PSI before/after |

과거 다른 모델·다른 GUI/메모리 조건·operator event 결과를 새 3B serving 행에 섞지 않는다.

## PR-sized sequence

### P11 — raw Q/K/V default-arithmetic qualifier

범위는 direct cuBLAS plan의 test-only surface와 immutable P2051 oracle artifact다. Q
`[2051, 2048, 2048]`, K/V `[2051, 256, 2048]`의 no-bias endpoints를 하나의 quality batch로
추가한다.

- P7-compatible oracle에 Q/K/V input, weight, default output, manifest/sidecar hashes를 고정한다.
- native/Rust probe에서 각 projection을 두 번 실행해 BF16 byte equality, allocation delta, plan/context
  close, metadata `(M,N,K)`, math/pointer/atomics mode를 검증한다.
- raw Q/K/V 중 하나라도 non-exact, lifecycle failure, artifact mismatch면 candidate는 test-only로 남기고
  P12/P13/serving 측정으로 진행하지 않는다.
- 결과 표에는 Q/K/V exactness와 eligibility만 추가한다. vLLM serving 열은 `미실행`이다.

완료 조건은 세 raw endpoint의 predeclared oracle equality와 artifact integrity다. 이 단계의 elapsed
time은 성능 수치가 아니다.

### P12 — bias boundary와 full-sequence projection gate

P11을 통과한 경우에만 direct raw output과 bias application의 numerical profile을 분리한다. 이 단계는
`direct-cuBLAS raw + staged bias`가 HF actual module output과 자동으로 같다고 가정하지 않는다.

- raw Q/K/V는 P11 default oracle과 exact해야 한다.
- staged bias 결과는 별도의 explicit staged reference와 비교하고, actual HF module output도 독립적으로
  비교한다. 둘 중 어느 contract를 selector candidate의 target으로 삼는지 결과 전 선언한다.
- full `M=2051` 및 serving-relevant small M set에서 Q/K/V의 bias-boundary trace, repeated output,
  allocation/lifetime failure cases를 receipt로 남긴다.
- strict staged profile과 fused `cuBLASLt BIAS` profile을 교차 참조하되, 서로 다른 rounding 결과를
  같은 정답으로 간주하지 않는다.

완료 조건은 선택 numerical profile의 Q/K/V full-sequence gate pass다. 실패 시 P11 결과는 arithmetic
diagnostic으로 보존하고 selector integration을 막는다.

### P13 — cache-off와 corrected cache-on full-forward quality gate

P12 합격 candidate만 실제 Qwen2.5-3B full-forward로 확대한다. 이 batch는 layer-local 성공을
model-level correctness로 과장하지 않기 위한 마지막 numerical gate다.

- cache-off teacher prefix에서 layer boundary, final norm, logits, selected-token sequence를 사전 선언한
  oracle과 비교한다.
- cache-on은 corrected logical position contract로 prefill 뒤 첫 decode와 연속 decode를 별도 검증한다.
  historical off-by-one cache receipt는 golden으로 재사용하지 않는다.
- Q/K/V candidate가 다른 graph/attention/FFN path와 함께 있을 때도 deterministic repeat, zero hot-path
  allocation growth, cancellation/close cleanup을 확인한다.
- exactness가 contract상 요구되는 endpoint에는 tolerance를 새로 도입하지 않는다. logits/token의 허용
  기준이 필요한 경우에는 실행 전에 numerical profile 문서와 artifact schema에 명시한다.

완료 조건은 full-forward quality, lifecycle, cache-on/off gate 모두 pass다. 실패하면 selector와 serving
benchmark는 보류하고 첫 divergent boundary만 다음 profiling target으로 기록한다.

### P14 — opt-in production integration

P13 통과 뒤에만 test feature와 분리된 opt-in production selector를 만든다. 이 PR은 scheduler나
HTTP behavior를 바꾸지 않고 model projection dispatch와 provenance만 연결한다.

- dedicated production feature/link contract를 만들고, default build에는 direct-cuBLAS symbols와 selector
  변경이 들어가지 않게 한다.
- cold prepare에서 handle/stream mode와 supported shape/profile을 확정하고 hot path에는 descriptor
  creation, heuristic, allocation, Python callback을 넣지 않는다.
- cold unsupported는 explicit strict fallback receipt만 허용한다. execute failure는 plan poison/error이며
  같은 request에서 silent fallback하지 않는다.
- selected profile, fallback reason, CUDA/cuBLAS metadata, graph/batch admission을 request-independent
  startup receipt에 기록한다.

완료 조건은 P13 regression 재통과와 strict/candidate selector identity proof다. 이 batch가 끝나면
첫 matched serving comparison을 실행할 자격이 생긴다.

### P15 — Riley baseline/candidate/vLLM matched AB/BA serving qualification

P14를 통과한 동일 source/model에서 baseline Riley, opt-in candidate Riley, pinned latest vLLM을
교차 실행한다. SGLang/TRT-LLM은 해당 모델·GPU·dtype에서 지원되고 동일 contract를 만들 수 있을 때만
추가하며, 미지원이면 이유를 표에 남긴다.

1. frozen baseline binary와 candidate binary, vLLM image/version/backend/options, checkpoint/tokenizer,
   GPU process state를 먼저 고정한다.
2. 20GiB operating cap 안에서 short prefill, long prefill, long decode를 포함한 3B serving cells와
   concurrency ladder를 선언한다. 각 cell은 backend order를 AB/BA로 교차하고 독립 process 반복과
   raw request rows를 보존한다.
3. 각 cell에서 throughput, TTFT/TPOT, E2E P50/P95/P99, failure/OOM/reject, GPU/CPU memory,
   PSI를 모두 수집한다. PSI는 설명용 covariate다.
4. [P10 비교표](33-cublaslt-bias-epilogue-qualification.md#serving-비교표-보고-계약)에 baseline,
   candidate, vLLM 결과를 같은 row에 기록한다. throughput 최소 기준은 vLLM 이상, 목표는 +15% 이상;
   TTFT/TPOT 최소 기준은 vLLM 이하, 목표는 각각 10% 이상 감소다. 같은 cell에서 이득이 repeat
   dispersion과 겹치면 `미확정`으로 기록한다.

완료 조건은 full-forward quality가 유지된 모든 required cell의 reproducible receipt다. 한 유리한
microbenchmark 또는 한 concurrency cell만으로 전체 serving 우위라고 결론내리지 않는다.

### P16 — architecture and scale follow-up

P15 결과를 기준으로 Hopper/Blackwell and multi-GPU enablement를 별도 PR들로 나눈다. SM89에서
unsupported test를 통과처럼 해석하지 않는다.

- Hopper/Blackwell: architecture-specific math/graph/backend dispatch, kernel build, same numerical profile
  quality gate, then matched vLLM/TRT-LLM/SGLang receipt.
- multi-GPU: dense TP partition, NCCL stream/event ordering, rank failure cleanup, per-rank memory accounting,
  topology-matched serving comparison.
- long-context: paged-KV capacity, shared-prefix admission, prefill/decode split and tail stability under the
  exact same numerical profile.

## Stop conditions and next decision

P11–P13 중 하나가 quality gate에 실패하면 candidate promotion과 vLLM serving runs을 시작하지 않는다.
그 artifact의 first divergent endpoint와 reduction/bias/cache profile을 다음 research/profiling input으로
남긴다. P14가 full-forward를 통과하면 P15가 다음 meaningful batch이며, 그 종료 시점에 처음으로
Riley/vLLM 수치 비교표를 채운다.
