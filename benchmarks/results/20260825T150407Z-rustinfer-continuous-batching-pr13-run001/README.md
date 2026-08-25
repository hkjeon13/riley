# PR 13 continuous batching evidence — run001

`a303a23633baf465f625154b688cafc4009801b6`에서 bounded Rust scheduler,
fixed-capacity Llama batch executor, ragged paged-KV CUDA path와 scheduler/runtime
adapter를 RTX 4090/sm89에서 검증했다. 모든 CUDA build, checkpoint load와 model
inference는 `server-4096`의 network-disabled container에서 수행했다. 로컬에서는
source 편집과 CUDA를 사용하지 않는 검사만 수행했다.

## 결론

PR 13 gate를 통과했다. FCFS admission, decode-first scheduling, bounded aging,
chunked prefill, token/KV budgets와 cancellation/close 상태 전이는 deterministic
simulation으로 검증했다. Scheduler가 만든 immutable plan은 단일 fixed-M CUDA
iteration으로 실행되고, dense output slot을 통해 결과가 원래 request로 되돌아간다.

SmolLM2와 Qwen2.5에서 concurrency-one prefill은 기존 단일 request path와 byte-exact였고,
32회 decode의 greedy top-1은 모두 exact였다. 두 독립 request, permuted output slots와
prefill/decode 혼합 iteration도 독립 실행과 일치했다. SmolLM2 1,001-iteration gate에서는
준비된 host/device allocation 수가 유지되고 close 뒤 runtime CUDA allocation accounting이
0으로 돌아왔다.

## 구현 계약

- Scheduler는 `Waiting → Admitted → Prefilling → Decoding → Finished`와
  `Cancelled`/`Failed` 전이를 한 곳에서 검증한다.
- Waiting request 수·prompt token 수, active sequence 수, request 최대 길이,
  iteration token 수, prefill chunk, promised KV block과 metric window가 모두 설정된
  상한을 가진다.
- 한 번에 immutable `IterationPlan` 하나만 outstanding일 수 있다. Plan은 physical
  block alias, malformed CSR/block table, duplicate request/output slot과 decode output
  누락을 실행 전에 거절한다.
- Runtime은 preallocated fixed-M workspace와 layer-major paged KV arena를 재사용한다.
  indexed RoPE, row gather, ragged KV scatter와 causal D64 GQA attention은 batch를
  요청별 serial dispatch로 풀지 않는다.
- Executor 진입 후 오류는 stream을 synchronize한 뒤
  `DeviceQuiescedMutationUnknown`으로 보수 분류한다. Synchronize 자체가 실패하면
  scheduler rollback에 사용할 abort data를 제공하지 않는다.
- Cancellation은 실행 중인 kernel을 즉시 중단하지 않는다. 해당 iteration이
  quiescent해진 뒤 host ownership과 KV block을 정확히 한 번 회수한다.

## 의미 보존 등급과 oracle

`E0`이다. Continuous batch는 동일한 dense Llama/Qwen 연산을 ragged row와 paged KV
주소로 재배치하며, 확률 분포·모델·근사 budget을 바꾸지 않는다. Batch attention은
기존 BF16 staging 계약(`BF16(QK)`, BF16 scale, softmax, reciprocal multiply)을 보존한다.

수치 gate는 결과를 본 뒤 새로 맞추지 않고 PR 01 E0 v2의 고정 final-logit bound를
재사용했다.

| Metric | 고정 bound | SmolLM2 worst | Qwen2.5 worst |
|---|---:|---:|---:|
| cosine | `>= 0.9979035305495393` | `0.998494007581` | `0.999260337367` |
| max absolute error | `<= 5.852936458587647` | `0.531250000` | `0.546875000` |
| mean absolute error | `<= 1.151280319263363` | `0.273596887` | `0.094341452` |
| 32-step greedy top-1 mismatch | `0` | `0` | `0` |

초기 ragged attention은 established BF16 staging을 빠뜨려 Qwen step 1 cosine과
step 14 top-1이 실패했다. Kernel과 CPU oracle 모두 staged BF16 scale과 reciprocal
multiply 계약으로 수정했으며 tolerance를 완화하지 않았다. 수정 후 두 model family가
위 고정 gate를 통과했다.

## 원격 GPU 검증

| Gate | 결과 |
|---|---|
| workspace all-features strict Clippy | pass |
| workspace all-targets/all-features | 237 passed, 0 failed, 65 ignored |
| workspace all-features doctests | 13 passed, 0 failed |
| scheduler unit + deterministic simulation | 21 + 14 passed |
| low-level indexed/ragged CUDA batch | 4 passed |
| SmolLM2 batch/model parity | 5 passed; 32-step top-1 exact |
| Qwen2.5 batch/model parity | 5 passed; 32-step top-1 exact |
| scheduler → CUDA → greedy sampler → commit | prefill then decode passed; close accounting 0 |
| actual long gate | 1,001 iterations passed; close accounting 0 |
| CUDA Compute Sanitizer memcheck | 0 errors |
| CUDA Compute Sanitizer racecheck | 0 hazards, 0 errors, 0 warnings |

## 환경과 고정 artifact

- Host: `server-4096`, Ubuntu 22.04.5, Linux 6.8.0-138-generic
- CPU/RAM: Intel Core i7-13700K, 24 logical CPUs, 67,185,598,464 bytes RAM
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB
- Driver 580.173.02, CUDA runtime 12.8.1, nvcc 12.8.93
- Rust/Cargo 1.85.0, `RUSTINFER_CUDA_ARCHITECTURES=89`
- Image ID: `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- SmolLM2: `HuggingFaceTB/SmolLM2-135M` revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16
- Qwen: `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16
- Container network disabled; source/checkpoints/Cargo registry read-only; target/evidence만 writable

주요 Cargo command는 모두 `--locked --offline`로 실행했다.

```text
cargo test --doc --workspace --all-features --locked --offline
cargo test -p rustinfer-cuda --features cuda --test batch_gpu --locked --offline -- --ignored --nocapture --test-threads=1
RUSTINFER_REAL_CHECKPOINT=<model> cargo test -p rustinfer-runtime --features cuda --test llama_batch_gpu --locked --offline -- --ignored --nocapture --test-threads=1
RUSTINFER_PR13_LONG_STEPS=true RUSTINFER_REAL_CHECKPOINT=<smollm2> cargo test -p rustinfer-runtime --features cuda --test llama_batch_gpu --locked --offline one_thousand_iterations_do_not_allocate_or_leak -- --ignored --nocapture --test-threads=1
compute-sanitizer --tool memcheck <batch-gpu-test-binary> --ignored --nocapture --test-threads=1
compute-sanitizer --tool racecheck <batch-gpu-test-binary> --ignored --nocapture --test-threads=1
RUSTINFER_REAL_CHECKPOINT=<smollm2> cargo test -p rustinfer-scheduler --features cuda --test llama_iteration_gpu --locked --offline -- --ignored --nocapture --test-threads=1
cargo clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo test --workspace --all-targets --all-features --locked --offline
```

## Artifact와 provenance

이 디렉터리의 `raw-events.jsonl`은 parity, scheduler integration, long-run과 sanitizer
결과를 기계 판독 가능하게 보존한다. 전체 stdout, exact source tar, normalized command,
host/GPU/toolchain/checkpoint provenance와 checksums는 다음 append-only 원격 root에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr13/a303a23633baf465f625154b688cafc4009801b6/run-20260825T150407Z
checksum payload lines: 15
regular files including SHA256SUMS: 16
regular-file bytes: 29,822,854
artifact-root apparent bytes: 29,826,950
SHA256SUMS sha256: 0af9140e388ca386b989b4825cca039178253bae72c379ec2ea8d1d4978b9604
source tar sha256: 854c784fd5620446e2fbfce81ea219c034a8b60b5a9b05ff5cf9b6a5e48d2eb3
```

## 제한

- Functional/correctness evidence이며 고정 clock, warmup, 반복 측정 protocol이 없으므로
  latency·throughput 개선을 주장하지 않는다. PR 15가 profiling/optimization 범위다.
- 검증된 maximum batch는 2 request, fixed token capacity `M=8`이다. 더 큰 production
  concurrency와 prompt 분포는 PR 14 load test와 PR 15 benchmark에서 확장한다.
- Synthetic CUDA device-loss/fault injection은 수행하지 않았다. Adapter unit test는
  pre-dispatch/commit failure를 검증하고, 실제 GPU gate는 정상 execution과 sanitizer를
  검증한다. 모든 post-execute error는 보수적으로 stream quiescence를 요구한다.
- Runtime allocation accounting은 rustinfer가 소유한 CUDA buffer 기준이며 process peak
  VRAM이나 CUDA driver 내부 allocation이 아니다.
- 단일 GPU/sm89만 검증했다. multi-GPU routing, prefix cache, speculative decoding,
  offload와 tenant priority는 범위 밖이다.
