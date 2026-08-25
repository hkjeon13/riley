# PR 15 profiling and optimization evidence — run001

`1826d09fd28825582f22a2d49390f6ed047562ea`에서 fixed-M inference iteration의
per-operation CUDA completion을 하나의 thread-owned command batch completion으로 바꾸는
`E0` 최적화를 RTX 4090/sm89에서 검증했다. CUDA build, 실제 checkpoint inference,
paired benchmark와 Nsight Compute 수집은 모두 `server-4096`의 network-disabled
container에서 수행했다. 로컬에서는 source 편집과 CUDA를 사용하지 않는 검사만 수행했다.

최종 CLI 기본값 승격은 후속 source
`c6e1c9140753c34de27156a107e106a1672a82e3`에서 이루어졌으며, 그 exact archive에
대해 별도 Python-free CUDA compile-only gate를 통과했다.

## 결론

PR 15의 최종 최적화 gate를 통과했다. `per-operation` baseline과 `iteration-batch`
candidate는 같은 source, release profile binary, checkpoint, prompt corpus와 greedy request를
사용했다. 5개 독립 process pair에서 primary metric
`aggregate.host.execute_ns`의 arm median은 **18.597%**, paired median은 **18.588%**
개선됐다. Output throughput median은 **22.442%** 증가했고 TTFT/TPOT p95 ratio는 각각
`0.772982`, `0.817256`으로 regression bound `1.05`를 통과했다.

16-step greedy 실행에서 committed token ID와 매 iteration raw logits가 byte-exact였고,
hot-loop allocation 변화와 owner close 뒤 allocation count는 모두 0이었다. Nsight에서는
두 구현의 incremental kernel launch가 똑같이 517회였으며 kernel duration도
`2.370176 ms` 대 `2.368192 ms`로 사실상 같았다. 따라서 측정된 이점은 kernel 수나
kernel 연산 변경이 아니라 primitive마다 발생하던 host synchronization을 iteration당 한 번으로
모은 결과다.

## 후보 선택과 rollback 기록

첫 profiler 후보 A는 `45b31e212f7b56c6ff6d4f89567485ac04685a1d`의 exact
Residual + RMSNorm fusion이었다. Primitive raw-byte와 16-step greedy correctness는
통과했지만 동일한 5-pair gate에서 primary paired improvement가 `1.611507%`에 그쳐 사전
고정한 `5%` 기준을 통과하지 못했다. TTFT/TPOT와 throughput regression gate는
통과했으나 효과 크기 gate 실패를 성공으로 해석하지 않았고,
`c57fe3a`에서 production 기본값을 separate로 rollback했다.

최종 후보는 PR 15 후보 E의 synchronization 축소다. Native stream command batch는
호출 thread가 소유하며 guard는 `!Send + !Sync`다. Primitive가 사용한 buffer/GEMM lease는
고정 용량 ledger에 보존되고, `finish`가 stream을 한 번 synchronize한 뒤 release한다.
Guard drop은 best-effort end를 정확히 한 번만 시도한다. Ambiguous completion에서는 lease를
보수적으로 유지하며, 정상 종료·validation failure 뒤 재사용은 실제 GPU test로 검증했다.

Rollback은 다음과 같이 독립적이다.

- 실행 completion: `--execution-completion per-operation`
- Residual + RMSNorm: `--residual-rmsnorm separate`
- 최종 CLI 승격 source에서 command-batch 문제가 발견되면 기본값만 per-operation으로 되돌리고
  exact fallback과 동일 binary paired gate를 재실행한다.

## Correctness와 build gate

| Gate | 결과 |
|---|---|
| Python-free CUDA production/profile compile 및 C ABI/dependency smoke | pass |
| workspace all-features/all-targets | pass |
| command-batch one-shot finish, nested begin 거부, Drop 뒤 stream 재사용 | pass |
| multi-primitive resource ledger와 validation fail-closed | raw-byte mismatch 0, allocation delta 0 |
| SmolLM2 per-operation 대 iteration-batch | 16 decode steps, raw-logit mismatch 0, token mismatch 0 |
| executor close accounting | live allocation count 0 |

Correctness report의 고정 binding은 다음과 같다.

```text
gate_id: pr15-iteration-command-batch-exact-v1
semantic_class: E0
source: 1826d09fd28825582f22a2d49390f6ed047562ea
source archive sha256: 39ce9b9898defc9cbc3b0a346739d0743de3833e189f65434985b38606abb86d
correctness report sha256: de5c7c1564290e2ea16cb05f24501f508fd84fd563b2527887ebd6b731bbce39
profile executable sha256: 60650fa3af1d8a761432672862d1c464db86f2080d1acd5d81dccc6d1f7c9be8
```

## Paired end-to-end benchmark

Workload는 `smollm2-c1-p128-o32-greedy-v1`이다. Concurrency 1, prompt 128 tokens,
output 32 tokens, warmup 5회와 measured iteration 30회를 각 process에서 실행했다.
Baseline/candidate 각 5회이며 pair 순서를 교차했다. Sampling은 greedy이고 seed는 `null`이다.

두 arm은 동일한 다음 provenance에 묶였다.

```text
request identity sha256: e6a99a749c41a8227574c96a1d23f8b7d877d6e75b0df4d99154db1b1921a2e6
workload sha256: 0c5f1fd51a2a83011334a1716ae6820aeeb89948d41b94b79eda2fdbc35b75ee
environment sha256: e12f7ff03aadee299697a48911aaba151e52ee5d286374ee74b0ca533b46def6
```

| Metric | Baseline median | Candidate median | 결과 |
|---|---:|---:|---:|
| `aggregate.host.execute_ns` | 8,328,279,797 | 6,779,500,596 | arm improvement 18.597% |
| paired primary improvement | — | — | median 18.588% |
| end-to-end latency | 281.699 ms | 230.022 ms | ratio 0.816554 |
| output throughput | 113.543 tok/s | 139.024 tok/s | +22.442% |
| TTFT p95 | 7.176 ms | 5.547 ms | ratio 0.772982 |
| TPOT p95 | 8.883 ms | 7.260 ms | ratio 0.817256 |

고정 checker는 failure 0, dropped trace record 0, TTFT/TPOT p95 ratio `<= 1.05`,
throughput ratio `>= 0.95`, primary paired improvement `>= 0.05`를 모두 확인했다.
Machine-readable report SHA-256은
`804f6ee39d3aada4bfb8853eaa4772941b7dfc545c9def7816c9d711a894e060`이다.

## Nsight Compute 해석

Nsight는 output 1/2 sentinel trace를 각 arm에서 수집했다. Output-2 trace의 두 번째
iteration 517개 record를 직접 합산해 첫 iteration과의 replay 편차를 섞지 않았다.

| Completion mode | Incremental launches | Incremental kernel duration |
|---|---:|---:|
| per-operation | 517 | 2.370176 ms |
| iteration-batch | 517 | 2.368192 ms |

Launch count는 줄지 않았고 candidate kernel duration 차이는 `-0.001984 ms`
(`-0.084%`)에 불과하다. Command batch는 CUDA work의 순서와 kernel 구현을 바꾸지 않고
host가 각 primitive 뒤에 기다리는 completion boundary만 합친다는 `E0` 가설과 일치한다.

네 raw NCU CSV의 SHA-256은 다음과 같다.

```text
per-operation output1: 247d018ebdc82c5d3c8e9d510e72dc485a217bdb50939bae0c8a0a3a5de1a1e5
per-operation output2: e37c00f1f292e57308dca2bb67496ae74de4eba64fe9f3e9f3990e02c7262b00
iteration-batch output1: e5f08658221e5904fa95cfcf44a991bae53f63ba47e7b739a31e7ae58d4b864c
iteration-batch output2: 9f1947dd0387eb69d0a6930fad1ba5482131207cfad5c7a33d45b8f4dde8db9a
```

## 환경과 고정 artifact

- Host: `server-4096`, Ubuntu 22.04.5, Linux 6.8.0-138-generic, x86_64
- CPU/RAM: Intel Core i7-13700K, 16 physical/24 logical cores,
  67,185,598,464 bytes RAM
- GPU: NVIDIA GeForce RTX 4090, UUID
  `GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0`, compute capability 8.9,
  25,757,220,864 bytes VRAM
- Driver 580.173.02, CUDA runtime 12.8.1, toolkit/nvcc 12.8.93,
  cuBLAS 12.8.4.1
- Rust 1.85.0, CUDA architecture 89
- Image ID:
  `sha256:fe0f909b27532a472f4417dc613271727651d2e52ce4b9fdb702c39f1e890849`
- SmolLM2 revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16
- Weights SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- Tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`
- Container network disabled; Cargo는 `--locked --offline`; source/checkpoint/cache는
  read-only이고 target/evidence만 writable
- Production dependency는 CUDA C++/C ABI, CUDA runtime, driver와 cuBLASLt 그대로이며
  Python, PyTorch, Triton, NVRTC 또는 새 runtime allocation을 추가하지 않았다.

## Authoritative artifact index

전체 stdout, exact source archive, 10개 raw paired run, checker report, correctness report와
네 NCU trace는 다음 append-only 원격 root에 있다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr15/1826d09fd28825582f22a2d49390f6ed047562ea/run-20260825T184555Z
SHA256SUMS payload lines: 57
regular files including root SHA256SUMS: 58
regular-file bytes: 56,198,040
artifact-root apparent bytes: 56,230,808
root SHA256SUMS sha256: f25393c891a78cfcf661e525192e1c24b2f472440221b49cc0820815acc22d20
```

Root의 정확한 payload 구성은 다음과 같다.

- `source.tar`
- `evidence/command-batch-lifecycle-gpu.log`
- `evidence/command-batch-primitives-gpu.log`
- `evidence/correctness-report-v1.json`
- `evidence/cuda-compile-only.log`
- `evidence/iteration-command-batch-model-parity-gpu.log`
- `evidence/workspace-all-features-all-targets.log`
- `evidence/native-profile-commands.sh`
- `evidence/native-profile-pair-check.log`
- `evidence/native-profile-pair-report.json`
- `evidence/native-profile-runs.log`
- `evidence/native-profile/baseline-{1..5}.json`
- `evidence/native-profile/candidate-{1..5}.json`
- `evidence/ncu-{per-operation,iteration-batch}-o{1,2}-run001/` 각각의
  `SHA256SUMS`, `completion.response`, `environment.txt`, `model-SHA256SUMS`,
  `ncu.csv`, `provenance.txt`, `runtime-flag.txt`, `server.stderr`, `server.stdout`

`native-profile-runs.log`와 네 `server.stderr`는 의도적으로 비어 있다. Run payload는 각
JSON에 직접 기록됐고 NCU server stderr가 없었음을 empty-file SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`로 보존한다.

Rejected Residual + RMSNorm 후보의 감사 가능한 원격 record는 다음과 같다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr15/45b31e212f7b56c6ff6d4f89567485ac04685a1d/run-20260825T175524Z
source archive sha256: 3296e447c11d79dd1f00a6a163fe64f0e64cb18e57aa93d1f652a823bf6d2d66
correctness report sha256: 23a82037607944ba673e809f244426a1721803ceb5a83b8398d43376ce5b7067
failed pair report sha256: d64c4e64e703a6636c36cd4cdcc7a34d007896d1bd363b8c72d78b826cdf8455
```

최종 CLI 승격 source의 compile-only record는 다음과 같다.

```text
server-4096:/home/psyche/rustinfer-artifacts/pr15/c6e1c9140753c34de27156a107e106a1672a82e3/run-20260825T185910Z
source archive sha256: 913093e20a0564d8abbd3d4c13200ab8465d2759138eb4e73238209fdc3a5ad9
compile-only log sha256: ad3a881fd7b7d068d51b5bd6aa3db5bbdb0aa5f1cafcb032a91159a4865faba4
SHA256SUMS sha256: 159eaa56e7f920e8d090e9efb90b90f3439430570a4ec4519e3e5b8beecedd81
```

## 제한

- 성능 수치는 단일 RTX 4090, SmolLM2-135M, concurrency 1, prompt 128/output 32의
  고정 workload에만 적용된다. Multi-GPU, 다른 model size, 높은 concurrency와 긴 context의
  개선 폭을 추정하지 않는다.
- NCU는 kernel launch/duration이 동일함을 뒷받침하지만 CUDA API trace 자체는 아니다.
  End-to-end paired metric과 native lifecycle test가 synchronization 효과의 운영 근거다.
- Final CLI promotion source `c6e1c914...`는 default 선택만 바꾼 후 compile-only로
  검증했다. 성능 수치는 exact implementation/profile binary source `1826d09...`에 묶이며,
  promotion commit에서 새 성능 수치를 만들었다고 주장하지 않는다.
- Residual + RMSNorm fusion은 correctness는 통과했으나 5% 효과 gate 실패로 기본 비활성이다.
  이 결과를 최종 성능 개선에 합산하지 않는다.
- Command-batch ledger는 고정 용량이며 unsupported/nested/cross-thread lifecycle은
  fail-closed다. Ambiguous CUDA completion 뒤 자동 recovery는 범위 밖이고 process restart와
  per-operation rollback을 사용한다.

## Memory와 dependency trade-off

추가 device/pinned allocation은 없다. Stream의 native host object에 고정 용량 resource
ledger와 owner token이 추가되며 hot-loop allocation delta는 0이다. Batch가 정상 종료될 때까지
사용 resource의 close/reuse가 지연되고, ambiguous completion에서는 안전을 위해 lease를
계속 보존할 수 있다. 이는 매 primitive synchronization을 제거하는 대신 선택한 명시적
lifecycle trade-off다.

Production binary는 계속 Python-free이며 CUDA C++와 기존 cuBLASLt 경계를 유지한다.
CUTLASS/custom GEMM, Triton과 NVRTC는 이번 병목에 필요하지 않았고 runtime dependency 변화가
없다.
