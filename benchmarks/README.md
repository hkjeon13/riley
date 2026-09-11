# Benchmark artifact 계약

재현 가능한 benchmark의 version-controlled 결과는 다음 경로에 append-only로 저장한다.

```text
benchmarks/results/<YYYYMMDDTHHMMSSZ>-<implementation-id>-<workload>-<run-id>/
├── README.md
├── metadata.json
└── raw.csv 또는 raw.jsonl
```

timestamp는 UTC를 사용하고 `run-id`는 같은 시각·구현·workload에서도 충돌하지 않는 실행 식별자다. PR 01의 공통 JSON schema와 benchmark matrix가 row 형식과 반복성 gate를 정의하고, 이 문서는 저장 위치와 보존 방식을 정의한다. 기존 결과를 새 측정으로 덮어쓰지 않고 매 실행마다 새 디렉터리를 만든다.

각 결과 디렉터리는 최소한 다음 정보를 보존한다.

- 측정 목적과 실행 명령
- Git revision과 dirty 여부
- GPU, compute capability, CPU, RAM, OS
- NVIDIA driver와 CUDA toolkit/runtime
- benchmark scope: `end-to-end` 또는 `microbenchmark`
- warm-up/측정 횟수와 cold/warm 상태
- 측정 단위와 median, p95, 측정한 경우 p99
- implementation ID, reference implementation, runtime dependency class
- `reference`, `E0`, `E1`, `A1`, `M1` 중 의미 보존 등급
- approximation/speculative parameter와 exact/reference fallback 결과

`reference`는 변환 전 기준 구현뿐 아니라 allocator, scheduler, launch orchestration처럼 수학·확률·근사 변환이 없는 exact systems 변경에도 사용한다. 이 경우 README에 보존한 inference 의미와 비교 대상 implementation을 명시한다. 문서 전용 PR의 `N/A`는 benchmark row의 `semantic_class` 값으로 사용하지 않는다.

End-to-end 결과에는 model/checkpoint revision, dtype/quantization, batch/concurrency, prompt/output 길이, sampling 설정, TTFT, TPOT/ITL, throughput, peak VRAM과 failure count를 포함한다. 성공 request의 `itl_ms`는 첫 token 이후 실제 per-token interval 배열이고 `mean_tpot_ms`는 그 산술 평균이다. Microbenchmark 결과에는 operation, shape/stride/layout, dtype, kernel/backend 설정, latency 또는 throughput, allocation/메모리 정보와 failure count를 포함한다. 현재 scope에 적용되지 않는 공통 필드는 `null`로 기록하고 README에 이유를 적는다.

`raw.csv` 또는 `raw.jsonl`에는 집계 전 반복 측정값을 둔다. `metadata.json`에는 측정 환경과 공통 조건을 기계가 읽을 수 있게 기록한다. `README.md`에는 결과 요약, 알려진 편차, 비교 가능 여부를 적는다.
PR 01 repeatability bundle은 외부 staging에서 dependency/cache preparation과 20개 measured subprocess를 끝낸 뒤에만 import한다. metadata/README에는 offline cache root, prime commands/logs, post-prime fingerprint와 각 measured invocation의 cache 불변 증거를 함께 보존한다. 이 bundle의 cold는 process/model-state cold이며 OS와 immutable disk/compile cache cold가 아니다.
Preparation 전/후의 전체 cache entry 목록은 deterministic
`cache.inventory.{before,after}.json.gz`로 보존한다. JSON은 UTF-8, key 정렬,
compact separator와 trailing newline을 사용하고 gzip은 level 9와 `mtime=0`을
고정한다. 압축 해제한 내용이 cache inventory v1 증거의 원문이다.

대형 profiler trace, checkpoint, model weight, 전체 tensor dump, 실행 binary는 Git에 넣지 않는다. 외부 artifact store를 사용하고 결과 README에 URI, 크기, SHA-256 checksum, 보존 기간을 기록한다. secret, credential, 사용자 prompt나 개인정보가 포함된 raw artifact는 저장하지 않는다.

서로 다른 model, dtype, sampling, prompt/output 길이, warm state의 결과를 직접적인 before/after 성능 주장에 사용하지 않는다. Python reference lane과 native production lane은 dependency class를 명시해 구분한다.

첫 릴리스 이후 확장은 구현 전에
[`benchmarks/extensions/`](extensions/README.md)의 별도 benchmark contract를
등록한다. 이 admission contract는 공통 `result.schema.json`을 변경하지 않으며,
서로 다른 Git-tracked reference/fallback과 workload artifact의 SHA-256, one-GPU
GPU/driver/CUDA/model ID/full 40-hex model revision/dtype, concurrency, prompt/output 길이,
sampling, cold/warm의 실제 값을 고정한다. primary/quality metric은 공통 result
schema의 서로 다른 scalar path여야 하고 primary는 exact track-required set에
포함되어야 한다.
적합한 quality field나 multi-GPU 표현이 없으면
공통 schema를 먼저 version-up하기 전까지 admission을 허용하지 않는다.
