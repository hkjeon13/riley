# Benchmark artifact 계약

재현 가능한 benchmark의 version-controlled 결과는 다음 경로에 append-only로 저장한다.

```text
benchmarks/results/<YYYYMMDDTHHMMSSZ>-<implementation-id>-<workload>-<run-id>/
├── README.md
├── metadata.json
└── raw.csv 또는 raw.jsonl
```

timestamp는 UTC를 사용하고 `run-id`는 같은 시각·구현·workload에서도 충돌하지 않는 실행 식별자다. PR 01에서 공통 JSON schema와 benchmark matrix를 추가하기 전까지는 이 문서가 저장 위치와 보존 방식만 정의한다. 기존 결과를 새 측정으로 덮어쓰지 않고 매 실행마다 새 디렉터리를 만든다.

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

End-to-end 결과에는 model/checkpoint revision, dtype/quantization, batch/concurrency, prompt/output 길이, sampling 설정, TTFT, TPOT/ITL, throughput, peak VRAM과 failure count를 포함한다. Microbenchmark 결과에는 operation, shape/stride/layout, dtype, kernel/backend 설정, latency 또는 throughput, allocation/메모리 정보와 failure count를 포함한다. 현재 scope에 적용되지 않는 공통 필드는 `null`로 기록하고 README에 이유를 적는다.

`raw.csv` 또는 `raw.jsonl`에는 집계 전 반복 측정값을 둔다. `metadata.json`에는 측정 환경과 공통 조건을 기계가 읽을 수 있게 기록한다. `README.md`에는 결과 요약, 알려진 편차, 비교 가능 여부를 적는다.

대형 profiler trace, checkpoint, model weight, 전체 tensor dump, 실행 binary는 Git에 넣지 않는다. 외부 artifact store를 사용하고 결과 README에 URI, 크기, SHA-256 checksum, 보존 기간을 기록한다. secret, credential, 사용자 prompt나 개인정보가 포함된 raw artifact는 저장하지 않는다.

서로 다른 model, dtype, sampling, prompt/output 길이, warm state의 결과를 직접적인 before/after 성능 주장에 사용하지 않는다. Python reference lane과 native production lane은 dependency class를 명시해 구분한다.
