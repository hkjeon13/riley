# C8 v3 preflight startup timeout

No timed serving lanes ran. The restored Riley prior generated shared/unique reference outputs successfully. The vLLM 0.29.0 preflight exceeded the controller's 600-second HTTP readiness deadline. It reached model loading and torch.compile initialization; the log reports weights loaded in 0.31 seconds and Dynamo bytecode transform in 8.42 seconds. These stage durations do not explain the entire cold start.

During startup, `/proc` observations showed the API process in `folio_wait_bit_common` / `wait_on_buffer` and cumulative disk reads increasing from 480,002,048 to 580,308,992 bytes. This establishes ongoing I/O waits, not a measured causal attribution of every second. The controller terminated the server process group and ultimately recorded exit -9. This is a startup failure, not an inference throughput or correctness result.

`serving.log` contains the readiness assertion; `preflight-vllm029.log` contains engine initialization. `blender-restored.json` confirms all three Blender RPC endpoints restored. Full references, launch manifests and snapshots remain remotely in `/dev/shm/riley-projection-cta-serving-c8-v3` and the corresponding lifecycle directory.

Next attempt v4 uses explicit `--readiness-timeout-seconds 1800`. The new controller option retains default 600, accepts 60–3600 seconds, and records the actual value in preparation.json. This changes only the pre-timing startup allowance for every engine. Model, frozen binaries, warmup/retained requests, correctness checks and timing summaries are unchanged. Cache state may differ after this cold initialization attempt, so neither startup times nor hypothetical serving results are pooled across attempts.
