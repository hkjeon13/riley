# Optional token HTTP correctness gate

This helper is a correctness gate for frozen source `a179617070526068b66ba5627ba82a7151da8c64` and Riley binary `18cbd5f8a8ad8583ecfcd2d38815a484831c05eb9f081b1db67be6194ffebd8c`. It makes no performance claim. The existing GPU full-logit/KV and default-wire receipts must already pass for the same build. It launches only its own Riley server, leaves retained desktop sessions running, and verifies private driver files, actual mapped libcuda, kernel version, model hashes, GPU UUID, source and binaries before and after each sampler.

Run on the authorized GPU host only after copying the frozen helper and token client beside the existing helpers:

```sh
/data/riley-vllm-interim.CfrT9T/venv/bin/python /tmp/riley-opt-260912/http_token_observation_check.py --root /tmp/riley-opt-260912 --base /tmp/riley-g04-vllm-profile-260911 --client-module /tmp/riley-opt-260912/serving_token_client.py
```

The output directory is exclusive: `/tmp/riley-opt-260912/http-token-observation-correctness`. Existing output is never replaced. Each sampler (`cpu`, `gpu-greedy`) runs 44 complete requests plus two deliberate disconnects: twelve fixed format/shape cases, two requests per worker at offered concurrency 1/2/4/8, and two exact O32 requests after disconnects. Engine active capacity is one. Shapes are P128/O32, P128/O1, and P128 with max32 and stop `I'm`. The stop reference is derived from the frozen generated IDs and tokenizer; the current expected prefix has three committed tokens with empty emitted text. Requests, complete raw JSON/SSE payloads, C02 committed audits and their completion markers, launch/configuration, private library mappings, and graceful native allocation cleanup are retained. Default SSE has no usage field; its terminal committed audit establishes counts. Raw SSE must publish each committed token separately, including blank text, and final usage.

Each HTTP request has a total deadline (30 seconds by default, configurable from 1 to 120), a bounded body, and an owned socket watchdog. Disconnect checks prove connection closure and exact subsequent engine reuse; they do not infer whether cancellation won the race against terminal completion. At most two additional full O32 audit records per sampler may come from those races. Owned server process cleanup runs on every exit path.

The top-level `completion.json` uses schema `riley.http-token-observation-correctness.v1`, with `completed: true`, `gpu_tests_executed: true`, `performance_claim: false`, exact source/build/binary/request/reference/client/helper/model pins, both samplers, all 92 checks, active capacity and offered concurrency, and eight explicit proof booleans. The receipt is published only after reconstruction of every saved response, exact committed-audit matching, complete coverage, overlap, and cleanup checks. An error writes `failure.json` and does not publish the top-level completion receipt.

Read-only validation, without starting a process or touching the GPU:

```sh
/data/riley-vllm-interim.CfrT9T/venv/bin/python /tmp/riley-opt-260912/http_token_observation_check.py --validate-only
```

Controllers may import `validate_completion(Path(...))`; it returns the reconstructed, validated receipt. They must additionally pin this validator and compare its source/build/reference identity with their campaign. Its `performance_claim` remains false even when numerical/HTTP correctness passes.

CPU-only local tests use synthetic IDs and a loopback server. They cover both samplers' twelve wire formats, blank stop commits, C8 overlap, mutated source audit/observation rejection, bounded HTTP/1.0 and HTTP/1.1 trickles, header timeout, close failures, disconnect/reuse, worker failure draining, owned process cleanup, native zero-gauge validation, all 92 mock checks, and failure before sampler completion. These tests do not qualify GPU arithmetic or a real serving build.
