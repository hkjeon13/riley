#!/usr/bin/env python3
"""Alternate Riley/vLLM on one GPU and require every declared cell to validate.

Legacy positional campaign layout is retained; --help lists explicit executable,
model-directory and load-script overrides. Each label must be new. A failed
cell exits nonzero, retains logs/run.json, and never prints COMPARE_DONE.
"""
from compare_runner import main

SHORT, LONG = "short", "long"
# name: (server config, load.py arguments)
SCENARIOS = {
    "closed-c1": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 1 --requests 64 --warmup 8"),
    "closed-c16": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 16 --requests 192"),
    "closed-c8": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 8 --requests 192"),
    "closed-c32": (SHORT, "--prompt-words 30 --max-tokens 64 --concurrency 32 --requests 192"),
    "open-r6": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 6 --requests 192"),
    "open-r12": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 12 --requests 192"),
    "open-r16": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 16 --requests 192"),
    "open-r24": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 24 --requests 192"),
    "open-r32": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 32 --requests 192"),
    "burst-r16": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 16 --burst 1,3 --requests 192"),
    "burst-r6": (SHORT, "--prompt-words 30 --max-tokens 64 --rate 6 --burst 1,3 --requests 192"),
    "mid-c32": (SHORT, "--prompt-words 300 --max-tokens 64 --concurrency 32 --requests 192"),
    "mid-r8": (SHORT, "--prompt-words 300 --max-tokens 64 --rate 8 --requests 192"),
    "mid-r16": (SHORT, "--prompt-words 300 --max-tokens 64 --rate 16 --requests 192"),
    "rag-c1": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4"),
    "rag-c8": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --concurrency 8 --requests 64 --warmup 8"),
    "rag-r1": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --rate 1 --requests 64 --warmup 8"),
    "rag-r3": (LONG, "--prompt-words 1500,3000 --max-tokens 128 --rate 3 --requests 64 --warmup 8"),
    "rag6k-c1": (LONG, "--prompt-words 3000,5000 --max-tokens 128 --concurrency 1 --requests 16 --warmup 4"),
    "reason-c8": (LONG, "--prompt-words 30 --max-tokens 1024 --concurrency 8 --requests 32 --warmup 8"),
    "reason-c32": (LONG, "--prompt-words 30 --max-tokens 1024 --concurrency 32 --requests 64 --warmup 32"),
    "shared-c8": (SHORT, "--shared-prefix-words 330 --prompt-words 20 --max-tokens 32 --concurrency 8 --requests 192"),
    "shared-c32": (SHORT, "--shared-prefix-words 330 --prompt-words 20 --max-tokens 32 --concurrency 32 --requests 192"),
    "logprobs-c8": (SHORT, '--prompt-words 30 --max-tokens 64 --concurrency 8 --requests 192 '
                           '--extra {"logprobs":true,"top_logprobs":5}'),
    "logprobs-c32": (SHORT, '--prompt-words 30 --max-tokens 64 --concurrency 32 --requests 192 '
                            '--extra {"logprobs":true,"top_logprobs":5}'),
}

if __name__ == "__main__":
    raise SystemExit(main(SCENARIOS))
