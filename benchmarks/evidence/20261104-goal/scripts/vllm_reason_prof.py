#!/usr/bin/env python3
"""vllm_reason_prof.py MODEL: reason_client.py's 8 prompts in one vllm.LLM batch with the server's limits."""
import sys
from vllm import LLM, SamplingParams
TOPICS = ["the history of bridges", "how compilers work", "the ocean currents", "a city at night",
          "the life of bees", "trains in winter", "the first computers", "a mountain village"]
llm = LLM(model=sys.argv[1], max_model_len=8192, gpu_memory_utilization=0.8, enable_prefix_caching=False,
          max_num_seqs=32, generation_config="vllm")
params = SamplingParams(temperature=0, max_tokens=1024)
llm.generate(["warm up"], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
outputs = llm.generate([f"Write a very long and detailed essay about {t}.\n\n" for t in TOPICS], params, use_tqdm=False)
print("completion tokens", [len(o.outputs[0].token_ids) for o in outputs], flush=True)
