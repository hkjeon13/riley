#!/usr/bin/env python3
"""vllm_prof.py MODEL: the prof_load.py prompts through vllm.LLM with the server's scheduler limits."""
import random, sys, time
from vllm import LLM, SamplingParams
llm = LLM(model=sys.argv[1], max_model_len=8192, gpu_memory_utilization=0.8, enable_prefix_caching=False,
          max_num_batched_tokens=2048, max_num_seqs=32, generation_config="vllm")
words = "the of model server request token memory context cache vector search document answer query system data result value time long short".split()
params = SamplingParams(temperature=0, max_tokens=4)
for seed in range(10):
    rng = random.Random(seed)
    text = " ".join(rng.choice(words) for _ in range(2600))
    start = time.time()
    out = llm.chat([{"role": "user", "content": text}], params, use_tqdm=False)
    print(seed, len(out[0].prompt_token_ids), round(time.time() - start, 3), flush=True)
    time.sleep(0.3)
