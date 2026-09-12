"""Replay the loaded vLLM model's MLP operations on fixed Riley tensors; correctness only."""
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', VLLM_DO_NOT_TRACK='1',
                  VLLM_NO_USAGE_STATS='1', VLLM_USE_FLASHINFER_SAMPLER='0')
from vllm import LLM, SamplingParams

root = Path(__file__).resolve().parent
old = Path('/tmp/riley-g04-readiness-260911')
env = json.loads((old/'environment.json').read_text())
request = json.loads((old/'request.json').read_text())
reference = json.loads((old/'parity.json').read_text())['vllm_output_tokens']
llm = LLM(model=env['vllm_checkpoint'], tokenizer=env['vllm_checkpoint'],
          dtype='bfloat16', max_model_len=160, max_num_seqs=1,
          max_num_batched_tokens=128, enable_prefix_caching=False,
          gpu_memory_utilization=.3, seed=0, worker_extension_cls='attention_worker.AttentionTrace')
def generate():
    return list(llm.generate([{'prompt_token_ids':request['prompt_token_ids']}],
        SamplingParams(temperature=0,top_p=1,repetition_penalty=1,
                       max_tokens=32,ignore_eos=True),use_tqdm=False)[0].outputs[0].token_ids)
before = generate()
assert before == reference


result = llm.collective_rpc('export_attention')
after = generate()
assert after == before
(root/'vllm-attention.json').write_text(json.dumps({'performance_trials':0,
    'before_tokens':before,'after_tokens':after,'model_outputs_unchanged':True,
    'worker_results':result},indent=2)+'\n')
print('VLLM_ATTENTION captured=true production_outputs_unchanged=true performance_trials=0')
