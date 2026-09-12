"""Isolate vLLM execution modes using correctness requests only."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=['default', 'no-graph', 'custom-ops', 'silu', 'rope', 'norm'])
args = parser.parse_args()
os.environ.update(HF_HUB_OFFLINE='1', VLLM_DO_NOT_TRACK='1',
                  VLLM_NO_USAGE_STATS='1', VLLM_USE_FLASHINFER_SAMPLER='0')
from vllm import LLM, SamplingParams

root = Path(__file__).resolve().parent
old = Path('/tmp/riley-g04-readiness-260911')
env = json.loads((old/'environment.json').read_text())
request = json.loads((old/'request.json').read_text())
options = {}
if args.mode == 'no-graph':
    options['compilation_config'] = {'cudagraph_mode': 'NONE'}
elif args.mode == 'custom-ops':
    options['compilation_config'] = {'custom_ops': ['all']}
elif args.mode in ['silu', 'rope', 'norm']:
    op = {'silu': 'silu_and_mul', 'rope': 'rotary_embedding', 'norm': 'rms_norm'}[args.mode]
    options['compilation_config'] = {'custom_ops': ['none', '+'+op]}
llm = LLM(model=env['vllm_checkpoint'], tokenizer=env['vllm_checkpoint'],
          dtype='bfloat16', max_model_len=160, max_num_seqs=1,
          max_num_batched_tokens=128, enable_prefix_caching=False,
          gpu_memory_utilization=.3, seed=0, **options)
result = {'correctness_only': True, 'performance_trials': 0, 'mode': args.mode,
          'options': options, 'requests': []}
for logprobs in [None, 5]:
    output = llm.generate([{'prompt_token_ids': request['prompt_token_ids']}],
        SamplingParams(temperature=0, top_p=1, repetition_penalty=1,
                       max_tokens=32, ignore_eos=True, logprobs=logprobs),
        use_tqdm=False)[0].outputs[0]
    item = {'logprobs_requested': logprobs, 'tokens': list(output.token_ids)}
    if output.logprobs is not None:
        item['ranks'] = [{str(k): {'logprob': v.logprob, 'rank': v.rank}
                         for k, v in entry.items()} for entry in output.logprobs]
    result['requests'].append(item)
    (root/(args.mode+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print('CORRECTNESS', args.mode, logprobs, item['tokens'], flush=True)
