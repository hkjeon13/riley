import json,os
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',VLLM_DO_NOT_TRACK='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0')
from vllm import LLM,SamplingParams
root=Path(__file__).resolve().parent
old=Path('/tmp/riley-g04-readiness-260911')
env=json.loads((old/'environment.json').read_text())
request=json.loads((old/'request.json').read_text())
reference=json.loads((old/'parity.json').read_text())['vllm_output_tokens']
llm=LLM(model=env['vllm_checkpoint'],tokenizer=env['vllm_checkpoint'],dtype='bfloat16',
        max_model_len=160,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,
        gpu_memory_utilization=.3,seed=0,compilation_config={'cudagraph_mode':'NONE'},
        worker_extension_cls='serving_worker.ServingTrace')
def generate():
    return list(llm.generate([{'prompt_token_ids':request['prompt_token_ids']}],
        SamplingParams(temperature=0,top_p=1,repetition_penalty=1,max_tokens=32,ignore_eos=True),
        use_tqdm=False)[0].outputs[0].token_ids)
before=generate();assert before==reference
installed=llm.collective_rpc('install_trace')
try: observed=generate()
finally: receipt=llm.collective_rpc('finish_trace')
after=generate()
assert before==observed==after==reference
(root/'invariance.json').write_text(json.dumps({'before':before,'observed':observed,'after':after,
    'matches_default_graph_reference':True,'installed':installed,'receipt':receipt,'performance_trials':0},indent=2)+'\n')
print('SERVING_TRACE actual_backend=true default_output_unchanged=true performance_trials=0')
