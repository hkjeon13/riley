"""Fresh-process vLLM correctness diagnosis; no performance samples."""
import os,json
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1', VLLM_DO_NOT_TRACK='1', VLLM_NO_USAGE_STATS='1', VLLM_USE_FLASHINFER_SAMPLER='0')
from vllm import LLM, SamplingParams
root=Path(__file__).resolve().parent
previous=Path('/tmp/riley-g04-readiness-260911')
env=json.loads((previous/'environment.json').read_text())
request=json.loads((previous/'request.json').read_text())
llm=LLM(model=env['vllm_checkpoint'],tokenizer=env['vllm_checkpoint'],dtype='bfloat16',max_model_len=160,max_num_seqs=1,max_num_batched_tokens=128,enable_prefix_caching=False,gpu_memory_utilization=.3,seed=0,enforce_eager=True)
out=llm.generate([{'prompt_token_ids':request['prompt_token_ids']}],SamplingParams(temperature=0,top_p=1,repetition_penalty=1,max_tokens=32,ignore_eos=True,logprobs=5),use_tqdm=False)[0]
result={'correctness_only':True,'performance_trials':0,'enforce_eager':True,'tokens':list(out.outputs[0].token_ids),'ranks':[{str(k):{'logprob':v.logprob,'rank':v.rank} for k,v in entry.items()} for entry in out.outputs[0].logprobs]}
(root/'vllm-eager-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print('VLLM_EAGER_CORRECTNESS',result['tokens'])
