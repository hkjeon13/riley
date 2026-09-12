"""Prepare the actual pinned adapter and make one correctness-only LLM request."""
from pathlib import Path
import json,os,sys
r=Path('/tmp/riley-g04-vllm-profile-260911');plan=json.loads((r/'measurement-plan.json').read_text());pin=json.loads((r/'environment.json').read_text());binding=json.loads((r/'native-binding.json').read_text())
os.environ.update(plan['engine_lanes']['vllm']['env']);sys.path.insert(0,str(Path(plan['source_root'])/'benchmarks/lanes/vllm'))
from riley_vllm_benchmark import adapter
original=adapter._llm_options
def options(*,max_num_seqs):
 assert max_num_seqs==1
 value=original(max_num_seqs=1);value.update(model=pin['vllm_checkpoint'],tokenizer=pin['vllm_checkpoint'],max_model_len=160,max_num_batched_tokens=128,gpu_memory_utilization=.3,seed=0);return value
adapter._llm_options=options
backend=adapter.load_default_backend(local_files_only=True,max_num_seqs=1)
try:
 adapter._validate_environment(backend.environment())
 ids=backend.materialize_token_ids(['Hello'*128],prompt_tokens=128);assert list(ids[0])==binding['input_token_ids']
 from vllm import SamplingParams,TokensPrompt
 # Do not invoke generate_batch/run_benchmark or retain any timing result.
 outputs=backend._llm.generate([TokensPrompt(prompt_token_ids=list(ids[0]))],SamplingParams(temperature=0,top_p=1,ignore_eos=True,max_tokens=32),use_tqdm=False)
 tokens=list(outputs[0].outputs[0].token_ids);assert tokens==binding['generated_token_ids']
 result={'adapter_prepared':True,'public_engine_api_validated':True,'environment':backend.environment(),'prompt_tokens':128,'generated_tokens':tokens,'reference_exact':True,'functional_requests':1,'performance_trials':0}
finally:backend.close()
(r/'vllm-preparation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
