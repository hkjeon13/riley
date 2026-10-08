import subprocess,json,hashlib
from pathlib import Path
R=Path('/data/riley-serving-261007');I=R/'qwen-layer3-isolated-norm-inputs-attempt01';O=R/'qwen-layer3-isolated-norm-hf-validation-attempt01';O.mkdir();failure=None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EXPECTED={'step109-input.bf16': '4d92b2bb7a1c5bc492334e33fc4f979c36f5e7da24c301d89de81be329f39227', 'step108-native_output.bf16': 'fadb0adf0a2569890f41b658e9902f9c762c819c294c72d6e080eb049bec9897', 'step108-input.bf16': 'd561acb2e513ce9a69ad4bcf16da8c4f9724c4d6c2beeb956daca165e7322a87', 'step108-HF_output.bf16': 'fadb0adf0a2569890f41b658e9902f9c762c819c294c72d6e080eb049bec9897', 'step109-native_output.bf16': 'db35c6b66e31759d1c6ab8d9da8af101c5c8b7565d9f065f152459456fbb7551', 'manifest.json': '53824d24e1d5ecd2d97e83846f5c40c25bbb7031f95975ebb1f1d5832e0c6ef2', 'step109-HF_output.bf16': 'e65d03769e8f45570c7bea256b7ceffeb69f3ba99dfc43db349fd6b53faedeed'}
SCRIPT_SHA='a40c0a16e3719ee9553ac3f892f97cd99b695acd6186946c5ae073a11fb9d66d'
try:
 assert json.loads((R/'kernel-batch04-profiling-attempt01/completion.json').read_text())['failure'] is None
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 script=R/'qwen_layer3_isolated_norm_hf_probe_attempt01.py';assert sha(script)==SCRIPT_SHA
 for name,h in EXPECTED.items():assert sha(I/name)==h
 model=Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
 files=[model/'config.json'];index=model/'model.safetensors.index.json'
 if index.exists():files.extend([index,model/json.loads(index.read_text())['weight_map']['model.layers.3.post_attention_layernorm.weight']])
 else:files.append(model/'model.safetensors')
 pins={str(p):sha(p) for p in files};pins.update({str(I/n):h for n,h in EXPECTED.items()});pins[str(script)]=SCRIPT_SHA
 image='vllm/vllm-openai@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2'
 argv=['docker','run','--rm','--gpus','device=0','--network','none','--entrypoint','python3','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','PYTHONDONTWRITEBYTECODE=1','-v',str(script)+':/probe.py:ro','-v',str(model)+':/checkpoint:ro','-v',str(I)+':/inputs:ro','-v',str(O)+':/output',image,'/probe.py']
 (O/'preparation.json').write_text(json.dumps({'argv':argv,'pins':pins,'gpu_before':subprocess.check_output(['nvidia-smi','-q'],text=True),'serving_performance':'미실행'},indent=2)+'\n')
 with (O/'container.log').open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 after={path:sha(path) for path in pins};(O/'pins-after.json').write_text(json.dumps(after,indent=2)+'\n');assert pins==after,'immutable input drift'
 assert p.returncode==0,'isolated HF replay failed; preserve evidence'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'controller-completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
