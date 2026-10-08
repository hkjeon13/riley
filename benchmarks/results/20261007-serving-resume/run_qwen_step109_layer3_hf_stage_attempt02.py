import subprocess,json,time,hashlib
from pathlib import Path
R=Path('/data/riley-serving-261007');O=R/'qwen-step109-layer3-hf-stage-validation-attempt02';O.mkdir();failure=None
base=Path('/data/riley-benchmarks/20260915T134348Z-n06a-shared-host');hf=base/'qwen3b-p2048-cacheon-layer-stage-r7-20260917T141511Z';teacher=base/'qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z'
image='vllm/vllm-openai@sha256:7ef5a35d1ef8ce2cf9d671dd91eec6e367c5849262e0362b4d3d4a26be0d87d2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
 assert json.loads((R/'kernel-batch04-server-validation-attempt01/completion.json').read_text())['failure'] is None,'Batch04 native/HTTP correctness terminal missing'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied; leave existing actors unchanged'
 script=R/'qwen_step109_layer3_hf_stage_probe_attempt02.py';source=R/'qwen-full128-source-attempt01';model=Path('/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1')
 pins={str(p):sha(p) for p in [script,hf/'qwen3b-p2048-cache-on-layer-stage.json',hf/'qwen3b-p2048-cache-on-layer-stage.safetensors',teacher/'teacher-forced-oracle.json',teacher/'cache-on-logits.safetensors']}
 argv=['docker','run','--rm','--gpus','device=0','--network','none','--entrypoint','python3','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','CUBLAS_WORKSPACE_CONFIG=:4096:8','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/repo/tools/python/reference','-v',str(source)+':/repo:ro','-v',str(script)+':/probe.py:ro','-v',str(model)+':/checkpoint:ro','-v',str(base/'inputs')+':/inputs:ro','-v',str(teacher)+':/teacher:ro','-v',str(hf)+':/evidence:ro','-v',str(O)+':/output',image,'/probe.py']
 (O/'preparation.json').write_text(json.dumps({'argv':argv,'pins':pins,'scope':'observer110 HF cache-on steps; every logits row exact before accepting selected108/109 reference','serving_performance':'미실행'},indent=2)+'\n')
 with (O/'container.log').open('x') as f:p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 assert all(sha(path)==h for path,h in pins.items()),'immutable input or observer changed'
 assert p.returncode==0,'HF observer failed; preserve source/log; no replacement reference accepted'
except BaseException as e:failure={'type':type(e).__name__,'message':str(e)};raise
finally:(O/'controller-completion.json').write_text(json.dumps({'failure':failure,'serving_performance':'미실행','goal_achieved':False},indent=2)+'\n')
