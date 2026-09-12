import hashlib,json
from pathlib import Path
ROOT=Path('/tmp/riley-opt-260912');BASE=Path('/tmp/riley-g04-vllm-profile-260911')
for label,p in [('baseline',BASE/'measurement-plan.json'),('candidate',ROOT/'candidate-plan.json')]:
 plan=json.loads(p.read_text())
 for path in [plan['engine_lanes']['vllm']['argv'][0],plan['reference_checker_python'],str(BASE/'vllm_lane.py'),str(BASE/'environment.json'),str(BASE/'candidate.json')]:
  plan['immutable_files'][path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 out=ROOT/(label+'-engine-plan.json')
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
