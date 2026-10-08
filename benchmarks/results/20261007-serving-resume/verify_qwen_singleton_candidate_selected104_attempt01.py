import json,hashlib,math
from pathlib import Path
R=Path('/Users/psyche/PycharmProjects/riley/benchmarks/results/20261007-serving-resume');C=R/'qwen-singleton-candidate-full-bounded-collection-attempt01';N=C/'qwen-full128-validation-attempt05';H=R/'qwen-step109-layer3-independent-localization-attempt02/qwen-step109-layer3-hf-stage-validation-attempt02'
sha=lambda b:hashlib.sha256(b).hexdigest();n=json.loads((N/'qwen3b-p2048-cache-on-full128-native-result.json').read_bytes());h=json.loads((H/'result.json').read_bytes());nr=(N/Path(n['selected_stage_raw_sidecar']['path']).name).read_bytes();hr=(H/'selected-stages.bf16').read_bytes()
assert n['source_commit']=='2612208b9ac02ffa79b563b87f073c239b35da18';assert h['source110_logits_all_exact'] and n['selected_layer_index']==h['selected_layer_index']==3
assert sha(nr)==n['selected_stage_raw_sidecar']['sha256'] and len(nr)==n['selected_stage_raw_sidecar']['bytes'];assert sha(hr)==h['raw_sha256'];assert len(n['selected_stages'])==len(h['selected_stages'])==104 and set(n['selected_stages'])==set(h['selected_stages'])
spans={'HF':[],'native':[]};rows=[]
for name,meta in h['selected_stages'].items():
 native=n['selected_stages'][name];a,b=meta['raw_data_offsets'];x,y=native['raw_data_offsets'];assert meta['dtype']=='BF16' and b-a==y-x==native['bytes']==2*math.prod(meta['shape']);assert 0<=a<b<=len(hr) and 0<=x<y<=len(nr)
 hb=hr[a:b];nb=nr[x:y];assert sha(hb)==meta['sha256'] and sha(nb)==native['sha256'];spans['HF'].append((a,b));spans['native'].append((x,y));rows.append({'name':name,'exact':nb==hb,'unequal_BF16_elements':sum(nb[i:i+2]!=hb[i:i+2] for i in range(0,len(nb),2))})
for key,length in [('HF',len(hr)),('native',len(nr))]:
 cursor=0
 for a,b in sorted(spans[key]):assert a==cursor;cursor=b
 assert cursor==length
result={'selected_raw_stages_independently_replayed':104,'exact_stages':sum(x['exact'] for x in rows),'mismatches':[x for x in rows if not x['exact']],'HF_result_sha256':sha((H/'result.json').read_bytes()),'HF_selected_raw_sha256':sha(hr),'native_selected_raw_sha256':sha(nr),'source_commit':n['source_commit'],'serving_performance':'미실행','goal_achieved':False}
(C/'selected104-independent-proof.json').write_text(json.dumps(result,indent=2)+'\n');assert result['exact_stages']==104;print(json.dumps(result,ensure_ascii=False))
