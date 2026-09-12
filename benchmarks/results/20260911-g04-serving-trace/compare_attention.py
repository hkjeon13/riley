import json,struct,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);candidate=Path(sys.argv[2])
def unpack(raw): return [struct.unpack('<f',struct.pack('<I',b[0]<<16))[0] for b in struct.iter_unpack('<H',raw)]
records=[]
for layer in range(30):
 for call in range(9):
  a=(root/f'layer{layer}-call{call}-context.bf16').read_bytes()
  b=(candidate/f'layer{layer}-call{call}-riley-context.bf16').read_bytes()
  assert len(a)==len(b)
  x,y=unpack(a),unpack(b)
  records.append({'layer':layer,'call':call,'elements':len(x),'unequal':sum(u!=v for u,v in zip(x,y)),
    'max_abs':max(abs(u-v) for u,v in zip(x,y)),
    'vllm_sha256':hashlib.sha256(a).hexdigest(),'riley_sha256':hashlib.sha256(b).hexdigest()})
out={'records':records,'exact_cases':sum(x['unequal']==0 for x in records),
     'total_unequal':sum(x['unequal'] for x in records),'performance_trials':0}
(candidate/'attention-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
print('layer0',[(x['call'],x['unequal'],x['max_abs']) for x in records if x['layer']==0])
