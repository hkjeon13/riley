from pathlib import Path
import struct,json,re
r=Path('/tmp/riley-g04-native-profile-260911');s=Path('/tmp/riley-g04-serving-trace-260911')
data=(r/'native-trace-all-final.bin').read_bytes();size=92164;seen=set();records=[]
assert len(data)%size==0
for off in range(0,len(data),size):
 pos=struct.unpack_from('<I',data,off)[0]
 if pos in seen:continue
 seen.add(pos);call=0 if pos<128 else pos-127;offset=off+4
 for layer in range(30):
  for name,width in [('q',576),('k',192),('v',192),('context',576)]:
   a=data[offset:offset+width*2];offset+=width*2;b=(s/f'layer{layer}-call{call}-{name}.bf16').read_bytes()
   if pos<128:b=b[pos*width*2:(pos+1)*width*2]
   assert len(a)==len(b)
   n=sum(x!=y for x,y in zip(struct.iter_unpack('<H',a),struct.iter_unpack('<H',b)))
   records.append(dict(position=pos,layer=layer,point=name,unequal=n))
assert seen==set(range(159))
ref=json.loads((s/'invariance.json').read_text())['before'];log=(r/'graph-all-final-test.log').read_text();tokens=json.loads(re.search(r'output_tokens=(\[[^\n]+\])',log)[1]);assert tokens==ref
result=dict(native_prefill=True,oracle_input=False,external_kv_input=False,positions=159,tensor_comparisons=len(records),unequal=sum(x['unequal'] for x in records),tokens_exact=tokens==ref,tokens=tokens,records=records,performance_trials=0)
(r/'native-all-final-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='records'})
print('first_difference',next((x for x in records if x['unequal']),None))
