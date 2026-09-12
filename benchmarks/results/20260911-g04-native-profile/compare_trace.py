from pathlib import Path
import struct,json
r=Path(__file__).resolve().parent;s=r.parent/'20260911-g04-serving-trace'
data=(r/'native-trace.bin').read_bytes();size=4+30*3072
assert len(data)%size==0
records=[];seen=set()
for off in range(0,len(data),size):
 pos=struct.unpack_from('<I',data,off)[0]
 if pos in seen:continue
 seen.add(pos)
 call=0 if pos<=127 else pos-127
 offset=off+4
 for layer in range(30):
  for point,width in [('q',576),('k',192),('v',192),('context',576)]:
   a=data[offset:offset+width*2];offset+=width*2
   b=(s/f'layer{layer}-call{call}-{point}.bf16').read_bytes()
   if pos<=127:b=b[pos*width*2:(pos+1)*width*2]
   assert len(a)==len(b)
   count=sum(x!=y for x,y in zip(struct.iter_unpack('<H',a),struct.iter_unpack('<H',b)))
   records.append({'position':pos,'layer':layer,'point':point,'unequal':count})
(r/'native-serving-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
for pos in [0,127,128,135,155,156]:
 rows=[x for x in records if x['position']==pos]
 print(pos,'first difference',next((x for x in rows if x['unequal']),None))
 print('layer0',[(x['point'],x['unequal']) for x in rows if x['layer']==0])
