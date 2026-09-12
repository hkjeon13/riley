import pathlib,json,struct,hashlib,math
r=pathlib.Path('/tmp/riley-opt-260912');out=r/'prefill-full-model-v11';out.mkdir(exist_ok=True)
p=pathlib.Path('/data/riley-benchmark/20260827T051948Z-d7ad713a/model/model.safetensors')
with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));base=8+n
names=['model.embed_tokens.weight','model.norm.weight',None]
roles=['input_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','post_attention_layernorm','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
for i in range(30):names.extend(f'model.layers.{i}.{role}.weight' for role in roles)
with p.open('rb') as src,(out/'weights.bin').open('wb') as f:
 for name in names:
  if name is None:f.write(struct.pack('<Q',0));continue
  x=h[name];assert x['dtype']=='BF16';a,b=x['data_offsets'];src.seek(base+a);data=src.read(b-a);assert len(data)==b-a;f.write(struct.pack('<Q',len(data)));f.write(data)
corpus=json.loads((r/'variable-corpus-v11/requests.json').read_text());chosen=[min(corpus,key=lambda x:abs(x['prompt_tokens']-target)) for target in [16,128,398]]
with (out/'requests.bin').open('wb') as f:
 f.write(struct.pack('<I',len(chosen)))
 for x in chosen:f.write(struct.pack('<I',len(x['token_ids'])));f.write(struct.pack('<'+'I'*len(x['token_ids']),*x['token_ids']))
with (out/'rope.bin').open('wb') as f:
 for fn in [math.cos,math.sin]:
  for pos in range(1024):
   for d in range(32):f.write(struct.pack('<f',fn(pos/(100000**(2*d/64)))))
(out/'manifest.json').write_text(json.dumps({'source_weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'weight_order':names,'prompt_lengths':[x['prompt_tokens'] for x in chosen],'purpose':'full versus chunked checkpoint prefill parity; no head or serving qualification'},indent=2)+'\n')
print([x['prompt_tokens'] for x in chosen])
