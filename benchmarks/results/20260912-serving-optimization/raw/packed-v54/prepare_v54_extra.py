from pathlib import Path
import subprocess
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11';d=r/'packed-value-v54'
old=subprocess.check_output(['git','show','HEAD:kernels/src/decode_gqa_attention_v50.cuh'],cwd=s,text=True).replace('riley_gqa50_attention','riley_gqa52_reference')
(d/'decode-baseline.cuh').write_text(old)
p=(r/'gqa-attention-v50-independent/expanded.cu').read_text().replace('#include "gqa_attention_v50.cuh"','#include "decode-baseline.cuh"\n#include "decode_gqa_attention_v50.cuh"')
p=p.replace('auto*a=alloc','auto*vp=alloc<__nv_bfloat16>(4096*32*192);\n auto*a=alloc',1)
a=p.index('  if(variant&1)');b=p.index('\n };',a)
p=p[:a]+'''  if(variant)riley_gqa50_attention::enqueue(stream,q,k,vp,b,sb,shape,shape+32,live,4096);
  else riley_gqa52_reference::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);'''+p[b:]
needle='v[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/64);}'
p=p.replace(needle,needle+'for(int i=0;i<4096*32*192;++i){int t=(i%1024)/64,dim=i%64;vp[(i/1024)*1024+(dim/8)*128+(t/8)*64+(dim%8)*8+t%8]=v[i];}')
p=p.replace('variant<=5','variant<=1').replace('execs[6]','execs[2]').replace('variant<6','variant<2').replace('order<6','order<2').replace('5-order','1-order').replace('(void*)v,(void*)a','(void*)v,(void*)vp,(void*)a')
(d/'decode.cu').write_text(p)
old=subprocess.check_output(['git','show','HEAD:kernels/src/mixed_rope_v49.cuh'],cwd=s,text=True).replace('mixed_rope_kv_v7','mixed_rope_kv_reference')
(d/'rope-baseline.cuh').write_text(old)
