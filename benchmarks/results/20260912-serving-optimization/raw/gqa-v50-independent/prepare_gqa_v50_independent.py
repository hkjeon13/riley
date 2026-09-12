from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'gqa-attention-v50-independent';d.mkdir(exist_ok=True)
base=(r/'prefill-shapes-source-v11/kernels/src/decode_shared32_attention.cuh').read_text()
s=base[base.index('__global__ void values'):base.index('inline void enqueue')]
s=s.replace('void values','void independent_values').replace('row=blockIdx.y/9','row=blockIdx.y/3').replace('head=blockIdx.y%9,block=blockIdx.x,lane=threadIdx.x,g=lane/4,t=lane%4','warp=threadIdx.x/32,head=(blockIdx.y%3)*3+warp,block=blockIdx.x,lane=threadIdx.x%32,g=lane/4,t=lane%4')
s=s.replace('__shared__ __nv_bfloat16 probs[128];','__shared__ __nv_bfloat16 all_probs[3][128];auto* probs=all_probs[warp];').replace('__shared__ float exponentials[128];','__shared__ float all_exponentials[3][128];auto* exponentials=all_exponentials[warp];')
head=(r/'gqa-attention-v50/gqa_attention_v50.cuh').read_text();head=head[:head.rfind('}')]+s+'\n}\n';(d/'gqa_attention_v50.cuh').write_text(head)
p=(r/'gqa-attention-v50/expanded.cu').read_text().replace('if(variant&2)riley_gqa50_attention::values','if(variant>=4)riley_gqa50_attention::independent_values<<<dim3(8,96),96,0,stream>>>(sb,v,b,shape,shape+32,live);\n  else if(variant&2)riley_gqa50_attention::values')
p=p.replace('variant<=3','variant<=5').replace('execs[4]','execs[6]').replace('variant<4','variant<6').replace('order<4','order<6').replace('3-order','5-order')
(d/'expanded.cu').write_text(p)
