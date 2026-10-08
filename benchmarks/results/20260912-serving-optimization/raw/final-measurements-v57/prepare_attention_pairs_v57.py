from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11/kernels/src';d=r/'attention-pairs-v57';d.mkdir(exist_ok=True)
mb=(s/'mixed_attention_v49.cuh').read_text();db=(s/'decode_gqa_attention_v50.cuh').read_text()
(d/'baseline.cuh').write_text(mb);(d/'decode-baseline.cuh').write_text(db.replace('riley_gqa50_attention','riley_gqa_reference_v57'))
helper='\n// All callers supply adjacent BF16 words at a four-byte aligned address.\n__device__ __forceinline__ uint32_t load_pair(const __nv_bfloat16* p){return *reinterpret_cast<const uint32_t*>(p);}\n'
def replace_pair(t,old,new):
 assert old in t,old;return t.replace(old,new)
for v in [1,2,3]:
 m=mb.replace('namespace riley_mixed_attention {','namespace riley_mixed_v57_'+str(v)+' {'+helper)
 q=db.replace('namespace riley_gqa50_attention {','namespace riley_decode_v57_'+str(v)+' {'+helper)
 if v in [1,3]:
  for base in ['qb','base']:
   for off in ['', '+8']:
    low=f'{base}+depth*16+2*t'+off;high=f'{base}+depth*16+2*t+'+('1' if not off else '9')
    m=replace_pair(m,f'pair(q[{low}],q[{high}])',f'load_pair(q+{low})')
  for off,last in [('', '1'),('+8','9')]:
   old=f'pair(k[kb+depth*16+2*t{off}],k[kb+depth*16+2*t+{last}])'
   m=replace_pair(m,old,f'load_pair(k+kb+depth*16+2*t{off})')
   q=replace_pair(q,'riley_prefill_shape::'+old,f'load_pair(k+kb+depth*16+2*t{off})')
   q=replace_pair(q,f'riley_prefill_shape::pair(qp[2*t{off}],qp[2*t+{last}])',f'load_pair(qp+2*t{off})')
 if v in [2,3]:
  m=m.replace('__shared__ __nv_bfloat16 probs','__shared__ __align__(4) __nv_bfloat16 probs')
  q=q.replace('__shared__ __nv_bfloat16 all_probs','__shared__ __align__(4) __nv_bfloat16 all_probs')
  for row in ['warp','group+h*8']:
   for off,last in [('', '1'),('+8','9')]:m=replace_pair(m,f'pair(probs[{row}][pi+2*t{off}],probs[{row}][pi+2*t+{last}])',f'load_pair(probs[{row}]+pi+2*t{off})')
  for off,last in [('', '1'),('+8','9')]:q=replace_pair(q,f'riley_prefill_shape::pair(probs[pi+2*t{off}],probs[pi+2*t+{last}])',f'load_pair(probs+pi+2*t{off})')
 (d/f'variant{v}.cuh').write_text(m);(d/f'decode{v}.cuh').write_text(q)
for src,dst in [('probe.cu','mixed.cu'),('boundary.cu','boundary.cu')]:
 p=(r/'attention-state-v52'/src).read_text()
 for v in [1,2,3]:p=p.replace(f'riley_attention_v52_{v}',f'riley_mixed_v57_{v}')
 (d/dst).write_text(p)
p=(r/'gqa-attention-v50-independent/expanded.cu').read_text().replace('#include "gqa_attention_v50.cuh"','#include "decode-baseline.cuh"\n'+''.join(f'#include "decode{i}.cuh"\n' for i in [1,2,3]))
a=p.index('  if(variant&1)');b=p.index('\n };',a)
p=p[:a]+'''  if(variant==0)riley_gqa_reference_v57::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);
'''+''.join(f'  if(variant=={i})riley_decode_v57_{i}::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);\n' for i in [1,2,3])+p[b:]
p=p.replace('variant<=5','variant<=3').replace('execs[6]','execs[4]').replace('variant<6','variant<4').replace('order<6','order<4').replace('5-order','3-order')
p=p.replace('for(int rows:{0,1,4,8,16,24,32,33}){','for(int rows:{0,1,4,8,16,24,32,33})for(int exceptional:{0,1,2,3}){').replace('if(timing&&(context<128','if(timing&&(exceptional||context<128')
needle='   for(int i=0;i<32*576+16;++i)a[i]'
insert='''   int pos=shape[1],at=((shape[32+pos/16]*3)*16+pos%16)*64+seed%64;auto oldk=k[at],oldq=q[seed%576];
   if(exceptional==1)k[at]=__ushort_as_bfloat16(0x7fc1);if(exceptional==2)k[at]=__ushort_as_bfloat16(0x7f80);if(exceptional==3)q[seed%576]=__ushort_as_bfloat16(0x7fc1);
''';assert needle in p;p=p.replace(needle,insert+needle)
needle='    for(auto graph_exec:execs)OK(cudaGraphExecDestroy(graph_exec));OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));\n   }';assert needle in p;p=p.replace(needle,needle+'\n   k[at]=oldk;q[seed%576]=oldq;')
p=p.replace('exact_scores_outputs_padding=true','exact_scores_outputs_padding=true finite_nonfinite_Q_K=true')
(d/'decode.cu').write_text(p)
