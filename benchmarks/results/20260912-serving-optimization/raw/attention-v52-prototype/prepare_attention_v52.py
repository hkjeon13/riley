from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'attention-state-v52';d.mkdir(exist_ok=True)
base=(r/'prefill-shapes-source-v11/kernels/src/mixed_attention_v49.cuh').read_text();(d/'baseline.cuh').write_text(base)
for variant in [1,2,3]:
 s=base.replace('namespace riley_mixed_attention','namespace riley_attention_v52_'+str(variant))
 if variant&1:
  s=s.replace('__shared__ float scores[16][128];','__shared__ float scores[TileRows][128];').replace('__shared__ float exps[16][128];','__shared__ float exps[TileRows][128];').replace('__shared__ __nv_bfloat16 probs[16][128];','__shared__ __nv_bfloat16 probs[TileRows][128];')
  s=s.replace('for(int h=0;h<2;++h)','for(int h=0;h<TileRows/8;++h)').replace('uint32_t query[2][4],query_hi[2][4];','uint32_t query[2][4]={},query_hi[2][4]={};').replace('float alpha[2];','float alpha[2]={1.F,1.F};').replace('uint32_t a[2],aa[2];','uint32_t a[2]={},aa[2]={};')
 if variant&2:
  s=s.replace('__shared__ float exps[TileRows][128];' if variant&1 else '__shared__ float exps[16][128];','float (*exps)[128]=scores;')
 (d/f'variant{variant}.cuh').write_text(s)
# Derive an established causal/nonfinite/mixed-owner oracle fixture, replacing only candidate dispatch and production packet extent.
p=(r/'mixed-attention-map-v49/probe.cu').read_text().replace('#include "mixed_attention_v49.cuh"','#include "baseline.cuh"\n#include "variant1.cuh"\n#include "variant2.cuh"\n#include "variant3.cuh"')
p=p.replace('32+32*416+1024','32+32*416+2048').replace('m[32+32*416+tiles+j]','m[14368+tiles+j]')
a=p.index('  for(unsigned blocks:');b=p.index('  if(timing)',a)
p=p[:a]+'''  for(unsigned variant:{0,1,2,3}){for(int i=0;i<capacity*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
  if(variant==0)riley_mixed_attention::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);
  if(variant==1)riley_attention_v52_1::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);
  if(variant==2)riley_attention_v52_2::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);
  if(variant==3)riley_attention_v52_3::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);
  CK(cudaDeviceSynchronize());for(int i=0;i<capacity*576+16;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch seed%d pattern%d owners%d exceptional%d variant%u at%d\\n",seed,pattern,owners,exceptional,variant,i);exit(3);}}
'''+p[b:]
a=p.index('if(variant==0)riley_mixed_attention::attention');b=p.index('CK(cudaStreamEndCapture',a)
p=p[:a]+'''if(variant==0)riley_mixed_attention::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);
if(variant==1)riley_attention_v52_1::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);
if(variant==2)riley_attention_v52_2::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);
if(variant==3)riley_attention_v52_3::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);
'''+p[b:]
p=p.replace('graph[5]','graph[4]').replace('exec[5]','exec[4]').replace('variant<5','variant<4').replace('order<5','order<4').replace('4-order','3-order').replace('i<5;','i<4;')
p=p.replace('riley_mixed_attention::compact_attention<<<dim3(128,9),32>>>','riley_attention_v52_3::mapped_attention<<<dim3(capacity,9),32>>>')
(d/'probe.cu').write_text(p)
