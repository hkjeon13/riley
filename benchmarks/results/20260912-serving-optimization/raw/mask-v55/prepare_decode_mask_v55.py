from pathlib import Path
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11/kernels/src';d=r/'decode-mask-v55';d.mkdir(exist_ok=True)
base=(s/'decode_gqa_attention_v50.cuh').read_text()
(d/'baseline.cuh').write_text((r/'packed-value-v54/decode-baseline.cuh').read_text())
(d/'packed-v54.cuh').write_text(base.replace('riley_gqa50_attention','riley_packed54_reference'))
helper='''// The caller supplies an allocated aligned pair in a valid physical KV page.
// Mask integer bits after loading; future BF16 NaN/Inf must become exact zero.
__device__ __forceinline__ uint32_t masked(const __nv_bfloat16* p,int pos,int end){
 uint32_t bits=*reinterpret_cast<const uint32_t*>(p);
 uint32_t mask=(0xffffU*uint32_t(pos<end))|(0xffff0000U*uint32_t(pos+1<end));
 return bits&mask;
}
'''
for v in [1,2,3]:
 p=base.replace('riley_gqa50_attention',f'riley_decode_mask_v55_{v}')
 p=p.replace('// Parallel QK',helper+'\n// Parallel QK')
 if v in [1,3]:p=p.replace('riley_packed_value_v54::masked_pair','masked')
 if v==2:
  p=p.replace('vb[part]=live?riley_packed_value_v54::masked_pair(v+vi+lane*2,at+2*t,end):0;\n    vbb[part]=live?riley_packed_value_v54::masked_pair(v+vi+64+lane*2,at+2*t+8,end):0;', '''if(at+16<=end){vb[part]=*reinterpret_cast<const uint32_t*>(v+vi+lane*2);vbb[part]=*reinterpret_cast<const uint32_t*>(v+vi+64+lane*2);}
    else{vb[part]=live?masked(v+vi+lane*2,at+2*t,end):0;vbb[part]=live?masked(v+vi+64+lane*2,at+2*t+8,end):0;}''')
 if v==3:
  a=p.index('  #pragma unroll 1\n  for(int token=begin;token<end;token+=64)');b=p.index('  __syncwarp();',a)
  p=p[:a]+'''  #pragma unroll 1
  for(int at=begin;at<end;at+=16){
   int pi=at-begin;
   uint32_t pa=riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]);
   uint32_t paa=riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]);
   int vi=(pages[at/16]*3+head/3)*1024+block*128;
   uint32_t vb=masked(v+vi+lane*2,at+2*t,end),vbb=masked(v+vi+64+lane*2,at+2*t+8,end);
   riley_prefill_shape::mma(accum,pa,pa,paa,paa,vb,vbb);
  }
'''+p[b:]
 (d/f'variant{v}.cuh').write_text(p)
p=(r/'packed-value-v54/decode.cu').read_text();a=p.index('#include "decode-baseline.cuh"');b=p.index('#include "decode_shared32_attention.cuh"')
p=p[:a]+''.join('#include "'+n+'"\n' for n in ['baseline.cuh','packed-v54.cuh','variant1.cuh','variant2.cuh','variant3.cuh'])+p[b:]
a=p.index('  if(variant)riley_gqa50_attention');b=p.index('\n };',a)
p=p[:a]+'''  if(variant==0)riley_gqa52_reference::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);
  if(variant==1)riley_packed54_reference::enqueue(stream,q,k,vp,b,sb,shape,shape+32,live,4096);
'''+''.join(f'  if(variant=={i+1})riley_decode_mask_v55_{i}::enqueue(stream,q,k,vp,b,sb,shape,shape+32,live,4096);\n' for i in [1,2,3])+p[b:]
p=p.replace('variant<=1','variant<=4').replace('execs[2]','execs[5]').replace('variant<2','variant<5').replace('order<2','order<5').replace('1-order','4-order')
(d/'probe.cu').write_text(p)
