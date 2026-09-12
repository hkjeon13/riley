from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/decode_shared_attention.cuh';s=p.read_text();s=s.replace('__shared__ __nv_bfloat16 probs[128];','__shared__ __nv_bfloat16 probs[128];\n __shared__ float exponentials[128];')
a='for(int i=lane;i<128;i+=32)probs[i]=__float2bfloat16_rn(i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.);'
b='''// Preserve unrounded exponentials for the original lane-local sum order.
  for(int i=lane;i<128;i+=32){float value=i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.;exponentials[i]=value;probs[i]=__float2bfloat16_rn(value);}
  __syncwarp();''';assert a in s;s=s.replace(a,b)
s=s.replace('local_den+=riley_prefill_shape::exponential(scores[head*4096+begin+i],mx)','local_den+=exponentials[i]')
a='int dim=block*8+g;auto val=[&](int pos){return pos<end?v[riley_prefill_shape::cache_index(pos,head/3,dim,pages)]:zero;};'
b='''// Each aligned K16 tile lies entirely in one physical KV page.
   int dim=block*8+g;int value_base=((pages[token/16]*3+head/3)*16)*64+dim;
   auto val=[&](int pos){return pos<end?v[value_base+(pos-token)*64]:zero;};''';assert a in s;s=s.replace(a,b);p.write_text(s)
