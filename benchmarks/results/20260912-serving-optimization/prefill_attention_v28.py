from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'kernels/src/prefill_shape_attention.cuh';s=p.read_text();s=s.replace('__shared__ __nv_bfloat16 probs[3][128];','__shared__ __nv_bfloat16 probs[3][128];\n __shared__ float exponentials[3][128];\n uint32_t query[4],query_hi[4];\n #pragma unroll\n for(int depth=0;depth<4;++depth){query[depth]=pair(q[qb+depth*16+2*t],q[qb+depth*16+2*t+1]);query_hi[depth]=pair(q[qb+depth*16+2*t+8],q[qb+depth*16+2*t+9]);}')
s=s.replace('uint32_t a=pair(q[qb+depth+2*t],q[qb+depth+2*t+1]);','uint32_t a=query[depth/16];').replace('uint32_t aa=pair(q[qb+depth+2*t+8],q[qb+depth+2*t+9]);','uint32_t aa=query_hi[depth/16];')
a=s.index('  float local_den=0.;');b=s.index('  den=local_den;',a)
s=s[:a]+'''  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   exponentials[warp][i]=p;probs[warp][i]=__float2bfloat16_rn(p);
  }
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponentials[warp][i];}
'''+s[b:]
a=s.index('  for(int block=0;block<8;++block){');b=s.index('  __syncwarp();',a)
s=s[:a]+'''  #pragma unroll
  for(int block=0;block<8;++block)for(int j=0;j<4;++j)accum[block][j]*=alpha;
  for(int token=begin;token<end;token+=16){
   int pi=token-begin;
   uint32_t a=pair(probs[warp][pi+2*t],probs[warp][pi+2*t+1]);
   uint32_t aa=pair(probs[warp][pi+2*t+8],probs[warp][pi+2*t+9]);
   // A K16 tile is aligned to one KV page; reuse its base and probabilities.
   int page_base=blocks?((blocks[token/16]*3+kvh)*16)*64:(token*3+kvh)*64;
   #pragma unroll
   for(int block=0;block<8;++block){
    int dim=block*8+group;
    auto val=[&](int pos){return pos<end?v[page_base+(pos-token)*(blocks?64:192)+dim]:zero;};
    uint32_t b=pair(val(token+2*t),val(token+2*t+1));
    uint32_t bb=pair(val(token+2*t+8),val(token+2*t+9));
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
'''+s[b:];p.write_text(s)
