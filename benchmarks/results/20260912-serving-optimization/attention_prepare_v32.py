from pathlib import Path
r=Path('/tmp/riley-opt-260912');p=r/'prefill-shapes-source-v11/kernels/src/decode_shared_attention.cuh';s=p.read_text();a=s.index('// Independent output');b=s.index('\ninline void enqueue',a)
s=s[:a]+'''// Each 128-float score tile becomes 64 packed BF16 pairs plus its FP32
// online scale. Packing stays within that tile, so unread lower tiles survive.
__global__ void prepare_probabilities(float* scratch,const uint32_t* shape,const uint32_t* live_rows){
 int row=blockIdx.x/9,head=blockIdx.x%9,lane=threadIdx.x,t=lane%4;
 uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;
 int count=shape[row*416+1]+1;if(count<1||count>4096)return;
 float* scores=scratch+(row*9+head)*4096;
 __shared__ float exponentials[128];float maximum=-CUDART_INF_F,den=0.;
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,n=min(128,count-begin);float mx=maximum;
  for(int i=lane;i<n;i+=32)mx=fmaxf(mx,scores[begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32)exponentials[i]=i<n?riley_prefill_shape::exponential(scores[begin+i],mx):0.;
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<n)local_den+=exponentials[i];}
  den=local_den;maximum=mx;
  for(int i=lane;i<64;i+=32)reinterpret_cast<uint32_t*>(scores+begin)[i]=riley_prefill_shape::pair(__float2bfloat16_rn(exponentials[2*i]),__float2bfloat16_rn(exponentials[2*i+1]));
  if(lane==0)scores[begin+64]=alpha;
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 if(lane<4)scores[65+lane]=den;
}
// Preserve all output CTAs, ordered K16 MMA steps and online rescaling.
__global__ void values(const float* scratch,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/9;uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;
 shape+=row*416;pages+=row*416;out+=row*576;
 int count=shape[1]+1,head=blockIdx.y%9,block=blockIdx.x,lane=threadIdx.x,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 const float* packed=scratch+(row*9+head)*4096;float accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float alpha=packed[begin+64];
  for(int j=0;j<4;++j)accum[j]*=alpha;
  const uint32_t* probs=reinterpret_cast<const uint32_t*>(packed+begin);
  for(int token=begin;token<end;token+=16){
   int pi=(token-begin)/2;uint32_t a=probs[pi+t],aa=probs[pi+t+4];
   int dim=block*8+g,value_base=((pages[token/16]*3+head/3)*16)*64+dim;
   auto val=[&](int pos){return pos<end?v[value_base+(pos-token)*64]:zero;};
   uint32_t b=riley_prefill_shape::pair(val(token+2*t),val(token+2*t+1));
   uint32_t bb=riley_prefill_shape::pair(val(token+2*t+8),val(token+2*t+9));
   riley_prefill_shape::mma(accum,a,a,aa,aa,b,bb);
  }
 }
 float inverse=1.0F/packed[65+t];
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}
''' +s[b:]
s=s.replace(' values<<<',' prepare_probabilities<<<72,32,0,stream>>>(scratch,shape,live_rows);\n values<<<');p.write_text(s)
s=(r/'build_test_v31.py').read_text().replace('v31','v32');s=s.replace("run('shared-v32-owned.log'","run('shared-v32-racecheck.log',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','racecheck','--error-exitcode','99',str(r/'shared-primitive-v32')])\nrun('shared-v32-owned.log'");(r/'build_test_v32.py').write_text(s)
for suffix in ('','_c8'):(r/f'run_v3_http_shared_v32{suffix}.py').write_text((r/f'run_v3_http_shared_v31{suffix}.py').read_text().replace('v31','v32'))
