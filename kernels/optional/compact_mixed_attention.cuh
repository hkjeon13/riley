#pragma once
#include "pod_mixed_attention.cuh"
// Native experimental POD-style CTA role assignment. Caller owns immutable
// validated metadata/page extents and per-stream Plan until graph completion.
// Store FP32 exponentials once; round to BF16 only when loading PV operands.
// Reverse softmax and MMA order are unchanged; shared probability mirror removed.
namespace riley_compact_mixed {
using namespace riley_mixed_attention;
constexpr unsigned MAX_TASKS=1024*9, MAX_SM_IDS=4096, WARPS=4;
using Plan=riley_pod_mixed::Plan;
__device__ void compact_single_query(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const uint32_t* blocks,int row,int qh,float (*scores)[128],float (*exponentials)[128]){
 int lane=threadIdx.x%32,warp=0,group=lane/4,t=lane%4;
 int kvh=qh/3,count=n-rows+row+1;
 int qb=(row*9+qh)*64;
 if(n<rows || n>4096){out[qb+lane]=__float2bfloat16_rn(CUDART_NAN_F);out[qb+lane+32]=__float2bfloat16_rn(CUDART_NAN_F);return;}
 uint32_t query[4],query_hi[4];
 #pragma unroll
 for(int depth=0;depth<4;++depth){query[depth]=pair(q[qb+depth*16+2*t],q[qb+depth*16+2*t+1]);query_hi[depth]=pair(q[qb+depth*16+2*t+8],q[qb+depth*16+2*t+9]);}
 float maximum=-CUDART_INF_F,den=0.;float accum[8][4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  for(int token=begin;token<end;token+=8){
   float d[4]={};
   uint32_t key[4],key_hi[4];bool valid=token+group<end;
   int kb=valid?cache_index(token+group,kvh,0,blocks):0;
   #pragma unroll
   for(int depth=0;depth<4;++depth){key[depth]=valid?pair(k[kb+depth*16+2*t],k[kb+depth*16+2*t+1]):0;key_hi[depth]=valid?pair(k[kb+depth*16+2*t+8],k[kb+depth*16+2*t+9]):0;}
   #pragma unroll
   for(int depth=0;depth<4;++depth)mma(d,query[depth],query[depth],query_hi[depth],query_hi[depth],key[depth],key_hi[depth]);
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[warp][token-begin+2*t+j]=d[j]*.125F;}
  }
  __syncwarp();
  float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[warp][i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32){
   float p=i<end-begin?exponential(scores[warp][i],mx):0.;
   exponentials[warp][i]=p;
  }
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;
    if(i<end-begin)local_den+=exponentials[warp][i];}
  den=local_den;
  maximum=mx;
  __syncwarp();
  #pragma unroll
  for(int block=0;block<8;++block)for(int j=0;j<4;++j)accum[block][j]*=alpha;
  for(int token=begin;token<end;token+=16){
   int pi=token-begin;
   uint32_t a=pair(__float2bfloat16_rn(exponentials[warp][pi+2*t]),__float2bfloat16_rn(exponentials[warp][pi+2*t+1]));
   uint32_t aa=pair(__float2bfloat16_rn(exponentials[warp][pi+2*t+8]),__float2bfloat16_rn(exponentials[warp][pi+2*t+9]));
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
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}


template<int TileRows>
__device__ __forceinline__ void attention_body(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks,const uint32_t* live_rows,uint32_t query_block,uint32_t query_head,float (*scores)[128]){
 static_assert(TileRows==8||TileRows==16,"query tile");
 if(live_rows){uint32_t live=*live_rows;if(!live||live>static_cast<uint32_t>(rows))return;rows=live;}
 if(dynamic_n)n=*dynamic_n+1;
 float (*exps)[128]=scores;
 // A small prefill has insufficient queries to amortize the larger tile state.
 if(rows<32){
  if(query_block<static_cast<uint32_t>(rows))compact_single_query(q,k,v,out,rows,n,blocks,query_block,query_head,scores,exps);
  return;
 }
 const int first=query_block*TileRows;
 if(first>=rows)return;
 const int lane=threadIdx.x%32,group=lane/4,t=lane%4,qh=query_head,kvh=qh/3;
 const int qr[2]={first+group,first+group+8};
 const bool valid[2]={qr[0]<rows,TileRows==16&&qr[1]<rows};
 const int count[2]={n-rows+qr[0]+1,n-rows+qr[1]+1};
 if(n<rows||n>4096){
  for(int h=0;h<TileRows/8;++h)if(valid[h])for(int d=t;d<64;d+=4)out[(qr[h]*9+qh)*64+d]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }

 uint32_t query[2][4]={},query_hi[2][4]={};
 #pragma unroll
 for(int h=0;h<TileRows/8;++h){
  const int base=(qr[h]*9+qh)*64;
  #pragma unroll
  for(int depth=0;depth<4;++depth){
   query[h][depth]=valid[h]?pair(q[base+depth*16+2*t],q[base+depth*16+2*t+1]):0;
   query_hi[h][depth]=valid[h]?pair(q[base+depth*16+2*t+8],q[base+depth*16+2*t+9]):0;
  }
 }
 float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},den[2]={};
 float accum[8][4]={};
 const int last_count=n-rows+min(first+TileRows,rows);
 for(int tile=(last_count-1)/128;tile>=0;--tile){
  const int begin=tile*128,end=min(begin+128,last_count);
  for(int token=begin;token<end;token+=8){
   float d[4]={};uint32_t key[4],key_hi[4];
   const bool key_valid=token+group<end;
   const int kb=key_valid?cache_index(token+group,kvh,0,blocks):0;
   #pragma unroll
   for(int depth=0;depth<4;++depth){
    key[depth]=key_valid?pair(k[kb+depth*16+2*t],k[kb+depth*16+2*t+1]):0;
    key_hi[depth]=key_valid?pair(k[kb+depth*16+2*t+8],k[kb+depth*16+2*t+9]):0;
   }
   #pragma unroll
   for(int depth=0;depth<4;++depth)mma(d,query[0][depth],query[1][depth],query_hi[0][depth],query_hi[1][depth],key[depth],key_hi[depth]);
   #pragma unroll
   for(int h=0;h<TileRows/8;++h)for(int z=0;z<2;++z)if(token+2*t+z<end)scores[group+h*8][token-begin+2*t+z]=d[h*2+z]*.125F;
  }
  __syncwarp();
  float alpha[2]={1.F,1.F};
  #pragma unroll
  for(int h=0;h<TileRows/8;++h){
   const int own_end=valid[h]?min(end,count[h]):begin;
   const bool active=own_end>begin;
   float mx=maximum[h];
   for(int i=t;i<own_end-begin;i+=4)mx=fmaxf(mx,scores[group+h*8][i]);
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2,4));
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1,4));
   alpha[h]=active?exp2f((maximum[h]-mx)*1.4426950408889634F):1.F;
   for(int i=t;i<128;i+=4){
    float p=i<own_end-begin?exponential(scores[group+h*8][i],mx):0.F;
    exps[group+h*8][i]=p;
   }
   maximum[h]=mx;
  }
  __syncwarp();
  #pragma unroll
  for(int h=0;h<TileRows/8;++h){
   const int own_end=valid[h]?min(end,count[h]):begin;
   float local=den[h]*alpha[h];
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<own_end-begin)local+=exps[group+h*8][i];}
   den[h]=local;
  }
  #pragma unroll
  for(int b=0;b<8;++b)for(int j=0;j<4;++j)accum[b][j]*=alpha[j/2];
  for(int token=begin;token<end;token+=16){
   const int pi=token-begin;
   uint32_t a[2]={},aa[2]={};
   #pragma unroll
   for(int h=0;h<TileRows/8;++h){a[h]=pair(__float2bfloat16_rn(exps[group+h*8][pi+2*t]),__float2bfloat16_rn(exps[group+h*8][pi+2*t+1]));aa[h]=pair(__float2bfloat16_rn(exps[group+h*8][pi+2*t+8]),__float2bfloat16_rn(exps[group+h*8][pi+2*t+9]));}
   const int base=blocks?((blocks[token/16]*3+kvh)*16)*64:(token*3+kvh)*64;
   bool nonfinite_value=false;
   #pragma unroll
   for(int b=0;b<8;++b){
    const int dim=b*8+group;
    auto val=[&](int pos){return pos<end?v[base+(pos-token)*(blocks?64:192)+dim]:__float2bfloat16_rn(0.F);};
    uint32_t vb=pair(val(token+2*t),val(token+2*t+1)),vhi=pair(val(token+2*t+8),val(token+2*t+9));
    nonfinite_value=nonfinite_value||((vb&0x7f80U)==0x7f80U)||((vb&0x7f800000U)==0x7f800000U)||((vhi&0x7f80U)==0x7f80U)||((vhi&0x7f800000U)==0x7f800000U);
    float next[4];
    #pragma unroll
    for(int j=0;j<4;++j)next[j]=accum[b][j];
    mma(next,a[0],a[1],aa[0],aa[1],vb,vhi);
    #pragma unroll
    for(int j=0;j<4;++j)if(valid[j/2]&&token<count[j/2])accum[b][j]=next[j];
   }
   if(__any_sync(0xffffffff,nonfinite_value)){
    __syncwarp();
    for(int row=first;row<min(first+TileRows,rows);++row){
     compact_single_query(q,k,v,out,rows,n,blocks,row,qh,scores,exps);
     __syncwarp();
    }
    return;
   }
  }
  __syncwarp();
 }
 #pragma unroll
 for(int h=0;h<TileRows/8;++h){
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],2,4);
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],1,4);
  const float inverse=1.F/den[h];
  if(valid[h])for(int b=0;b<8;++b)for(int z=0;z<2;++z)out[(qr[h]*9+qh)*64+b*8+2*t+z]=__float2bfloat16_rn(accum[b][h*2+z]*inverse);
 }
}
__global__ void prepare(const unsigned* meta,unsigned capacity,Plan* p,bool telemetry) {
 for(unsigned i=threadIdx.x;i<MAX_SM_IDS;i+=blockDim.x){p->sm_ticks[i]=0;if(telemetry)p->sm_roles[i][0]=p->sm_roles[i][1]=0;}
 if(threadIdx.x==0){p->count[0]=p->count[1]=p->assigned[0]=p->assigned[1]=p->error=0;}
 __syncthreads();
 const unsigned active=meta[5],total=meta[9],tiles=meta[24];
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total)return;
 for(unsigned job=threadIdx.x;job<tiles*9;job+=blockDim.x){
  unsigned tile=job/9,head=job%9,entry=meta[32+32*416+1024+tile],owner=entry>>16,local=entry&0xffff;
  if(owner>=active)continue;
  const auto* s=meta+32+owner*416;unsigned count=s[2],offset=s[16],nt=count<32?count:(count+7)/8;
  if(!count||offset>total||count>total-offset||local>=nt)continue;
  unsigned role=s[18]==1?1:0;
  unsigned at=atomicAdd(p->count+role,1);p->jobs[role][at]=(tile<<4)|head;
 }
}
// Mode0 isolates queue packing; 1 is alternating-SM; 2 is proportional-SM.
template<unsigned Mode,bool Audit=false>
__global__ void execute(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,unsigned capacity,const unsigned* meta,Plan* p,unsigned* coverage=nullptr){
 const unsigned counts[2]={(p->count[0]+WARPS-1)/WARPS,(p->count[1]+WARPS-1)/WARPS};
 if(blockIdx.x>=counts[0]+counts[1])return;
 __shared__ unsigned chosen[2];
 __shared__ float scores[WARPS][8][128];
 if(threadIdx.x==0){
  unsigned role=0,index=blockIdx.x;
  if constexpr(Mode==0){role=index>=counts[0];if(role)index-=counts[0];}
  else {
   unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));
   if(sm>=MAX_SM_IDS){atomicExch(&p->error,1);role=0;index=counts[0];}
   else {
    unsigned ticket=atomicAdd(p->sm_ticks+sm,1);
    if constexpr(Mode==1)role=ticket&1;
    else {unsigned total=counts[0]+counts[1];role=((ticket+1)*counts[1]/total)!=(ticket*counts[1]/total);}
    index=atomicAdd(p->assigned+role,1);
    if(index>=counts[role]){role^=1;index=atomicAdd(p->assigned+role,1);}
   }
  }
  if constexpr(Audit)if(index<counts[role]){unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));if(sm<MAX_SM_IDS)atomicAdd(&p->sm_roles[sm][role],1);}
  chosen[0]=role;chosen[1]=index;
 }
 __syncthreads();
 const unsigned warp=threadIdx.x/32,role=chosen[0],task=chosen[1]*WARPS+warp;
 if(task>=p->count[role])return;
 const unsigned job=p->jobs[role][task],tile=job>>4,head=job&15;
 const unsigned entry=meta[32+32*416+1024+tile],owner=entry>>16,local=entry&0xffff;
 const auto* shape=meta+32+owner*416;const unsigned offset=shape[16];
 attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local,head,scores[warp]);
 if constexpr(Audit)if(threadIdx.x%32==0)atomicAdd(coverage+tile*9+head,1);
}
template<unsigned Mode,bool Audit=false>
inline cudaError_t enqueue(cudaStream_t s,const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,__nv_bfloat16*out,unsigned capacity,const unsigned*meta,Plan*p,unsigned*coverage=nullptr){
 if(!q||!k||!v||!out||!meta||!p||!capacity||capacity>1024)return cudaErrorInvalidValue;
 if constexpr(Audit)if(!coverage)return cudaErrorInvalidValue;
 riley_compact_mixed::prepare<<<1,256,0,s>>>(meta,capacity,p,Audit);
 auto err=cudaGetLastError();if(err!=cudaSuccess)return err;
 riley_compact_mixed::execute<Mode,Audit><<<(capacity*9+WARPS-1)/WARPS,32*WARPS,0,s>>>(q,k,v,out,capacity,meta,p,coverage);
 return cudaGetLastError();
}
template<bool Audit=false>
__global__ void mapped_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta,unsigned* coverage=nullptr){
 const uint32_t active=meta[5],total=meta[9],tiles=meta[24],tile=blockIdx.x;
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total||tile>=tiles)return;
 const uint32_t entry=meta[32+32*416+1024+tile],owner=entry>>16,local=entry&0xffffU;
 if(owner>=active)return;const uint32_t* shape=meta+32+owner*416;const uint32_t count=shape[2],offset=shape[16],nt=count<32?count:(count+7)/8;

 if(!count||offset>total||count>total-offset||local>=nt)return;
 __shared__ float scores[8][128];
 if constexpr(Audit)if(threadIdx.x==0)atomicAdd(coverage+tile*9+blockIdx.y,1);
 attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local,blockIdx.y,scores);
}

}
