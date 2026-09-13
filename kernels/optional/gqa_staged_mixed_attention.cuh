#pragma once
#include "compact_mixed_attention.cuh"
// Isolated GQA staging candidate. No serving/backend default changes.
// Three warps own distinct query heads of the same KV head and query tile.
namespace riley_gqa_staging {
using riley_prefill_shape::pair;
using riley_prefill_shape::mma;
using riley_prefill_shape::cache_index;
using riley_prefill_shape::exponential;
using riley_compact_mixed::compact_single_query;
template<int TileRows>
__device__ __forceinline__ void staged_body(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks,const uint32_t* live_rows,uint32_t query_block,uint32_t query_head,float (*scores)[128],__nv_bfloat16* staged_k,__nv_bfloat16* staged_v){
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
  // All three head warps share this query range and execute every CTA barrier.
  // 16-byte copies preserve BF16 storage exactly; zero-fill the causal tail.
  for(int vector=threadIdx.x;vector<128*8;vector+=blockDim.x){
   const int token=begin+vector/8,dim=(vector%8)*8;
   const bool active=token<end;
   const int global=active?cache_index(token,kvh,dim,blocks):0;
   const unsigned kd=static_cast<unsigned>(__cvta_generic_to_shared(staged_k+vector*8));
   const unsigned vd=static_cast<unsigned>(__cvta_generic_to_shared(staged_v+vector*8));
   const int bytes=active?16:0;
   asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(kd),"l"(k+global),"r"(bytes));
   asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"::"r"(vd),"l"(v+global),"r"(bytes));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  bool bad=false;
  for(int i=threadIdx.x;i<(end-begin)*64;i+=blockDim.x)
   bad=bad||((__bfloat16_as_ushort(staged_v[i])&0x7f80U)==0x7f80U);
  if(__syncthreads_or(bad)){
   // Restart all heads together on the original ordered nonfinite-safe path.
   riley_compact_mixed::attention_body<TileRows>(q,k,v,out,rows,n,nullptr,blocks,nullptr,query_block,query_head,scores);
   return;
  }

  for(int token=begin;token<end;token+=8){
   float d[4]={};uint32_t key[4],key_hi[4];
   const bool key_valid=token+group<end;
   const int kb=(token+group-begin)*64;
   #pragma unroll
   for(int depth=0;depth<4;++depth){
    key[depth]=key_valid?pair(staged_k[kb+depth*16+2*t],staged_k[kb+depth*16+2*t+1]):0;
    key_hi[depth]=key_valid?pair(staged_k[kb+depth*16+2*t+8],staged_k[kb+depth*16+2*t+9]):0;
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
   #pragma unroll
   for(int b=0;b<8;++b){
    const int dim=b*8+group;
    auto val=[&](int pos){return pos<end?staged_v[(pos-begin)*64+dim]:__float2bfloat16_rn(0.F);};
    uint32_t vb=pair(val(token+2*t),val(token+2*t+1)),vhi=pair(val(token+2*t+8),val(token+2*t+9));
    float next[4];
    #pragma unroll
    for(int j=0;j<4;++j)next[j]=accum[b][j];
    mma(next,a[0],a[1],aa[0],aa[1],vb,vhi);
    #pragma unroll
    for(int j=0;j<4;++j)if(valid[j/2]&&token<count[j/2])accum[b][j]=next[j];
   }

  }
  __syncthreads();
 }
 #pragma unroll
 for(int h=0;h<TileRows/8;++h){
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],2,4);
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],1,4);
  const float inverse=1.F/den[h];
  if(valid[h])for(int b=0;b<8;++b)for(int z=0;z<2;++z)out[(qr[h]*9+qh)*64+b*8+2*t+z]=__float2bfloat16_rn(accum[b][h*2+z]*inverse);
 }
}

template<bool Staged,bool Audit=false>
__global__ void mapped(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta,unsigned* coverage=nullptr){
 const unsigned active=meta[5],total=meta[9],tiles=meta[24],tile=blockIdx.x;
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total||tile>=tiles)return;
 const unsigned entry=meta[32+32*416+1024+tile],owner=entry>>16,local=entry&0xffffU;
 if(owner>=active)return;
 const unsigned* shape=meta+32+owner*416;
 const unsigned count=shape[2],offset=shape[16],nt=count<32?count:(count+7)/8;
 if(!count||offset>total||count>total-offset||local>=nt)return;
 const unsigned warp=threadIdx.x/32,head=blockIdx.y*3+warp;
 if constexpr(Audit)if(threadIdx.x%32==0)atomicAdd(coverage+tile*9+head,1);
 __shared__ float scores[3][8][128];
 if constexpr(Staged){
  __shared__ __align__(16) __nv_bfloat16 shared_k[128*64],shared_v[128*64];
  staged_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local,head,scores[warp],shared_k,shared_v);
 }else{
  riley_compact_mixed::attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local,head,scores[warp]);
 }
}
}
