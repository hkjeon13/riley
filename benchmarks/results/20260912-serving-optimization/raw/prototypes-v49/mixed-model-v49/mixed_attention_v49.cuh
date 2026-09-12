#pragma once
#include "prefill_shape_attention.cuh"
// Experimental only: exact ordered arithmetic against the independent row oracle.
namespace riley_mixed_attention {
using riley_prefill_shape::pair;
using riley_prefill_shape::mma;
using riley_prefill_shape::cache_index;
using riley_prefill_shape::exponential;
// Exceptional nonfinite V reuses the original per-query arithmetic and causal mask.
__device__ void single_query(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const uint32_t* blocks,int row,int qh,float (*scores)[128],__nv_bfloat16 (*probs)[128],float (*exponentials)[128]){
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
   exponentials[warp][i]=p;probs[warp][i]=__float2bfloat16_rn(p);
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
  __syncwarp();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);
 den+=__shfl_xor_sync(0xffffffff,den,1);
 float inverse=1.0F/den;
 if(group==0)for(int block=0;block<8;++block)for(int j=0;j<2;++j)
  out[qb+block*8+2*t+j]=__float2bfloat16_rn(accum[block][j]*inverse);
}

template<int TileRows>
__device__ __forceinline__ void attention_body(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const int* dynamic_n,const uint32_t* blocks,const uint32_t* live_rows,uint32_t query_block){
 static_assert(TileRows==8||TileRows==16,"query tile");
 if(live_rows){uint32_t live=*live_rows;if(!live||live>static_cast<uint32_t>(rows))return;rows=live;}
 if(dynamic_n)n=*dynamic_n+1;
 __shared__ float scores[16][128];
 __shared__ float exps[16][128];
 __shared__ __nv_bfloat16 probs[16][128];
 // A small prefill has insufficient queries to amortize the larger tile state.
 if(rows<32){
  if(query_block<static_cast<uint32_t>(rows))single_query(q,k,v,out,rows,n,blocks,query_block,blockIdx.y,scores,probs,exps);
  return;
 }
 const int first=query_block*TileRows;
 if(first>=rows)return;
 const int lane=threadIdx.x,group=lane/4,t=lane%4,qh=blockIdx.y,kvh=qh/3;
 const int qr[2]={first+group,first+group+8};
 const bool valid[2]={qr[0]<rows,TileRows==16&&qr[1]<rows};
 const int count[2]={n-rows+qr[0]+1,n-rows+qr[1]+1};
 if(n<rows||n>4096){
  for(int h=0;h<2;++h)if(valid[h])for(int d=t;d<64;d+=4)out[(qr[h]*9+qh)*64+d]=__float2bfloat16_rn(CUDART_NAN_F);
  return;
 }

 uint32_t query[2][4],query_hi[2][4];
 #pragma unroll
 for(int h=0;h<2;++h){
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
   for(int h=0;h<2;++h)for(int z=0;z<2;++z)if(token+2*t+z<end)scores[group+h*8][token-begin+2*t+z]=d[h*2+z]*.125F;
  }
  __syncwarp();
  float alpha[2];
  #pragma unroll
  for(int h=0;h<2;++h){
   const int own_end=valid[h]?min(end,count[h]):begin;
   const bool active=own_end>begin;
   float mx=maximum[h];
   for(int i=t;i<own_end-begin;i+=4)mx=fmaxf(mx,scores[group+h*8][i]);
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2,4));
   mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1,4));
   alpha[h]=active?exp2f((maximum[h]-mx)*1.4426950408889634F):1.F;
   for(int i=t;i<128;i+=4){
    float p=i<own_end-begin?exponential(scores[group+h*8][i],mx):0.F;
    exps[group+h*8][i]=p;probs[group+h*8][i]=__float2bfloat16_rn(p);
   }
   maximum[h]=mx;
  }
  __syncwarp();
  #pragma unroll
  for(int h=0;h<2;++h){
   const int own_end=valid[h]?min(end,count[h]):begin;
   float local=den[h]*alpha[h];
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<own_end-begin)local+=exps[group+h*8][i];}
   den[h]=local;
  }
  #pragma unroll
  for(int b=0;b<8;++b)for(int j=0;j<4;++j)accum[b][j]*=alpha[j/2];
  for(int token=begin;token<end;token+=16){
   const int pi=token-begin;
   uint32_t a[2],aa[2];
   #pragma unroll
   for(int h=0;h<2;++h){a[h]=pair(probs[group+h*8][pi+2*t],probs[group+h*8][pi+2*t+1]);aa[h]=pair(probs[group+h*8][pi+2*t+8],probs[group+h*8][pi+2*t+9]);}
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
     single_query(q,k,v,out,rows,n,blocks,row,qh,scores,probs,exps);
     __syncwarp();
    }
    return;
   }
  }
  __syncwarp();
 }
 #pragma unroll
 for(int h=0;h<2;++h){
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],2,4);
  den[h]+=__shfl_xor_sync(0xffffffff,den[h],1,4);
  const float inverse=1.F/den[h];
  if(valid[h])for(int b=0;b<8;++b)for(int z=0;z<2;++z)out[(qr[h]*9+qh)*64+b*8+2*t+z]=__float2bfloat16_rn(accum[b][h*2+z]*inverse);
 }
}
// Each CTA belongs to exactly one owner's query/head tile. Keys and causal
// positions stay owner-local even though projection rows are densely packed.
__global__ void attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta){
 uint32_t owner=blockIdx.z,active=meta[5],total=meta[9];if(!active||active>32||owner>=active||!total||total>capacity)return;
 const uint32_t* shape=meta+32+owner*416;uint32_t offset=shape[16],count=shape[2];
 if(!count||offset>total||count>total-offset)return;
 attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,blockIdx.x);
}

// Compact tile offsets are prototype metadata at row word17 and header word24.
// V7 authority validation will own these fields before production integration.
__global__ void compact_attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta){
 const uint32_t active=meta[5],total=meta[9],tiles=meta[24];
 if(!active||active>32||!total||total>capacity||!tiles||tiles>total)return;
 for(uint32_t tile=blockIdx.x;tile<tiles;tile+=gridDim.x){
  const uint32_t* shape=nullptr;uint32_t local=0;
  for(uint32_t owner=0;owner<active;++owner){
   const uint32_t* row=meta+32+owner*416;const uint32_t count=row[2],offset=row[16],start=row[17],nt=count<32?count:(count+7)/8;
   if(!count||offset>total||count>total-offset||start>tiles||nt>tiles-start)return;
   if(tile>=start&&tile-start<nt){shape=row;local=tile-start;break;}
  }
  if(!shape)return;
  attention_body<8>(q+shape[16]*576,k,v,out+shape[16]*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2,local);
  __syncwarp();
 }
}

}
