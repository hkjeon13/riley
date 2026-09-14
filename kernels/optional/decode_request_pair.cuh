#pragma once
#include "decode_gqa_attention_v50.cuh"

// Experimental paired execution. Descriptors must cover every active row once,
// with a==b representing a singleton. No production dispatch uses this yet.
namespace riley_request_pair {
struct Pair {uint32_t a,b;};

__global__ void scores(const __nv_bfloat16* q,const __nv_bfloat16* k,float* result,const uint32_t* shape,const uint32_t* pages,const Pair* pairs,uint32_t pair_count){
 int id=blockIdx.y/3;if(id>=pair_count)return;
 Pair pair=pairs[id];int rows[2]={int(pair.a),int(pair.b)};
 int counts[2]={int(shape[rows[0]*416+1])+1,int(shape[rows[1]*416+1])+1};
 int lane=threadIdx.x,g=lane/4,t=lane%4,kvhead=blockIdx.y%3;
 for(int token=blockIdx.x*8;token<max(counts[0],counts[1]);token+=gridDim.x*8){
  bool shared=rows[0]!=rows[1]&&token+8<=min(counts[0],counts[1])&&pages[rows[0]*416+token/16]==pages[rows[1]*416+token/16];
  for(int pass=0;pass<(shared?1:(rows[0]==rows[1]?1:2));++pass){
   int owner=shared?g/3:pass;bool live_query=shared?g<6:g<3;
   int row=rows[live_query?owner:0],head=kvhead*3+(shared?g%3:g);
   int keyrow=rows[shared?0:pass],count=counts[shared?0:pass];
   if(!shared&&token>=count)continue;
   float d[4]={};uint32_t qa[4],qb[4],ka[4],kb[4];
   bool valid=token+g<count;
   int keybase=valid?riley_prefill_shape::cache_index(token+g,kvhead,0,pages+keyrow*416):0;
   #pragma unroll
   for(int depth=0;depth<4;++depth){
    const auto* qp=q+row*576+(live_query?head:0)*64+depth*16;
    qa[depth]=live_query?riley_prefill_shape::pair(qp[2*t],qp[2*t+1]):0;
    qb[depth]=live_query?riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]):0;
    ka[depth]=valid?riley_prefill_shape::pair(k[keybase+depth*16+2*t],k[keybase+depth*16+2*t+1]):0;
    kb[depth]=valid?riley_prefill_shape::pair(k[keybase+depth*16+2*t+8],k[keybase+depth*16+2*t+9]):0;
   }
   #pragma unroll
   for(int depth=0;depth<4;++depth)riley_prefill_shape::mma(d,qa[depth],0,qb[depth],0,ka[depth],kb[depth]);
   if(live_query)for(int j=0;j<2;++j)if(token+2*t+j<counts[owner])result[(row*9+head)*4096+token+2*t+j]=d[j]*.125F;
  }
 }
}

__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const Pair* pairs,uint32_t pair_count){
 int id=blockIdx.y/3;if(id>=pair_count)return;
 Pair pair=pairs[id];int rows[2]={int(pair.a),int(pair.b)},owners=pair.a==pair.b?1:2;
 int counts[2]={int(shape[rows[0]*416+1])+1,int(shape[rows[1]*416+1])+1};
 int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,t=lane%4,head=(blockIdx.y%3)*3+warp,block=blockIdx.x;
 __shared__ __nv_bfloat16 all_probs[3][2][128];
 __shared__ float all_exp[3][128];auto* ex=all_exp[warp];
 float maximum[2]={-CUDART_INF_F,-CUDART_INF_F},den[2]={},accum[2][4]={};
 const auto zero=__float2bfloat16_rn(0.);
 for(int tile=(max(counts[0],counts[1])-1)/128;tile>=0;--tile){
  int begin=tile*128;bool shared=owners==2&&begin+128<=min(counts[0],counts[1]);
  if(shared)for(int p=0;p<8;++p)shared=shared&&pages[rows[0]*416+tile*8+p]==pages[rows[1]*416+tile*8+p];
  for(int r=0;r<owners;++r){
   if(begin>=counts[r])continue;
   int end=min(begin+128,counts[r]);float mx=maximum[r];auto* probs=all_probs[warp][r];const float* s=scores+(rows[r]*9+head)*4096;
   for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,s[begin+i]);
   for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
   float alpha=exp2f((maximum[r]-mx)*1.4426950408889634F);
   for(int i=lane;i<128;i+=32){float x=i<end-begin?riley_prefill_shape::exponential(s[begin+i],mx):0.;ex[i]=x;probs[i]=__float2bfloat16_rn(x);}
   __syncwarp();float sum=den[r]*alpha;
   for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)sum+=ex[i];}
   den[r]=sum;maximum[r]=mx;for(int j=0;j<4;++j)accum[r][j]*=alpha;__syncwarp();
  }
  for(int pass=0;pass<(shared?1:owners);++pass){
   int end=min(begin+128,counts[pass]);if(begin>=end)continue;
   float d[4]={};
   if(shared){int r=g==1?1:0;d[0]=accum[r][0];d[1]=accum[r][1];}
   else for(int j=0;j<4;++j)d[j]=accum[pass][j];
   #pragma unroll 1
   for(int token=begin;token<end;token+=64){
    uint32_t pa[4],paa[4],vb[4],vbb[4];
    #pragma unroll
    for(int part=0;part<4;++part){
     int at=token+part*16,pi=at-begin;bool live=at<end;int r=shared?(g==1?1:0):pass;auto* probs=all_probs[warp][r];
     pa[part]=live?riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]):0;
     paa[part]=live?riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]):0;
     int base=live?((pages[rows[pass]*416+at/16]*3+head/3)*16)*64+block*8+g:0;
     auto val=[&](int pos){return live&&pos<end?v[base+(pos-at)*64]:zero;};
     vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));
    }
    #pragma unroll
    for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(d,pa[part],pa[part],paa[part],paa[part],vb[part],vbb[part]);
   }
   if(shared)for(int r=0;r<2;++r){accum[r][0]=__shfl_sync(0xffffffff,d[0],r*4+t);accum[r][1]=__shfl_sync(0xffffffff,d[1],r*4+t);}
   else for(int j=0;j<4;++j)accum[pass][j]=d[j];
  }
  __syncwarp();
 }
 for(int r=0;r<owners;++r){den[r]+=__shfl_xor_sync(0xffffffff,den[r],2);den[r]+=__shfl_xor_sync(0xffffffff,den[r],1);float inv=1.0F/den[r];if(g==0)for(int j=0;j<2;++j)out[rows[r]*576+head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[r][j]*inv);}
}
}
