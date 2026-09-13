#pragma once
#include "../src/decode_shape.cuh"
// Diagnostic score->packed probability state->value protocol; destructive scratch.
namespace riley_planned_softmax {
__global__ void normalize(float* scores,const uint32_t* shape,const uint32_t* live_rows){
 unsigned row=blockIdx.y,head=blockIdx.x,rows=*live_rows;
 if(!rows||rows>32||row>=rows)return;
 unsigned count=shape[row*416+1]+1;if(!count||count>4096)return;
 float* base=scores+(row*9+head)*4096;
 // The last tile is consumed first. Its upper half can then hold 32 rescale
 // factors plus the final inverse, disjoint from its packed 128 BF16 values.
 float* states=base+((count-1)/128)*128+64;
 __shared__ float exponentials[128];
 int lane=threadIdx.x,t=lane%4;float maximum=-CUDART_INF_F,den=0.F;
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,int(count));float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,base[begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  for(int i=lane;i<128;i+=32)exponentials[i]=i<end-begin?riley_prefill_shape::exponential(base[begin+i],mx):0.F;
  __syncwarp(); // All original scores consumed before packing overwrites them.
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[i];}
  den=local_den;maximum=mx;
  auto* packed=reinterpret_cast<__nv_bfloat16*>(base+begin);
  for(int i=lane;i<128;i+=32)packed[i]=__float2bfloat16_rn(exponentials[i]);
  if(lane==0)states[tile]=alpha;
  __syncwarp(); // Exponentials are shared with the next reverse tile.
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 if(lane==0)states[32]=1.F/den;
}
__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/3,head=(blockIdx.y%3)*3+threadIdx.x/32,block=blockIdx.x;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;
 scores+=row*9*4096;out+=row*576;
 int count=shape[1]+1,lane=threadIdx.x%32,g=lane/4,t=lane%4;
 if(count<1||count>4096)return;
 float accum[4]={};const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 const float* states=scores+head*4096+((count-1)/128)*128+64;
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);
  const auto* probs=reinterpret_cast<const __nv_bfloat16*>(scores+head*4096+begin);
  float alpha=states[tile];for(int j=0;j<4;++j)accum[j]*=alpha;
  #pragma unroll 1
  for(int token=begin;token<end;token+=64){
   uint32_t pa[4],paa[4],vb[4],vbb[4];
   #pragma unroll
   for(int part=0;part<4;++part){
    int at=token+part*16,pi=at-begin;bool live=at<end;
    pa[part]=live?riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]):0;
    paa[part]=live?riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]):0;
    int dim=block*8+g;int value_base=live?((pages[at/16]*3+head/3)*16)*64+dim:0;
    auto val=[&](int pos){return pos<end&&live?v[value_base+(pos-at)*64]:zero;};
    vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));
    vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));
   }
   #pragma unroll
   for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(accum,pa[part],pa[part],paa[part],paa[part],vb[part],vbb[part]);
  }
  __syncwarp();
 }
 float inverse=states[32];
 if(g==0)for(int j=0;j<2;++j)out[head*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);
}

}
