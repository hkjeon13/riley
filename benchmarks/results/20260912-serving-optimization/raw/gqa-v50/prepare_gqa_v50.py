from pathlib import Path
r=Path('/tmp/riley-opt-260912');d=r/'gqa-attention-v50';d.mkdir(exist_ok=True);src=r/'prefill-shapes-source-v11/kernels/src'
s=(src/'decode_shared32_attention.cuh').read_text().replace('namespace riley_shared32_attention','namespace riley_gqa50_attention')
a=s.index('// Independent output');s=s[:a]
s=s.replace('blockIdx.y/9','blockIdx.y/3').replace('head=blockIdx.y%9','kvhead=blockIdx.y%3').replace('int lane=threadIdx.x,g=lane/4,t=lane%4;','int lane=threadIdx.x,g=lane/4,t=lane%4,head=kvhead*3+g;')
s=s.replace('auto* qp=q+head*64+depth*16;query[depth]=','auto* qp=q+(g<3?head:0)*64+depth*16;query[depth]=')
s=s.replace('query[depth]=riley_prefill_shape::pair(qp[2*t],qp[2*t+1]);query_hi[depth]=riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]);','query[depth]=g<3?riley_prefill_shape::pair(qp[2*t],qp[2*t+1]):0;query_hi[depth]=g<3?riley_prefill_shape::pair(qp[2*t+8],qp[2*t+9]):0;')
s=s.replace('token+g,head/3','token+g,kvhead').replace('query[depth],query[depth],query_hi[depth],query_hi[depth]','query[depth],0,query_hi[depth],0').replace('if(g==0)','if(g<3)')
s+='''
// Three query heads form distinct MMA rows sharing one KV head.
__global__ void values(const float* scores,const __nv_bfloat16* v,__nv_bfloat16* out,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){
 int row=blockIdx.y/3;uint32_t active=*live_rows;if(active<1||active>32||row>=active)return;
 shape+=row*416;pages+=row*416;scores+=row*9*4096;out+=row*576;
 int count=shape[1]+1,kvhead=blockIdx.y%3,block=blockIdx.x;
 int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,t=lane%4,head=kvhead*3+warp;
 if(count<1||count>4096)return;
 __shared__ __nv_bfloat16 probs[3][128];
 __shared__ float exponentials[3][128],alphas[3],denoms[3];
 float maximum=-CUDART_INF_F,den=0.,accum[4]={};
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
 for(int tile=(count-1)/128;tile>=0;--tile){
  int begin=tile*128,end=min(begin+128,count);float mx=maximum;
  for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[head*4096+begin+i]);
  for(int offset=16;offset>0;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
  float alpha=exp2f((maximum-mx)*1.4426950408889634F);
  if(lane==0)alphas[warp]=alpha;
  for(int i=lane;i<128;i+=32){float value=i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0.;exponentials[warp][i]=value;probs[warp][i]=__float2bfloat16_rn(value);}
  __syncwarp();
  float local_den=den*alpha;
  for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)local_den+=exponentials[warp][i];}
  den=local_den;maximum=mx;
  __syncthreads();
  if(warp==0){
   for(int j=0;j<4;++j)accum[j]*=g<3?alphas[g]:0.F;
   #pragma unroll 1
   for(int token=begin;token<end;token+=64){
    uint32_t pa[4],paa[4],vb[4],vbb[4];
    #pragma unroll
    for(int part=0;part<4;++part){
     int at=token+part*16,pi=at-begin;bool live=at<end;
     pa[part]=live&&g<3?riley_prefill_shape::pair(probs[g][pi+2*t],probs[g][pi+2*t+1]):0;
     paa[part]=live&&g<3?riley_prefill_shape::pair(probs[g][pi+2*t+8],probs[g][pi+2*t+9]):0;
     int dim=block*8+g;int value_base=live?((pages[at/16]*3+kvhead)*16)*64+dim:0;
     auto val=[&](int pos){return pos<end&&live?v[value_base+(pos-at)*64]:zero;};
     vb[part]=riley_prefill_shape::pair(val(at+2*t),val(at+2*t+1));
     vbb[part]=riley_prefill_shape::pair(val(at+2*t+8),val(at+2*t+9));
    }
    #pragma unroll
    for(int part=0;part<4;++part)if(token+part*16<end)riley_prefill_shape::mma(accum,pa[part],0,paa[part],0,vb[part],vbb[part]);
   }
  }
  __syncthreads();
 }
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 if(lane==0)denoms[warp]=den;
 __syncthreads();
 if(warp==0&&g<3){float inverse=1.0F/denoms[g];for(int j=0;j<2;++j)out[(kvhead*3+g)*64+block*8+2*t+j]=__float2bfloat16_rn(accum[j]*inverse);}
}
inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 scores<<<dim3(min((context+7)/8,32u),96),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 values<<<dim3(8,96),96,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
'''
(d/'gqa_attention_v50.cuh').write_text(s)
p=(r/'prefill-shapes-source-v11/kernels/tests/shared16_primitive_probe.cu').read_text();start=p.index('void attention()');p=p[:p.index('template<int N')]+p[start:p.index('int main()')]
p=p.replace('#include "decode_shared16.cuh"','#include "gqa_attention_v50.cuh"').replace('#include "decode_shared16_attention.cuh"','#include "decode_shared32_attention.cuh"')
p=p.replace('16*576','32*576').replace('4096*16*192','4096*32*192').replace('16*9*4096','32*9*4096').replace('16*416','32*416').replace('row<16','row<32').replace('rows<=17','rows<=33').replace('rows<=16','rows<=32').replace('counts[row]','counts[row%16]')
p=p.replace('riley_shared16_attention::enqueue','riley_gqa50_attention::enqueue')
p=p.replace('attention active_rows=1..16 invalid_rows=0,17','attention active_rows=1..32 invalid_rows=0,33')
p+='int main(){attention();}\n';(d/'probe.cu').write_text(p)
print(d)
