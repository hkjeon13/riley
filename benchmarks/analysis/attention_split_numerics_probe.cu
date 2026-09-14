// Compatibility probe, NOT a LeanAttention implementation or performance result.
// Compare legacy BF16 online recurrence with two independently normalized tiles.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../kernels/src/decode_gqa_attention_v50.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(2);}}while(0)
struct Partial {float maximum,denominator,values[64];};
__global__ void partials(const float* scores,const __nv_bfloat16* values,Partial* states,unsigned count){
 const int tile=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,g=lane/4,t=lane%4;
 const int begin=tile*128,end=min(begin+128,int(count));
 Partial& state=states[head*2+tile];
 if(begin>=end){if(lane==0){state.maximum=-CUDART_INF_F;state.denominator=0;}for(int i=lane;i<64;i+=32)state.values[i]=0;return;}
 __shared__ float exponentials[128];__shared__ __nv_bfloat16 probs[128];
 float mx=-CUDART_INF_F;
 for(int i=lane;i<end-begin;i+=32)mx=fmaxf(mx,scores[head*4096+begin+i]);
 for(int offset=16;offset;offset>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,offset));
 for(int i=lane;i<128;i+=32){float p=i<end-begin?riley_prefill_shape::exponential(scores[head*4096+begin+i],mx):0;exponentials[i]=p;probs[i]=__float2bfloat16_rn(p);}
 __syncwarp();float den=0;
 for(int j=0;j<16;++j)for(int z=0;z<2;++z){int i=2*t+j*8+z;if(i<end-begin)den+=exponentials[i];}
 den+=__shfl_xor_sync(0xffffffff,den,2);den+=__shfl_xor_sync(0xffffffff,den,1);
 if(lane==0){state.maximum=mx;state.denominator=den;}
 for(int block=0;block<8;++block){float accum[4]={};
  for(int token=begin;token<end;token+=16){
   int pi=token-begin,dim=block*8+g;
   auto val=[&](int pos){return pos<end?values[((pos/16*3+head/3)*16+pos%16)*64+dim]:__float2bfloat16_rn(0.F);};
   auto a=riley_prefill_shape::pair(probs[pi+2*t],probs[pi+2*t+1]);
   auto aa=riley_prefill_shape::pair(probs[pi+2*t+8],probs[pi+2*t+9]);
   riley_prefill_shape::mma(accum,a,a,aa,aa,riley_prefill_shape::pair(val(token+2*t),val(token+2*t+1)),riley_prefill_shape::pair(val(token+2*t+8),val(token+2*t+9)));
  }
  if(g==0)for(int j=0;j<2;++j)state.values[block*8+2*t+j]=accum[j];
 }
}
__global__ void merge(const Partial* states,__nv_bfloat16* out){
 int head=blockIdx.x;const auto& low=states[head*2];const auto& high=states[head*2+1];
 float mx=fmaxf(low.maximum,high.maximum);
 float a=low.denominator?exp2f((low.maximum-mx)*1.4426950408889634F):0;
 float b=high.denominator?exp2f((high.maximum-mx)*1.4426950408889634F):0;
 float den=high.denominator*b+low.denominator*a;
 for(int i=threadIdx.x;i<64;i+=32)out[head*64+i]=__float2bfloat16_rn((high.values[i]*b+low.values[i]*a)/den);
}
template<class T>T* alloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 const size_t Q=32*576,KV=256*16*192,S=32*9*4096;
 auto*q=alloc<__nv_bfloat16>(Q),*k=alloc<__nv_bfloat16>(KV),*v=alloc<__nv_bfloat16>(KV),*original=alloc<__nv_bfloat16>(Q),*candidate=alloc<__nv_bfloat16>(Q);
 auto*scores=alloc<float>(S);auto*states=alloc<Partial>(18);auto*shape=alloc<unsigned>(32*416),*pages=alloc<unsigned>(32*416),*active=alloc<unsigned>(1);
 std::vector<__nv_bfloat16> hq(Q),hk(KV),hv(KV);std::vector<unsigned> hs(32*416),hp(32*416);unsigned one=1;
 for(int h=0;h<9;++h)hq[h*64]=__float2bfloat16_rn(1.F);
 for(int r=0;r<32;++r)for(int p=0;p<256;++p)hp[r*416+p]=p;
 for(int p=0;p<128;++p)for(int h=0;h<3;++h)for(int d=0;d<64;++d)hv[((p/16*3+h)*16+p%16)*64+d]=__float2bfloat16_rn(1.F);
 CK(cudaMemcpy(q,hq.data(),Q*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(v,hv.data(),KV*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(pages,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(active,&one,4,cudaMemcpyHostToDevice));
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 for(unsigned count:{128u,256u})for(float key:{0.F,51.F/64.F}){
  for(int p=128;p<256;++p)for(int h=0;h<3;++h)hk[((p/16*3+h)*16+p%16)*64]=__float2bfloat16_rn(key);
  hs[1]=count-1;CK(cudaMemcpy(k,hk.data(),KV*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(shape,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemset(original,0x55,Q*2));CK(cudaMemset(candidate,0x55,Q*2));
  cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  riley_gqa50_attention::enqueue(stream,q,k,v,original,scores,shape,pages,active,256);
  partials<<<dim3(2,9),32,0,stream>>>(scores,v,states,count);merge<<<9,32,0,stream>>>(states,candidate);
  CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
  for(int replay=0;replay<2;++replay){CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));}
  std::vector<unsigned short>a(Q),b(Q);CK(cudaMemcpy(a.data(),original,Q*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),candidate,Q*2,cudaMemcpyDeviceToHost));
  size_t differences=0;for(int i=0;i<576;++i)differences+=a[i]!=b[i];
  for(size_t i=576;i<Q;++i)if(a[i]!=0x5555||b[i]!=0x5555)return 3;
  bool counterexample=count==256&&key!=0;
  std::printf("count=%u key=%.8f mismatches=%zu baseline=%.8f split=%.8f expected_compatible=%s\n",count,key,differences,__bfloat162float(__ushort_as_bfloat16(a[0])),__bfloat162float(__ushort_as_bfloat16(b[0])),counterexample?"false":"true");
  if(counterexample?differences!=576:differences!=0)return 4;
  CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
 }
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)original,(void*)candidate,(void*)scores,(void*)states,(void*)shape,(void*)pages,(void*)active})CK(cudaFree(p));
 CK(cudaStreamDestroy(stream));std::puts("legacy_profile_split_compatibility=REJECT counterexample_verified=true");
}
