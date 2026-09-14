#include <math_constants.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include "../../kernels/optional/context_split_mixed.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 auto*q=alloc<__nv_bfloat16>(8*576),*k=alloc<__nv_bfloat16>(32*16*192),*v=alloc<__nv_bfloat16>(32*16*192),*out=alloc<__nv_bfloat16>(8*576);
 auto*scores=alloc<float>(32*9*4096);auto*w=alloc<riley_split_fp32::MixedWorkspace>(1);auto*m=alloc<unsigned>(32+32*416+2048);
 CK(cudaMemset(q,0,8*576*2));CK(cudaMemset(k,0,32*16*192*2));
 std::vector<__nv_bfloat16> values(32*16*192,__float2bfloat16_rn(.5F));CK(cudaMemcpy(v,values.data(),values.size()*2,cudaMemcpyHostToDevice));
 std::vector<unsigned> meta(32+32*416+2048);meta[5]=3;meta[9]=8;
 for(int row=0;row<3;++row){auto*d=meta.data()+32+row*416;d[1]=row==1?16:31;d[2]=row==0?6:1;d[16]=row==0?0:row+5;d[18]=row==0?0:1;for(int p=0;p<256;++p)d[32+p]=(p*7)%32;}
 CK(cudaMemcpy(m,meta.data(),meta.size()*4,cudaMemcpyHostToDevice));
 cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));CK(riley_split_fp32::enqueue_mixed(stream,q,k,v,out,scores,m,w,8,128));CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 for(int replay=0;replay<3;++replay){
  // Replay with zero decode, then restore the original ragged batch.
  meta[32+416+18]=replay==1?0:1;meta[32+832+18]=replay==1?0:1;
  CK(cudaMemcpy(m,meta.data(),meta.size()*4,cudaMemcpyHostToDevice));CK(cudaMemset(out,0x55,8*576*2));CK(cudaMemset(w,0xff,sizeof(*w)));
  CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));std::vector<__nv_bfloat16> result(8*576);CK(cudaMemcpy(result.data(),out,result.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<8*576;++i){if(i<6*576||replay==1){if(__bfloat16_as_ushort(result[i])!=0x5555)return 3;}else if(__bfloat162float(result[i])!=.5F)return 4;}
 }
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)out,(void*)scores,(void*)w,(void*)m})CK(cudaFree(p));
 std::puts("mixed_decode_map replay=3 ragged_counts=17,32 empty_decode_replay=true prefill_sentinel=true exact_constant=true");
}
