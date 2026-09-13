#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../kernels/src/decode_gqa_attention_v50.cuh"
#include "../../kernels/optional/persistent_attention.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s: %s\n",__LINE__,#call,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 constexpr size_t Q=32*576,KV=256*16*192,S=32*9*4096;
 auto*q=alloc<__nv_bfloat16>(Q),*k=alloc<__nv_bfloat16>(KV),*v=alloc<__nv_bfloat16>(KV);
 auto*oa=alloc<__nv_bfloat16>(Q),*ob=alloc<__nv_bfloat16>(Q);
 auto*sa=alloc<float>(S),*sb=alloc<float>(S);
 auto*shape=alloc<unsigned>(32*416),*pages=alloc<unsigned>(32*416),*active=alloc<unsigned>(1);
 std::vector<__nv_bfloat16> data(KV);unsigned long long seed=9130701;
 auto fill=[&](auto* dst,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;data[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/32768.F);}CHECK(cudaMemcpy(dst,data.data(),n*2,cudaMemcpyHostToDevice));};
 fill(k,KV);fill(v,KV);
 std::vector<unsigned> hs(32*416),hp(32*416);
 for(int row=0;row<32;++row)for(int page=0;page<256;++page)hp[row*416+page]=(page*37+row*13)%256;
 CHECK(cudaMemcpy(pages,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
 riley_persistent_attention::Plan plan;CHECK(riley_persistent_attention::prepare(&plan));
 riley_persistent_attention::Args args{q,k,v,ob,sb,shape,pages,active};
 cudaGraph_t graph;cudaGraphExec_t exec;
 CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 riley_gqa50_attention::enqueue(stream,q,k,v,oa,sa,shape,pages,active,4096);
 CHECK(riley_persistent_attention::enqueue(stream,plan,args));
 CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 std::vector<unsigned> as(S),bs(S);std::vector<unsigned short> ao(Q),bo(Q);
 const unsigned lengths[]={1,7,8,9,15,16,17,127,128,129,511,512,513,4095,4096};
 for(int iteration=0;iteration<3;++iteration)for(unsigned rows:{0u,1u,16u,32u,33u}){
  for(unsigned row=0;row<32;++row)hs[row*416+1]=(iteration==1?4096:lengths[(row+iteration)%15])-1;
  if(iteration==2)hs[1]=4096; // invalid context must preserve outputs as in V7
  fill(q,Q);CHECK(cudaMemcpy(shape,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(active,&rows,4,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(sa,0x55,S*4));CHECK(cudaMemset(sb,0x55,S*4));CHECK(cudaMemset(oa,0x55,Q*2));CHECK(cudaMemset(ob,0x55,Q*2));
  CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(as.data(),sa,S*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bs.data(),sb,S*4,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(ao.data(),oa,Q*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bo.data(),ob,Q*2,cudaMemcpyDeviceToHost));
  size_t sm=0,om=0;for(size_t i=0;i<S;++i)sm+=as[i]!=bs[i];for(size_t i=0;i<Q;++i)om+=ao[i]!=bo[i];
  std::printf("iteration=%d rows=%u score_mismatches=%zu output_mismatches=%zu\n",iteration,rows,sm,om);std::fflush(stdout);if(sm||om)return 1;
  for(size_t i=0;i<Q;++i)if((rows==0||rows>32||i/576>=rows||hs[(i/576)*416+1]>=4096)&&bo[i]!=0x5555)return 3;
 }
 auto invalid=plan;invalid.blocks=2147483647;if(riley_persistent_attention::enqueue(stream,invalid,args)!=cudaErrorCooperativeLaunchTooLarge)return 4;
 cudaFuncAttributes attrs;CHECK(cudaFuncGetAttributes(&attrs,riley_persistent_attention::execute));
 std::printf("cases=15 blocks=%d registers=%d shared=%zu local=%zu exact=true\n",plan.blocks,attrs.numRegs,attrs.sharedSizeBytes,attrs.localSizeBytes);
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)oa,(void*)ob,(void*)sa,(void*)sb,(void*)shape,(void*)pages,(void*)active})CHECK(cudaFree(p));
}
