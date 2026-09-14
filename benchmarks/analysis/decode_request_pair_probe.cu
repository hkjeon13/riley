#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include "../../kernels/optional/decode_request_pair.cuh"
#define CK(...) do{auto e=(__VA_ARGS__);if(e!=cudaSuccess){std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 bool profile=argc>1&&std::string(argv[1])=="--profile";
 bool small=argc>1&&std::string(argv[1])=="--small";
 constexpr unsigned Q=32*576,KV=512*16*192,S=32*9*4096,M=32*416;
 auto*q=alloc<__nv_bfloat16>(Q),*k=alloc<__nv_bfloat16>(KV),*v=alloc<__nv_bfloat16>(KV),*a=alloc<__nv_bfloat16>(Q),*b=alloc<__nv_bfloat16>(Q);
 auto*score=alloc<float>(S),*alpha=alloc<float>(32*9*32),*inverse=alloc<float>(32*9*4);
 auto*pairs=alloc<riley_request_pair::Pair>(32);
 auto*prob=alloc<__nv_bfloat16>(S);auto*shape=alloc<unsigned>(M),*pages=alloc<unsigned>(M),*active=alloc<unsigned>(1);
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 std::vector<__nv_bfloat16> data(KV);unsigned long long seed=9142501;
 auto fill=[&](auto*dst,unsigned n,float scale){for(unsigned i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;data[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/32768.F*scale);}CK(cudaMemcpy(dst,data.data(),n*2,cudaMemcpyHostToDevice));};
 fill(v,KV,1);fill(k,KV,1);
 std::vector<unsigned> hs(M),hp(M);std::vector<unsigned short> ha(Q),hb(Q);
 unsigned checks=0;
 for(unsigned count:profile?std::vector<unsigned>{512}:small?std::vector<unsigned>{1,129}:std::vector<unsigned>{1,15,16,17,63,64,65,127,128,129,512,4095,4096})
 for(unsigned rows:profile?std::vector<unsigned>{32}:small?std::vector<unsigned>{1,8}:std::vector<unsigned>{1,3,8,16,32})
 for(unsigned shared=profile?1:0;shared<(profile?2:3);++shared){
  for(unsigned r=0;r<32;++r){hs[r*416+1]=count-1-((shared==2&&r%2)?min(count-1,17u):0);for(unsigned p=0;p<256;++p)hp[r*416+p]=(p*37+((shared&&p<((count-1)/128)*8)?0:r*13))%512;}
  CK(cudaMemcpy(shape,hs.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(pages,hp.data(),M*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(active,&rows,4,cudaMemcpyHostToDevice));
  std::vector<riley_request_pair::Pair> descriptors;for(unsigned r=0;r<rows;r+=2)descriptors.push_back({r,min(r+1,rows-1)});CK(cudaMemcpy(pairs,descriptors.data(),descriptors.size()*sizeof(descriptors[0]),cudaMemcpyHostToDevice));
  auto launch=[&](unsigned mode){if(mode==0)riley_gqa50_attention::enqueue(stream,q,k,v,a,score,shape,pages,active,count);else {riley_request_pair::scores<<<dim3(min((count+7)/8,32u),descriptors.size()*3),32,0,stream>>>(q,k,score,shape,pages,pairs,descriptors.size());riley_request_pair::values<<<dim3(8,descriptors.size()*3),96,0,stream>>>(score,v,b,shape,pages,pairs,descriptors.size());}CK(cudaGetLastError());};
  cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
  for(unsigned mode=0;mode<2;++mode){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(mode);CK(cudaStreamEndCapture(stream,&graphs[mode]));CK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));}
  for(float scale:{1.F,32.F}){
   fill(q,Q,scale);CK(cudaMemset(a,0x55,Q*2));CK(cudaMemset(b,0x55,Q*2));
   launch(0);launch(1);CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(ha.data(),a,Q*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hb.data(),b,Q*2,cudaMemcpyDeviceToHost));
   unsigned mismatch=0;for(unsigned i=0;i<Q;++i)mismatch+=ha[i]!=hb[i];
   CK(cudaMemset(b,0x55,Q*2));CK(cudaGraphLaunch(execs[1],stream));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(hb.data(),b,Q*2,cudaMemcpyDeviceToHost));
   unsigned replay=0;for(unsigned i=0;i<Q;++i)replay+=ha[i]!=hb[i];
   std::printf("CHECK context=%u rows=%u shared=%u scale=%.0f mismatch=%u replay=%u\n",count,rows,shared,scale,mismatch,replay);std::fflush(stdout);if(mismatch||replay)return 1;++checks;
  }
  if(!small&&(count==512||count==4096)){
   cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
   for(unsigned order=0;order<2;++order)for(unsigned at=0;at<2;++at){unsigned mode=order?1-at:at;for(unsigned i=0;i<10;++i)CK(cudaGraphLaunch(execs[mode],stream));CK(cudaEventRecord(begin,stream));for(unsigned i=0;i<100;++i)CK(cudaGraphLaunch(execs[mode],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));std::printf("TIME context=%u rows=%u shared=%u order=%u mode=%u us=%.6f\n",count,rows,shared,order,mode,ms*10);std::fflush(stdout);}
   CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
  }
  for(unsigned mode=0;mode<2;++mode){CK(cudaGraphExecDestroy(execs[mode]));CK(cudaGraphDestroy(graphs[mode]));}
 }
 auto attributes=[&](const char* name,auto kernel,unsigned threads){cudaFuncAttributes a;CK(cudaFuncGetAttributes(&a,kernel));int blocks;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,0));std::printf("RESOURCE kernel=%s registers=%d shared_bytes=%zu local_bytes=%zu resident_blocks=%d threads=%u\n",name,a.numRegs,a.sharedSizeBytes,a.localSizeBytes,blocks,threads);};
 attributes("original_scores",riley_gqa50_attention::scores,32);attributes("pair_scores",riley_request_pair::scores,32);attributes("original_values",riley_gqa50_attention::independent_values,96);attributes("pair_values",riley_request_pair::values,96);
 std::printf("DONE checks=%u\n",checks);CK(cudaStreamDestroy(stream));
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)score,(void*)alpha,(void*)inverse,(void*)prob,(void*)shape,(void*)pages,(void*)active,(void*)pairs})CK(cudaFree(p));
}
