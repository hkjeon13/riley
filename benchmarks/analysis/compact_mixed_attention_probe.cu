#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include "../../kernels/optional/compact_mixed_attention.cuh"
#define CK(...) do{auto e=(__VA_ARGS__);if(e!=cudaSuccess){std::fprintf(stderr,"line%d %s %s\n",__LINE__,#__VA_ARGS__,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* allocate(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
struct Case {unsigned prefill,decode,start,context;};
int main(int argc,char**argv){
 bool small=argc>1&&std::string(argv[1])=="--sanitizer-small";
 bool check_only=small||(argc>1&&std::string(argv[1])=="--check-only");
 constexpr unsigned cap=1024,Q=cap*576,KV=256*16*192,META=32+32*416+1024+1024;
 auto*q=allocate<__nv_bfloat16>(Q),*k=allocate<__nv_bfloat16>(KV),*v=allocate<__nv_bfloat16>(KV);
 auto*a=allocate<__nv_bfloat16>(Q),*b=allocate<__nv_bfloat16>(Q);
 auto*m=allocate<unsigned>(META),*coverage=allocate<unsigned>(riley_pod_mixed::MAX_TASKS);
 auto*p=allocate<riley_pod_mixed::Plan>(1);
 cudaStream_t stream;CK(cudaStreamCreate(&stream));
 std::vector<__nv_bfloat16> host(KV);unsigned long long seed=9140401;
 auto fill=[&](auto*dst,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;host[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/32768.F);}CK(cudaMemcpy(dst,host.data(),n*2,cudaMemcpyHostToDevice));};
 fill(k,KV);fill(v,KV);
 auto launch=[&](unsigned mode,unsigned*seen){
  if(mode==0){riley_mixed_attention::mapped_attention<<<dim3(cap,9),32,0,stream>>>(q,k,v,a,cap,m);CK(cudaGetLastError());}
  else if(mode==1){if(seen){CK(cudaMemsetAsync(p,0,sizeof(*p),stream));riley_compact_mixed::mapped_attention<true><<<dim3(cap,9),32,0,stream>>>(q,k,v,b,cap,m,seen);}else riley_compact_mixed::mapped_attention<false><<<dim3(cap,9),32,0,stream>>>(q,k,v,b,cap,m);CK(cudaGetLastError());}
  else if(mode==2){if(seen)CK(riley_pod_mixed::enqueue<1,true>(stream,q,k,v,b,cap,m,p,seen));else CK(riley_pod_mixed::enqueue<1>(stream,q,k,v,b,cap,m,p));}
  else {if(seen)CK(riley_compact_mixed::enqueue<1,true>(stream,q,k,v,b,cap,m,p,seen));else CK(riley_compact_mixed::enqueue<1>(stream,q,k,v,b,cap,m,p));}
 };
 cudaGraph_t graphs[4];cudaGraphExec_t execs[4];
 for(unsigned mode=0;mode<4;++mode){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(mode,nullptr);CK(cudaStreamEndCapture(stream,&graphs[mode]));CK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));}
 std::vector<unsigned short> ha(Q),hb(Q);std::vector<unsigned> hm(META),hc(riley_pod_mixed::MAX_TASKS);
 auto*hp=new riley_pod_mixed::Plan;
 unsigned cases=0;
 auto cases_to_run=small?std::vector<Case>{{0,0,0,1},{17,4,0,129},{128,8,16,256}}:std::vector<Case>{{0,0,0,1},{0,1,0,1},{0,32,0,4096},{1,31,0,128},{15,16,0,129},{16,16,16,512},{17,16,0,513},{31,8,0,1024},{32,8,0,1024},{128,31,0,512},{398,31,0,1024},{512,31,3584,4096},{1024,0,3072,4096}};
 for(Case c:cases_to_run){
  hm.assign(META,0);unsigned active=0,total=0,tiles=0;
  auto owner=[&](unsigned count,unsigned start,bool decode){
   unsigned id=active++;auto*s=hm.data()+32+id*416;s[1]=start+count-1;s[2]=count;s[16]=total;s[18]=decode;
   for(unsigned pg=0;pg<256;++pg)s[32+pg]=(pg*37+id*13)%256;
   unsigned nt=count<32?count:(count+7)/8;
   for(unsigned i=0;i<nt;++i)hm[32+32*416+1024+tiles++]=(id<<16)|i;
   total+=count;
  };
  if(c.prefill)owner(c.prefill,c.start,false);
  for(unsigned d=0;d<c.decode;++d)owner(1,c.context-1,true);
  hm[5]=active;hm[9]=total;hm[24]=tiles;
  CK(cudaMemcpy(m,hm.data(),META*4,cudaMemcpyHostToDevice));
  __nv_bfloat16 saved;CK(cudaMemcpy(&saved,v,2,cudaMemcpyDeviceToHost));
  for(unsigned repeat=0;repeat<(check_only?3u:2u);++repeat){
   if(repeat==2){unsigned short bits=0x7fc1;CK(cudaMemcpy(v,&bits,2,cudaMemcpyHostToDevice));}
   fill(q,Q);CK(cudaMemset(a,0x55,Q*2));launch(0,nullptr);CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(ha.data(),a,Q*2,cudaMemcpyDeviceToHost));
   for(unsigned mode=1;mode<=3;++mode){
    CK(cudaMemset(b,0x55,Q*2));CK(cudaMemset(coverage,0,riley_pod_mixed::MAX_TASKS*4));launch(mode,coverage);CK(cudaStreamSynchronize(stream));
    CK(cudaMemcpy(hb.data(),b,Q*2,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hc.data(),coverage,hc.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(hp,p,sizeof(*hp),cudaMemcpyDeviceToHost));
    size_t mismatch=0,bad_coverage=0;for(unsigned i=0;i<Q;++i)mismatch+=ha[i]!=hb[i];for(unsigned i=0;i<hc.size();++i)bad_coverage+=hc[i]!=(i<tiles*9?1u:0u);
    CK(cudaMemset(b,0x55,Q*2));CK(cudaGraphLaunch(execs[mode],stream));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(hb.data(),b,Q*2,cudaMemcpyDeviceToHost));
    size_t release_mismatch=0;for(unsigned i=0;i<Q;++i)release_mismatch+=ha[i]!=hb[i];
    unsigned both=0;for(unsigned sm=0;sm<riley_pod_mixed::MAX_SM_IDS;++sm)both+=hp->sm_roles[sm][0]>0&&hp->sm_roles[sm][1]>0;
    std::printf("CHECK prefill=%u decode=%u start=%u context=%u repeat=%u mode=%u mismatches=%zu coverage_errors=%zu release_mismatches=%zu error=%u sms_saw_both_roles=%u\n",c.prefill,c.decode,c.start,c.context,repeat,mode,mismatch,bad_coverage,release_mismatch,hp->error,both);std::fflush(stdout);
    if(mismatch||release_mismatch||bad_coverage||hp->error)return 1;
   }
   if(repeat==2)CK(cudaMemcpy(v,&saved,2,cudaMemcpyHostToDevice));
   ++cases;
  }
  if(!check_only){
   cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
   for(unsigned order=0;order<2;++order)for(unsigned at=0;at<4;++at){unsigned mode=order?3-at:at;
    for(int i=0;i<10;++i)CK(cudaGraphLaunch(execs[mode],stream));CK(cudaEventRecord(start,stream));
    for(int i=0;i<50;++i)CK(cudaGraphLaunch(execs[mode],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));
    std::printf("TIME prefill=%u decode=%u start=%u context=%u order=%u mode=%u graph_us=%.6f\n",c.prefill,c.decode,c.start,c.context,order,mode,ms*1000/50);std::fflush(stdout);
   }CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
  }
 }
 cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,riley_compact_mixed::execute<1>));
 int original_blocks=0,candidate_blocks=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&original_blocks,riley_mixed_attention::mapped_attention,32,0));CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&candidate_blocks,riley_compact_mixed::execute<1>,128,0));
 std::printf("OCCUPANCY original_ctas_per_sm=%d candidate_ctas_per_sm=%d original_warps_per_sm=%d candidate_warps_per_sm=%d\n",original_blocks,candidate_blocks,original_blocks,candidate_blocks*4);
 cudaFuncAttributes direct;CK(cudaFuncGetAttributes(&direct,riley_compact_mixed::mapped_attention<false>));
 std::printf("DIRECT registers=%d shared_bytes=%zu local_bytes=%zu\n",direct.numRegs,direct.sharedSizeBytes,direct.localSizeBytes);
 std::printf("DONE cases=%u registers=%d shared_bytes=%zu local_bytes=%zu workspace_bytes=%zu\n",cases,at.numRegs,at.sharedSizeBytes,at.localSizeBytes,sizeof(*hp));
 delete hp;for(unsigned i=0;i<4;++i){CK(cudaGraphExecDestroy(execs[i]));CK(cudaGraphDestroy(graphs[i]));}CK(cudaStreamDestroy(stream));
 for(void*ptr:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)m,(void*)coverage,(void*)p})CK(cudaFree(ptr));
}
