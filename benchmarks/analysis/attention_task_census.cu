#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../../kernels/src/decode_gqa_attention_v50.cuh"
#include "../../kernels/optional/persistent_attention.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s: %s\n",__LINE__,#call,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<bool Score,bool Value> __global__ void task_phase(riley_persistent_attention::Args a){
 const unsigned rows=*a.active;
 if(!rows||rows>32)return;
 __shared__ unsigned prefix[33];
 __shared__ __nv_bfloat16 probs[8][128];
 __shared__ float exps[8][128];
 if(threadIdx.x==0){
  prefix[0]=0;
  for(unsigned r=0;r<rows;++r){
   unsigned count=a.shape[r*416+1]+1;
   prefix[r+1]=prefix[r]+(count>=1&&count<=4096?3*((count+7)/8):0);
  }
 }
 __syncthreads();
 unsigned warp=threadIdx.x/32,worker=blockIdx.x*8+warp,workers=gridDim.x*8;
 if constexpr(Score){
 for(unsigned task=worker;task<prefix[rows];task+=workers){
  unsigned row=0;while(row+1<rows&&task>=prefix[row+1])++row;
  unsigned tiles=(a.shape[row*416+1]+8)/8,local=task-prefix[row];
  riley_attention_tasks::score_task(a.q,a.k,a.scores,a.shape,a.pages,a.active,
      row,local/tiles,(local%tiles)*8,tiles*8);
 }
 }
 if constexpr(Value){
 // Interleave rows across resident warp workers to distribute ragged contexts.
 for(unsigned task=worker;task<rows*72;task+=workers){
  riley_attention_tasks::value_task(a.scores,a.v,a.out,a.shape,a.pages,a.active,
      task%rows,(task/rows)%9,task/(rows*9),probs[warp],exps[warp]);
  __syncwarp();
 }
 }
}
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

 const char* names[]={"original_score","original_value","original_pair","task_score","task_value","task_pair","persistent"};
 cudaGraph_t graphs[7];cudaGraphExec_t execs[7];
 for(int mode=0;mode<7;++mode){
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  if(mode==0||mode==2)riley_gqa50_attention::scores<<<dim3(32,96),32,0,stream>>>(q,k,sa,shape,pages,active);
  if(mode==1||mode==2)riley_gqa50_attention::independent_values<<<dim3(8,96),96,0,stream>>>(sa,v,oa,shape,pages,active);
  if(mode==3||mode==5)task_phase<true,false><<<plan.blocks,256,0,stream>>>(args);
  if(mode==4||mode==5)task_phase<false,true><<<plan.blocks,256,0,stream>>>(args);
  if(mode==6)CHECK(riley_persistent_attention::enqueue(stream,plan,args));
  CHECK(cudaStreamEndCapture(stream,&graphs[mode]));CHECK(cudaGraphInstantiate(&execs[mode],graphs[mode],nullptr,nullptr,0));
 }
 cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
 for(unsigned scenario=0;scenario<2;++scenario)for(unsigned rows:{1u,16u,32u}){
  for(unsigned r=0;r<32;++r)hs[r*416+1]=(scenario?lengths[r%15]:(r%3==0?16:r%3==1?128:398))-1;
  CHECK(cudaMemcpy(shape,hs.data(),hs.size()*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(active,&rows,4,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(sa,0x55,S*4));CHECK(cudaMemset(sb,0x55,S*4));CHECK(cudaMemset(oa,0x55,Q*2));CHECK(cudaMemset(ob,0x55,Q*2));
  CHECK(cudaGraphLaunch(execs[2],stream));CHECK(cudaGraphLaunch(execs[6],stream));CHECK(cudaStreamSynchronize(stream));
  for(int pair=0;pair<2;++pair)for(int order=0;order<7;++order){int mode=pair?6-order:order;
   for(int i=0;i<30;++i)CHECK(cudaGraphLaunch(execs[mode],stream));
   CHECK(cudaEventRecord(start,stream));for(int i=0;i<300;++i)CHECK(cudaGraphLaunch(execs[mode],stream));
   CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
   std::printf("census scenario=%u rows=%u pair=%d mode=%s graph_us=%.6f\n",scenario,rows,pair,names[mode],ms*1000/300);
   if(mode==5){CHECK(cudaMemcpy(bs.data(),sb,S*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(as.data(),sa,S*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(bo.data(),ob,Q*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ao.data(),oa,Q*2,cudaMemcpyDeviceToHost));if(as!=bs||ao!=bo)return 9;}
  }
 }
 CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));
 for(int mode=0;mode<7;++mode){CHECK(cudaGraphExecDestroy(execs[mode]));CHECK(cudaGraphDestroy(graphs[mode]));}
 auto invalid=plan;invalid.blocks=2147483647;if(riley_persistent_attention::enqueue(stream,invalid,args)!=cudaErrorCooperativeLaunchTooLarge)return 4;
 cudaFuncAttributes attrs;CHECK(cudaFuncGetAttributes(&attrs,riley_persistent_attention::execute));
 std::printf("cases=15 blocks=%d registers=%d shared=%zu local=%zu exact=true\n",plan.blocks,attrs.numRegs,attrs.sharedSizeBytes,attrs.localSizeBytes);
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)oa,(void*)ob,(void*)sa,(void*)sb,(void*)shape,(void*)pages,(void*)active})CHECK(cudaFree(p));
}
