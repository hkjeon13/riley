// Independent native numerical/timing screen, not model or serving evidence.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "../../kernels/src/decode_gqa_attention_v50.cuh"
#include "../../kernels/optional/context_split_fp32.cuh"
#define CK(x) do { auto error=(x); if(error!=cudaSuccess) {std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(error));std::exit(2);} } while(0)
template<class T> T* alloc(size_t n) {T* p; CK(cudaMalloc(&p,n*sizeof(T))); return p;}
float score(unsigned row,unsigned head,unsigned token) {return float(int((token*17+head*11+row*7)%97)-48)/16.F;}
float value(unsigned page,unsigned head,unsigned pos,unsigned dim) {return float(int((page*7+head*13+pos*3+dim*11)%67)-33)/32.F;}
int main(int argc,char** argv) {
 bool bounded=argc>1 && std::strcmp(argv[1],"--bounded")==0;
 constexpr size_t score_size=32*9*4096, kv_size=32*256*3*16*64, output_size=32*576;
 auto* scores=alloc<float>(score_size); auto* values=alloc<__nv_bfloat16>(kv_size);
 auto* output=alloc<__nv_bfloat16>(output_size); auto* original=alloc<__nv_bfloat16>(output_size);
 auto* scratch=alloc<riley_split_fp32::Partial>(riley_split_fp32::partial_count);
 auto* shape=alloc<unsigned>(32*416); auto* pages=alloc<unsigned>(32*416); auto* live=alloc<unsigned>(1);
 std::vector<float> hs(score_size); std::vector<__nv_bfloat16> hv(kv_size);
 std::vector<unsigned> hp(32*416),sh(32*416); std::vector<__nv_bfloat16> result(output_size),legacy(output_size);
 for(unsigned r=0;r<32;++r) {
  for(unsigned p=0;p<256;++p) hp[r*416+p]=r*256+(p*73)%256; // Noncontiguous, bijective pages.
  for(unsigned h=0;h<9;++h) for(unsigned t=0;t<4096;++t) hs[(r*9+h)*4096+t]=score(r,h,t);
 }
 for(unsigned p=0;p<32*256;++p) for(unsigned h=0;h<3;++h) for(unsigned t=0;t<16;++t) for(unsigned d=0;d<64;++d)
  hv[((p*3+h)*16+t)*64+d]=__float2bfloat16_rn(value(p,h,t,d));
 CK(cudaMemcpy(scores,hs.data(),hs.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(values,hv.data(),hv.size()*2,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(pages,hp.data(),hp.size()*4,cudaMemcpyHostToDevice));
 cudaStream_t stream; CK(cudaStreamCreate(&stream));
 auto launch=[&](unsigned splits,unsigned capacity){return riley_split_fp32::enqueue(stream,scores,values,scratch,output,shape,pages,live,splits,capacity);};
 if(launch(0,128)!=cudaErrorInvalidValue || launch(33,128)!=cudaErrorInvalidValue || launch(1,0)!=cudaErrorInvalidValue || launch(1,4097)!=cudaErrorInvalidValue) return 3;
 unsigned case_count=0; double overall_error=0; size_t differences=0;
 std::vector<unsigned> capacities=bounded?std::vector<unsigned>{1,17,129,257}:std::vector<unsigned>{1,17,129,257,1024,4096};
 std::vector<unsigned> actives=bounded?std::vector<unsigned>{0,1,3,33}:std::vector<unsigned>{0,1,3,32,33};
 for(unsigned capacity:capacities) for(unsigned active:actives) for(unsigned splits:{1u,2u,4u,32u}) {
  for(unsigned r=0;r<32;++r) sh[r*416+1]=std::max(1,int(capacity)-int(r*13))-1;
  CK(cudaMemcpy(shape,sh.data(),sh.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(live,&active,4,cudaMemcpyHostToDevice));
  CK(cudaMemset(output,0x55,output_size*2)); CK(cudaMemset(original,0x55,output_size*2));
  // Poison all partial storage; empty partitions must explicitly initialize it.
  CK(cudaMemset(scratch,0xff,riley_split_fp32::partial_count*sizeof(riley_split_fp32::Partial)));
  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); CK(launch(splits,capacity));
  CK(cudaStreamEndCapture(stream,&graph)); CK(cudaGraphInstantiate(&exec,graph,0));
  for(int replay=0;replay<2;++replay) {CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));}
  riley_gqa50_attention::independent_values<<<dim3(8,96),96,0,stream>>>(scores,values,original,shape,pages,live); CK(cudaGetLastError()); CK(cudaStreamSynchronize(stream));
  CK(cudaMemcpy(result.data(),output,output_size*2,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(legacy.data(),original,output_size*2,cudaMemcpyDeviceToHost));
  double error=0;
  for(unsigned r=0;r<32;++r) {
   for(unsigned d=0;d<576;++d) {
    unsigned index=r*576+d;
    if(active>32 || r>=active) {if(__bfloat16_as_ushort(result[index])!=0x5555) return 4;}
    else {if(!std::isfinite(__bfloat162float(result[index]))) return 5; differences+=__bfloat16_as_ushort(result[index])!=__bfloat16_as_ushort(legacy[index]);}
   }
   if(active>32 || r>=active) continue;
   unsigned count=sh[r*416+1]+1;
   for(unsigned head:{0u,4u,8u}) for(unsigned dim:{0u,31u,63u}) {
    double maximum=-INFINITY,den=0,numerator=0;
    for(unsigned t=0;t<count;++t) maximum=std::max(maximum,double(score(r,head,t)));
    for(unsigned t=0;t<count;++t) {double p=std::exp(double(score(r,head,t))-maximum);den+=p;numerator+=p*value(hp[r*416+t/16],head/3,t%16,dim);}
    double reference=numerator/den,observed=__bfloat162float(result[r*576+head*64+dim]);
    error=std::max(error,std::abs(reference-observed));
    // One-token attention must select V exactly, independent of split count.
    if(count==1 && observed!=reference) return 6;
   }
  }
  overall_error=std::max(overall_error,error);++case_count;
  std::printf("numerics capacity=%u active=%u splits=%u sampled_fp64_max_abs=%.12g\n",capacity,active,splits,error);
  CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph));
 }
 // Invalid device extent must not overwrite outputs.
 unsigned active=1; sh[1]=4096; CK(cudaMemcpy(shape,sh.data(),sh.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(live,&active,4,cudaMemcpyHostToDevice));
 CK(cudaMemset(output,0x55,output_size*2));CK(launch(2,4096));CK(cudaStreamSynchronize(stream));CK(cudaMemcpy(result.data(),output,output_size*2,cudaMemcpyDeviceToHost));
 for(auto x:result) if(__bfloat16_as_ushort(x)!=0x5555) return 7;
 if(!bounded) {
  for(unsigned capacity:{128u,512u,4096u}) for(unsigned rows:{1u,8u,32u}) {
   for(unsigned r=0;r<32;++r) sh[r*416+1]=capacity-1;
   CK(cudaMemcpy(shape,sh.data(),sh.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
   for(unsigned splits:{0u,1u,4u,16u,32u}) {
    cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    if(splits) CK(launch(splits,capacity)); else riley_gqa50_attention::independent_values<<<dim3(8,96),96,0,stream>>>(scores,values,original,shape,pages,live);
    CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
    for(int i=0;i<5;++i) CK(cudaGraphLaunch(exec,stream));
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));CK(cudaEventRecord(start,stream));
    for(int i=0;i<40;++i) CK(cudaGraphLaunch(exec,stream));
    CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));
    std::printf("timing capacity=%u active=%u splits=%u microseconds=%.6f qk_included=false merge_included=true\n",capacity,rows,splits,ms*1000/40);
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
   }
  }
 }
 for(void* p:{(void*)scores,(void*)values,(void*)output,(void*)original,(void*)scratch,(void*)shape,(void*)pages,(void*)live}) CK(cudaFree(p));
 CK(cudaStreamDestroy(stream));
 std::printf("complete cases=%u sampled_fp64_max_abs=%.12g legacy_mismatches=%zu partial_scratch_bytes=%zu model_quality_accepted=false serving_measured=false\n",case_count,overall_error,differences,riley_split_fp32::partial_count*sizeof(riley_split_fp32::Partial));
}
