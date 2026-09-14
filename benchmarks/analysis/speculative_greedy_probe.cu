#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../kernels/optional/speculative_select.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 using namespace riley_speculative_select;
 auto*l=alloc<__nv_bfloat16>(32*49152);auto*out=alloc<GreedyRecord>(32);auto*m=alloc<unsigned>(32+32*416);auto*status=alloc<unsigned>(1);
 cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));greedy<<<32,256,0,stream>>>(l,out,m,32,status);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 unsigned cases=0;
 for(unsigned active:{1u,7u,32u})for(unsigned mode=0;mode<5;++mode){
  std::vector<unsigned> meta(32+32*416);meta[5]=(active+7)/8;meta[9]=active;
  for(unsigned i=0,offset=0;i<meta[5];++i){auto*r=meta.data()+32+i*416;r[2]=std::min(8u,active-offset);r[16]=offset;offset+=r[2];}
  std::vector<__nv_bfloat16> logits(32*49152,__float2bfloat16_rn(-20.F));
  for(unsigned slot=0;slot<active;++slot){auto*r=logits.data()+slot*49152;r[17+slot]=__float2bfloat16_rn(-1.F);r[30000+slot]=__float2bfloat16_rn(-1.F);
   if(mode==1)r[49]=__ushort_as_bfloat16(0x7fc0);if(mode==2)r[49]=__ushort_as_bfloat16(0x7f80);if(mode==3)r[49]=__ushort_as_bfloat16(0xff80);
  }
  unsigned failed=mode==4?8:0;
  CK(cudaMemcpy(m,meta.data(),meta.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(status,&failed,4,cudaMemcpyHostToDevice));CK(cudaMemcpy(l,logits.data(),logits.size()*2,cudaMemcpyHostToDevice));
  for(unsigned replay=0;replay<2;++replay){CK(cudaMemset(out,0xa5,512));CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));
   std::vector<GreedyRecord> result(32);CK(cudaMemcpy(result.data(),out,512,cudaMemcpyDeviceToHost));
   for(unsigned slot=0;slot<32;++slot){auto r=result[slot];if(r.slot!=slot)return 3;
    if(slot>=active){if(r.valid||r.error||r.token)return 4;}
    else {if(r.valid!=1 || bool(r.error)!=(mode!=0))return 5;if(mode==0&&r.token!=17+slot)return 6;}
   }
  }++cases;
 }
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));
 for(void*p:{(void*)l,(void*)out,(void*)m,(void*)status})CK(cudaFree(p));
 std::printf("verification_greedy cases=%u replays_each=2 ties=true negative=true nonfinite_rejected=true padding_zero=true\n",cases);
}
