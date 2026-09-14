#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../kernels/optional/speculative_select.cuh"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
 auto*in=alloc<__nv_bfloat16>(32*576),*out=alloc<__nv_bfloat16>(32*576);auto*m=alloc<unsigned>(32+32*416);auto*status=alloc<unsigned>(1);
 std::vector<__nv_bfloat16> hidden(32*576);for(unsigned row=0;row<32;++row)for(unsigned d=0;d<576;++d)hidden[row*576+d]=__float2bfloat16_rn(float(row+1));
 CK(cudaMemcpy(in,hidden.data(),hidden.size()*2,cudaMemcpyHostToDevice));
 cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));riley_speculative_select::gather<<<32,256,0,stream>>>(in,out,m,32,status);CK(cudaStreamEndCapture(stream,&graph));CK(cudaGraphInstantiate(&exec,graph,0));
 unsigned cases=0;
 for(auto counts: {std::vector<unsigned>{1},std::vector<unsigned>{8,8,8,8},std::vector<unsigned>{1,8,3,7},std::vector<unsigned>{9}})for(unsigned failed:{0u,1u}) {
  std::vector<unsigned> meta(32+32*416);meta[5]=counts.size();unsigned total=0;std::vector<unsigned> expected(32);unsigned extra=counts.size();
  for(unsigned owner=0;owner<counts.size();++owner){auto*row=meta.data()+32+owner*416;row[2]=counts[owner];row[16]=total;
   expected[owner]=total+counts[owner];for(unsigned local=0;local+1<counts[owner];++local)expected[extra+local]=total+local+1;extra+=counts[owner]-1;total+=counts[owner];}
  meta[9]=total;bool eligible=riley_speculative_select::eligible(meta.data(),32);
  CK(cudaMemcpy(m,meta.data(),meta.size()*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(status,&failed,4,cudaMemcpyHostToDevice));CK(cudaMemset(out,0x55,32*576*2));
  for(int replay=0;replay<2;++replay){CK(cudaGraphLaunch(exec,stream));CK(cudaStreamSynchronize(stream));}
  std::vector<__nv_bfloat16> result(32*576);CK(cudaMemcpy(result.data(),out,result.size()*2,cudaMemcpyDeviceToHost));
  for(unsigned row=0;row<32;++row)for(unsigned d=0;d<576;++d){auto actual=result[row*576+d];if(!eligible){if(__bfloat16_as_ushort(actual)!=0x5555)return 3;}else if(__bfloat162float(actual)!=float(failed?0:expected[row]))return 4;}
  ++cases;
 }
 CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));CK(cudaStreamDestroy(stream));
 for(void*p:{(void*)in,(void*)out,(void*)m,(void*)status})CK(cudaFree(p));
 std::printf("verification_select cases=%u replays_each=2 ragged=true padding_zero=true status_zeroing=true unsupported_unchanged=true\n",cases);
}
