#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../../kernels/src/decode_gate_v56.cuh"
#include "../../kernels/optional/ffn_pipeline.cuh"
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T* allocate(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
  constexpr size_t X=32*576, G=1536*576, O=32*1536, P=5*32*576;
  auto* x=allocate<__nv_bfloat16>(X); auto* gate=allocate<__nv_bfloat16>(G);
  auto* up=allocate<__nv_bfloat16>(G); auto* down=allocate<__nv_bfloat16>(G);
  auto* a=allocate<__nv_bfloat16>(O);auto* b=allocate<__nv_bfloat16>(O);
  auto* pa=allocate<float>(P);auto* pb=allocate<float>(P);auto* active=allocate<unsigned>(1);
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  std::vector<__nv_bfloat16> host(G);std::vector<unsigned short> ha(O),hb(O);
  std::vector<float> hpa(P),hpb(P);uint64_t seed=9130501;
  auto fill=[&](auto* dst,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;
    host[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/32768.F);}
    CHECK(cudaMemcpy(dst,host.data(),n*2,cudaMemcpyHostToDevice));};
  fill(gate,G);fill(up,G);fill(down,G);
  cudaGraph_t graph;cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(x,gate,up,a,active);
  shared32_projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(a,down,pa,nullptr,active);
  riley_ffn_pipeline::gate_up<<<192,64,0,stream>>>(x,gate,up,b,active);
  riley_ffn_pipeline::down_parts<<<dim3(72,5),32,0,stream>>>(b,down,pb,active);
  CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  unsigned cases=0;
  for(int iteration=0;iteration<3;++iteration)for(unsigned rows:{0u,1u,15u,16u,17u,31u,32u,33u}){
    fill(x,X);CHECK(cudaMemcpy(active,&rows,4,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(a,0x55,O*2));CHECK(cudaMemset(b,0x55,O*2));
    CHECK(cudaMemset(pa,0x55,P*4));CHECK(cudaMemset(pb,0x55,P*4));
    CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hpa.data(),pa,P*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,P*4,cudaMemcpyDeviceToHost));
    size_t gm=0,dm=0;for(size_t i=0;i<O;++i)gm+=ha[i]!=hb[i];
    for(size_t i=0;i<P;++i)dm+=std::memcmp(&hpa[i],&hpb[i],4)!=0;
    std::printf("{\"iteration\":%d,\"rows\":%u,\"gate_bit_mismatches\":%zu,\"down_bit_mismatches\":%zu}\n",iteration,rows,gm,dm);std::fflush(stdout);
    if(gm||dm)return 1;++cases;
  }
  cudaFuncAttributes ga,da;CHECK(cudaFuncGetAttributes(&ga,riley_ffn_pipeline::gate_up));CHECK(cudaFuncGetAttributes(&da,riley_ffn_pipeline::down_parts));
  std::printf("{\"cases\":%u,\"gate_shared\":%zu,\"down_shared\":%zu,\"gate_registers\":%d,\"down_registers\":%d,\"gate_local\":%zu,\"down_local\":%zu}\n",cases,ga.sharedSizeBytes,da.sharedSizeBytes,ga.numRegs,da.numRegs,ga.localSizeBytes,da.localSizeBytes);
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
  for(void* p:{(void*)x,(void*)gate,(void*)up,(void*)down,(void*)a,(void*)b,(void*)pa,(void*)pb,(void*)active})CHECK(cudaFree(p));
  return 0;
}
