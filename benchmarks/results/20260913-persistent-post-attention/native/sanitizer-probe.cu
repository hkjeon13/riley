#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "../../kernels/src/decode_gate_v56.cuh"
#include "../../kernels/optional/persistent_post_attention.cuh"
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e));std::exit(2);} } while(0)
template<class T> T* allocate(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(){
  constexpr size_t X=32*576, G=1536*576, O=32*1536, P=5*32*576;
  auto* x=allocate<__nv_bfloat16>(X); auto* gate=allocate<__nv_bfloat16>(G);
  auto* up=allocate<__nv_bfloat16>(G); auto* down=allocate<__nv_bfloat16>(G);
  auto* a=allocate<__nv_bfloat16>(O);auto* b=allocate<__nv_bfloat16>(O);
  auto* pa=allocate<float>(P);auto* pb=allocate<float>(P);auto* active=allocate<unsigned>(3);
  auto* norm=allocate<__nv_bfloat16>(576);auto* residual=allocate<float>(X);
  auto* ra=allocate<__nv_bfloat16>(X);auto* rb=allocate<__nv_bfloat16>(X);auto* oa=allocate<__nv_bfloat16>(X);auto* ob=allocate<__nv_bfloat16>(X);
  auto* attention=allocate<__nv_bfloat16>(X);auto* projection=allocate<__nv_bfloat16>(576*576);
  auto* hidden=allocate<__nv_bfloat16>(X);auto* mid_norm=allocate<__nv_bfloat16>(576);
  auto* xb=allocate<__nv_bfloat16>(X);auto* residual_b=allocate<float>(X);
  cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
  std::vector<__nv_bfloat16> host(G);std::vector<unsigned short> ha(O),hb(O);
  std::vector<float> hpa(P),hpb(P);uint64_t seed=9130501;
  auto fill=[&](auto* dst,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;
    host[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/32768.F);}
    CHECK(cudaMemcpy(dst,host.data(),n*2,cudaMemcpyHostToDevice));};
  fill(gate,G);fill(up,G);fill(down,G);fill(norm,576);fill(attention,X);fill(projection,576*576);fill(hidden,X);fill(mid_norm,576);
  std::vector<float> hr(X);for(size_t i=0;i<X;++i)hr[i]=float(int(i%31)-15)/64.F;CHECK(cudaMemcpy(residual,hr.data(),X*4,cudaMemcpyHostToDevice));
  riley_persistent_post_attention::Plan plan;CHECK(riley_persistent_post_attention::prepare(&plan));
  riley_persistent_post_attention::Args args{xb,gate,up,down,norm,attention,projection,hidden,mid_norm,residual_b,b,rb,ob,pb,active};
  cudaGraph_t graph;cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  shared32_projection_parts<576,576,128,false><<<dim3(72,5),32,0,stream>>>(attention,projection,pa,nullptr,active+2);
    riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(pa,hidden,mid_norm,residual,x,1,active,32);
    riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(x,gate,up,a,active+2);
  shared32_projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(a,down,pa,nullptr,active+2);
  riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(pa,residual,norm,ra,oa,2,active,32);
  CHECK(riley_persistent_post_attention::enqueue(stream,plan,args));
  CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  unsigned cases=0;
  for(int iteration=0;iteration<3;++iteration)for(unsigned rows:{0u,1u,15u,16u,17u,31u,32u,33u}){
    fill(attention,X);CHECK(cudaMemcpy(active+2,&rows,4,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(a,0x55,O*2));CHECK(cudaMemset(b,0x55,O*2));
    CHECK(cudaMemset(pa,0x55,P*4));CHECK(cudaMemset(pb,0x55,P*4));
    for(auto* p:{ra,rb,oa,ob,x,xb})CHECK(cudaMemset(p,0x55,X*2));
    CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hpa.data(),pa,P*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hpb.data(),pb,P*4,cudaMemcpyDeviceToHost));
    size_t gm=0,dm=0;for(size_t i=0;i<O;++i)gm+=ha[i]!=hb[i];
    for(size_t i=0;i<P;++i)dm+=std::memcmp(&hpa[i],&hpb[i],4)!=0;
    std::printf("{\"iteration\":%d,\"rows\":%u,\"gate_bit_mismatches\":%zu,\"down_bit_mismatches\":%zu}\n",iteration,rows,gm,dm);std::fflush(stdout);
    if(gm||dm)return 1;
    for(auto pair:{std::make_pair(ra,rb),std::make_pair(oa,ob),std::make_pair(x,xb)}){
     CHECK(cudaMemcpy(ha.data(),pair.first,X*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),pair.second,X*2,cudaMemcpyDeviceToHost));
     for(size_t i=0;i<X;++i){if(ha[i]!=hb[i]){std::printf("norm mismatch index=%zu\n",i);return 1;}
      if((rows==0||rows>32||i>=rows*576)&&hb[i]!=0x5555)return 4;}
    }
    ++cases;
  }
  auto too_large=plan;too_large.blocks=2147483647;
  if(riley_persistent_post_attention::enqueue(stream,too_large,args)!=cudaErrorCooperativeLaunchTooLarge)return 5;
  auto wrong_device=plan;wrong_device.device=-1;
  if(riley_persistent_post_attention::enqueue(stream,wrong_device,args)!=cudaErrorInvalidDevice)return 6;
  auto null_args=args;null_args.parts=nullptr;
  if(riley_persistent_post_attention::enqueue(stream,plan,null_args)!=cudaErrorInvalidValue)return 7;
  cudaFuncAttributes attributes;CHECK(cudaFuncGetAttributes(&attributes,riley_persistent_post_attention::execute));
  std::printf("cases=%u blocks=%d registers=%d shared=%zu local=%zu exact_all_outputs=true\n",cases,plan.blocks,attributes.numRegs,attributes.sharedSizeBytes,attributes.localSizeBytes);
  CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
  for(void* p:{(void*)x,(void*)gate,(void*)up,(void*)down,(void*)a,(void*)b,(void*)pa,(void*)pb,(void*)active,(void*)norm,(void*)residual,(void*)ra,(void*)rb,(void*)oa,(void*)ob,(void*)attention,(void*)projection,(void*)hidden,(void*)mid_norm,(void*)xb,(void*)residual_b})CHECK(cudaFree(p));
  return 0;
}
