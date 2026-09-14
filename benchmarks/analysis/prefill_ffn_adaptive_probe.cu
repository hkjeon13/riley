#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include "../../kernels/optional/prefill_ffn_pipeline.cuh"
#include "../../kernels/optional/prefill_ffn_adaptive.cuh"
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%d %s: %s\n",__LINE__,#call,cudaGetErrorString(e));std::exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char** argv){
 if(argc!=2)return 3;
 constexpr int L=30;
 constexpr unsigned capacity=1024;constexpr size_t X=capacity*576,G=1536*576,O=capacity*1536,D=capacity*576;
 auto*x=alloc<__nv_bfloat16>(X);auto*g=alloc<__nv_bfloat16>(G*L);auto*u=alloc<__nv_bfloat16>(G*L);auto*w=alloc<__nv_bfloat16>(G*L);
 auto*a=alloc<__nv_bfloat16>(O);auto*b=alloc<__nv_bfloat16>(O);auto*da=alloc<__nv_bfloat16>(D);auto*db=alloc<__nv_bfloat16>(D);auto*live=alloc<unsigned>(1);
 std::vector<__nv_bfloat16> host(G);unsigned long long seed=9130502;
 auto fill=[&](auto* p,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;host[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/131072.F);}CHECK(cudaMemcpy(p,host.data(),n*2,cudaMemcpyHostToDevice));};
 for(int l=0;l<L;++l)for(int kind=0;kind<3;++kind){
  std::string file=std::string(argv[1])+"/"+std::to_string(l)+"-"+(kind==0?"gate":kind==1?"up":"down")+".bin";
  FILE* f=std::fopen(file.c_str(),"rb");if(!f)return 4;
  if(std::fread(host.data(),2,G,f)!=G||std::fgetc(f)!=EOF)return 5;std::fclose(f);
  CHECK(cudaMemcpy((kind==0?g:kind==1?u:w)+l*G,host.data(),G*2,cudaMemcpyHostToDevice));
 }fill(x,X);
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
 for(int lane=0;lane<2;++lane){CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
  for(int l=0;l<L;++l)if(!lane){riley_prefill_ffn_pipeline::gate_up<<<dim3(48,capacity/16),128,0,stream>>>(x,g+l*G,u+l*G,a,capacity,live);
   riley_prefill_ffn_pipeline::down<<<dim3(36,capacity/16),64,0,stream>>>(a,w+l*G,da,capacity,live);
  }else{riley_prefill_ffn_adaptive::gate_up<<<dim3(48,capacity/16),128,0,stream>>>(x,g+l*G,u+l*G,b,capacity,live);
   riley_prefill_ffn_adaptive::down<<<dim3(36,capacity/16),64,0,stream>>>(b,w+l*G,db,capacity,live);}
  CHECK(cudaStreamEndCapture(stream,&graphs[lane]));CHECK(cudaGraphInstantiate(&execs[lane],graphs[lane],nullptr,nullptr,0));
 }
 std::vector<unsigned short> ha(O),hb(O),hda(D),hdb(D);
 std::vector<unsigned> cases={0,1,15,16,17,31,32,33,63,64,65,128,160,191,192,193,224,256,288,320,352,384,398,512,1024,1025};
 for(int repeat=0;repeat<2;++repeat)for(unsigned rows:cases){
  fill(x,X);CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
  CHECK(cudaMemset(a,0x55,O*2));CHECK(cudaMemset(b,0x55,O*2));CHECK(cudaMemset(da,0x55,D*2));CHECK(cudaMemset(db,0x55,D*2));
  for(int l=0;l<L;++l){
   riley_prefill_ffn_pipeline::gate_up<<<dim3(48,capacity/16),128,0,stream>>>(x,g+l*G,u+l*G,a,capacity,live);
   riley_prefill_ffn_pipeline::down<<<dim3(36,capacity/16),64,0,stream>>>(a,w+l*G,da,capacity,live);
   riley_prefill_ffn_adaptive::gate_up<<<dim3(48,capacity/16),128,0,stream>>>(x,g+l*G,u+l*G,b,capacity,live);
   riley_prefill_ffn_adaptive::down<<<dim3(36,capacity/16),64,0,stream>>>(b,w+l*G,db,capacity,live);
   CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hda.data(),da,D*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hdb.data(),db,D*2,cudaMemcpyDeviceToHost));
  size_t gm=0,dm=0,dirty=0;for(size_t i=0;i<O;++i){gm+=ha[i]!=hb[i];if(rows>capacity||i>=rows*1536)dirty+=(ha[i]!=0x5555||hb[i]!=0x5555);}
  for(size_t i=0;i<D;++i){dm+=hda[i]!=hdb[i];if(rows>capacity||i>=rows*576)dirty+=(hda[i]!=0x5555||hdb[i]!=0x5555);}
  std::printf("{\"layer\":%d,\"repeat\":%d,\"rows\":%u,\"gate_mismatches\":%zu,\"down_mismatches\":%zu,\"inactive_dirty\":%zu}\n",l,repeat,rows,gm,dm,dirty);std::fflush(stdout);if(gm||dm||dirty)return 1;
  }
  for(auto exec:execs)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaMemcpy(ha.data(),a,O*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,O*2,cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hda.data(),da,D*2,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hdb.data(),db,D*2,cudaMemcpyDeviceToHost));if(ha!=hb||hda!=hdb)return 6;
 }
 {cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
  for(unsigned rows:{8u,16u,32u,64u,128u,160u,191u,192u,193u,224u,256u,288u,320u,352u,384u,398u,512u,1024u}){CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
   for(int order=0;order<2;++order)for(int i=0;i<2;++i){int lane=order?1-i:i;
    for(int n=0;n<10;++n)CHECK(cudaGraphLaunch(execs[lane],stream));
    CHECK(cudaEventRecord(start,stream));for(int n=0;n<100;++n)CHECK(cudaGraphLaunch(execs[lane],stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
    std::printf("{\"timing_rows\":%u,\"order\":%d,\"candidate\":%d,\"pair_us\":%.6f}\n",rows,order,lane,ms*10.F/L);
   }
  }CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));
 }
 cudaFuncAttributes ga,down;CHECK(cudaFuncGetAttributes(&ga,riley_prefill_ffn_adaptive::gate_up));CHECK(cudaFuncGetAttributes(&down,riley_prefill_ffn_adaptive::down));
 std::printf("{\"gate_shared\":%zu,\"down_shared\":%zu,\"gate_registers\":%d,\"down_registers\":%d,\"gate_local\":%zu,\"down_local\":%zu}\n",ga.sharedSizeBytes,down.sharedSizeBytes,ga.numRegs,down.numRegs,ga.localSizeBytes,down.localSizeBytes);
 for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(execs[i]));CHECK(cudaGraphDestroy(graphs[i]));}CHECK(cudaStreamDestroy(stream));
 for(void*p:{(void*)x,(void*)g,(void*)u,(void*)w,(void*)a,(void*)b,(void*)da,(void*)db,(void*)live})CHECK(cudaFree(p));
}
