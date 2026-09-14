#include "../../kernels/optional/decode_projection_split_rows.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}} while(0)
template<class T>T* allocate(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char**) {
 constexpr size_t Q=576*576,K=192*576,W=Q+2*K+Q+576*1536;
 constexpr size_t P=3*32*960,S=5*32*576,Total=P+2*S+32;
 auto* weights=allocate<__nv_bfloat16>(30*W);auto* x=allocate<__nv_bfloat16>(32*1536);
 auto* a=allocate<float>(Total);auto* b=allocate<float>(Total);auto* live=allocate<unsigned>(1);
 std::vector<__nv_bfloat16> h(30*W);unsigned long long seed=9140601;
 auto fill=[&](auto* device,size_t n){for(size_t i=0;i<n;++i){seed=seed*6364136223846793005ULL+1;h[i]=__float2bfloat16_rn(float(int(seed>>48)-32768)/131072.F);}CHECK(cudaMemcpy(device,h.data(),n*2,cudaMemcpyHostToDevice));};
 fill(weights,30*W);fill(x,32*1536);
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));
 auto launch=[&](bool candidate,int layers){auto* p=candidate?b:a;
  for(int layer=0;layer<layers;++layer){auto* w=weights+layer*W;
   if(candidate){
    riley_projection_split::qkv_parts<<<dim3(120,3),64,0,stream>>>(x,w,w+Q,w+Q+K,p,live);
    riley_projection_split::projection_parts<576,576,128,false><<<dim3(72,5),64,0,stream>>>(x,w+Q+2*K,p+P,live);
    riley_projection_split::projection_parts<576,1536,320,true><<<dim3(72,5),64,0,stream>>>(x,w+2*Q+2*K,p+P+S,live);
   }else{
    riley_adaptive_decode::qkv_parts<<<dim3(120,3),32,0,stream>>>(x,w,w+Q,w+Q+K,p,live);
    riley_adaptive_decode::projection_parts<576,576,128,false><<<dim3(72,5),32,0,stream>>>(x,w+Q+2*K,p+P,live);
    riley_adaptive_decode::projection_parts<576,1536,320,true><<<dim3(72,5),32,0,stream>>>(x,w+2*Q+2*K,p+P+S,live);
   }
  }
 };
 cudaGraph_t graphs[2];cudaGraphExec_t execs[2];
 for(int lane=0;lane<2;++lane){CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(lane,1);CHECK(cudaStreamEndCapture(stream,&graphs[lane]));CHECK(cudaGraphInstantiate(&execs[lane],graphs[lane],nullptr,nullptr,0));}
 std::vector<unsigned> ha(Total),hb(Total);
 int comparisons=0;
 for(int repeat=0;repeat<(argc>1?1:3);++repeat){fill(x,32*1536);
  for(unsigned rows=0;rows<=33;++rows){CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));CHECK(cudaMemset(a,0x55,Total*4));CHECK(cudaMemset(b,0x55,Total*4));
   for(auto exec:execs)CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));CHECK(cudaMemcpy(ha.data(),a,Total*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(hb.data(),b,Total*4,cudaMemcpyDeviceToHost));
   if(ha!=hb){fprintf(stderr,"part bit mismatch rows=%u repeat=%d\n",rows,repeat);return 3;}
   auto guard=[&](size_t offset,int chunks,int n){for(int chunk=0;chunk<chunks;++chunk)for(unsigned row=(rows<=32?rows:0);row<32;++row)for(int col=0;col<n;++col)if(hb[offset+(chunk*32+row)*n+col]!=0x55555555u){fprintf(stderr,"inactive write rows=%u\n",rows);exit(4);}};
   guard(0,3,576);guard(3*32*576,3,192);guard(3*32*(576+192),3,192);guard(P,5,576);guard(P+S,5,576);
   for(size_t i=P+2*S;i<Total;++i)if(hb[i]!=0x55555555u)return 5;
   ++comparisons;
  }
 }
 printf("{\"graph_comparisons\":%d,\"all_partial_bits_exact\":true,\"inactive_guards\":true}\n",comparisons);
 for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(execs[i]));CHECK(cudaGraphDestroy(graphs[i]));}
 if(argc==1){
  for(int lane=0;lane<2;++lane){CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(lane,30);CHECK(cudaStreamEndCapture(stream,&graphs[lane]));CHECK(cudaGraphInstantiate(&execs[lane],graphs[lane],nullptr,nullptr,0));}
  cudaEvent_t start,end;CHECK(cudaEventCreate(&start));CHECK(cudaEventCreate(&end));
  for(unsigned rows:{1u,8u,16u,17u,24u,32u}){CHECK(cudaMemcpy(live,&rows,4,cudaMemcpyHostToDevice));
   for(int order=0;order<4;++order)for(int j=0;j<2;++j){int lane=order%2?1-j:j;
    for(int n=0;n<20;++n)CHECK(cudaGraphLaunch(execs[lane],stream));CHECK(cudaEventRecord(start,stream));
    for(int n=0;n<100;++n)CHECK(cudaGraphLaunch(execs[lane],stream));CHECK(cudaEventRecord(end,stream));CHECK(cudaEventSynchronize(end));float ms;CHECK(cudaEventElapsedTime(&ms,start,end));
    printf("{\"rows\":%u,\"order\":%d,\"candidate\":%d,\"three_ops_30_weights_us\":%.6f}\n",rows,order,lane,ms*10.F);
   }
  }
  CHECK(cudaEventDestroy(start));CHECK(cudaEventDestroy(end));for(int i=0;i<2;++i){CHECK(cudaGraphExecDestroy(execs[i]));CHECK(cudaGraphDestroy(graphs[i]));}
 }
 cudaFuncAttributes attrs;CHECK(cudaFuncGetAttributes(&attrs,riley_projection_split::qkv_parts));printf("{\"qkv_regs\":%d,\"qkv_local_bytes\":%zu,\"weight_working_set_bytes\":%zu,\"full_model\":false}\n",attrs.numRegs,attrs.localSizeBytes,30*W*2);
 CHECK(cudaStreamDestroy(stream));for(void* p:{(void*)weights,(void*)x,(void*)a,(void*)b,(void*)live})CHECK(cudaFree(p));
}
