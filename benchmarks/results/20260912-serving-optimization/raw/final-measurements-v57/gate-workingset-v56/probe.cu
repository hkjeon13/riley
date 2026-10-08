#include "gate.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto ck_error=(x);if(ck_error!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(ck_error));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(){
 constexpr int layers=30,weight=1536*576,input=32*576,output=32*1536;
 auto*x=alloc<__nv_bfloat16>(layers*input);auto*g=alloc<__nv_bfloat16>(layers*weight);auto*u=alloc<__nv_bfloat16>(layers*weight);auto*a=alloc<__nv_bfloat16>(layers*output+16);auto*b=alloc<__nv_bfloat16>(layers*output+16);auto*active=alloc<uint32_t>(1);
 for(int i=0;i<layers*input;++i)x[i]=__float2bfloat16_rn(float((i*7+1)%127-63)/128);
 for(int i=0;i<layers*weight;++i){g[i]=__float2bfloat16_rn(float((i*13+1)%251-125)/256);u[i]=__float2bfloat16_rn(float((i*17+1)%127-63)/128);}
 cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));fprintf(stderr,"layers=%d weight_bytes=%zu l2_bytes=%d\n",layers,size_t(layers)*weight*4,prop.l2CacheSize);
 cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 auto launch=[&](int variant,__nv_bfloat16* out){for(int l=0;l<layers;++l){
  if(variant==0)shared32_gate_up_swiglu<true><<<192,64,0,stream>>>(x+l*input,g+l*weight,u+l*weight,out+l*output,active);
  else riley_gate_v56::split_rows<4><<<192,64,0,stream>>>(x+l*input,g+l*weight,u+l*weight,out+l*output,active);
 }};
 for(int rows:{1,4,8,16,24,32}){
  *active=rows;for(int i=0;i<layers*output+16;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
  launch(0,a);launch(1,b);CK(cudaDeviceSynchronize());if(memcmp(a,b,(layers*output+16)*2)){fprintf(stderr,"FAIL rows=%d\n",rows);return 3;}
  cudaGraphExec_t es[2];for(int v=0;v<2;++v){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(v,b);CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&es[v],gr,0,0,0));CK(cudaGraphDestroy(gr));}
  cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
  for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){int v=pair%2?1-order:order;for(int i=0;i<5;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaDeviceSynchronize());CK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));printf("{\"rows\":%d,\"variant\":%d,\"pair\":%d,\"us_per_layer\":%.6f}\n",rows,v,pair,ms*10/layers);}
  for(auto ge:es)CK(cudaGraphExecDestroy(ge));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
 }
 fprintf(stderr,"PASS six30layer_cases_full_output_padding_exact=true\n");
 for(void*p:{(void*)x,(void*)g,(void*)u,(void*)a,(void*)b,(void*)active})CK(cudaFree(p));CK(cudaStreamDestroy(stream));
}
