#include "gate.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 bool timing=argc>1;auto*x=alloc<__nv_bfloat16>(32*576);auto*g=alloc<__nv_bfloat16>(1536*576);auto*u=alloc<__nv_bfloat16>(1536*576);
 auto*a=alloc<__nv_bfloat16>(32*1536+16);auto*b=alloc<__nv_bfloat16>(32*1536+16);auto*active=alloc<uint32_t>(1);
 cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 auto launch=[&](int variant,__nv_bfloat16* out){
  if(variant==0)shared32_gate_up_swiglu<true><<<192,64,0,stream>>>(x,g,u,out,active);
  if(variant==1)riley_gate_v56::gate_up<4,false><<<192,32,0,stream>>>(x,g,u,out,active);
  if(variant==2)riley_gate_v56::gate_up<2,false><<<192,32,0,stream>>>(x,g,u,out,active);
  if(variant==3)riley_gate_v56::gate_up<2,true><<<192,32,0,stream>>>(x,g,u,out,active);
  if(variant==4)riley_gate_v56::gate_up<1,true><<<192,32,0,stream>>>(x,g,u,out,active);
 };
 int cases=0;
 for(int seed:{1,17,99}){if(timing&&seed!=1)continue;
  for(int i=0;i<32*576;++i)x[i]=__float2bfloat16_rn(float((i*7+seed)%127-63)/128);
  for(int i=0;i<1536*576;++i){g[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);u[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);}
  for(int rows=0;rows<=33;++rows)for(int exc:{0,1,2,3}){
   if(timing&&(exc||!(rows==1||rows==4||rows==8||rows==16||rows==24||rows==32)))continue;
   *active=rows;int xi=seed%576,wi=seed% (1536*576);auto oldx=x[xi],oldg=g[wi];
   if(exc==1)x[xi]=__ushort_as_bfloat16(0x7fc1);if(exc==2)g[wi]=__ushort_as_bfloat16(0x7f80);if(exc==3)x[xi]=__ushort_as_bfloat16(0x7f80);
   for(int i=0;i<32*1536+16;++i)a[i]=__ushort_as_bfloat16(0x4123);
   launch(0,a);CK(cudaDeviceSynchronize());
   for(int variant=1;variant<5;++variant){
    for(int i=0;i<32*1536+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
    launch(variant,b);CK(cudaDeviceSynchronize());
    if(memcmp(a,b,(32*1536+16)*2)){fprintf(stderr,"FAIL seed=%d rows=%d exc=%d variant=%d\n",seed,rows,exc,variant);return 3;}++cases;
   }
   if(timing){cudaGraphExec_t es[5];for(int variant=0;variant<5;++variant){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(variant,b);CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&es[variant],gr,0,0,0));CK(cudaGraphDestroy(gr));}
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int pair=0;pair<4;++pair)for(int order=0;order<5;++order){int v=pair%2?4-order:order;for(int i=0;i<10;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaDeviceSynchronize());CK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));printf("{\"rows\":%d,\"variant\":%d,\"pair\":%d,\"us\":%.6f}\n",rows,v,pair,ms*10);}
    for(auto graph_exec:es)CK(cudaGraphExecDestroy(graph_exec));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
   }
   x[xi]=oldx;g[wi]=oldg;
  }
 }
 fprintf(stderr,"PASS cases=%d full_output_and_inactive_guards_exact=true finite_nonfinite=true\n",cases);
 for(void*p:{(void*)x,(void*)g,(void*)u,(void*)a,(void*)b,(void*)active})CK(cudaFree(p));CK(cudaStreamDestroy(stream));
}
