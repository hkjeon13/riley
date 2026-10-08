#include "decode-baseline.cuh"
#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "decode_shared32_attention.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 bool timing=argc>1;cudaStream_t stream;OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 auto*q=alloc<__nv_bfloat16>(32*576);auto*k=alloc<__nv_bfloat16>(4096*32*192);auto*v=alloc<__nv_bfloat16>(4096*32*192);
 auto*a=alloc<__nv_bfloat16>(32*576+16);auto*b=alloc<__nv_bfloat16>(32*576+16);auto*sa=alloc<float>(32*9*4096+16);auto*sb=alloc<float>(32*9*4096+16);auto*shape=alloc<uint32_t>(32*416);auto*live=alloc<uint32_t>(1);
 memset(shape,0,32*416*4);int counts[16]={1,16,17,127,128,129,2049,4096,2,15,73,255,256,398,1024,4095};
 auto launch=[&](int variant){
  if(variant==0)riley_gqa_reference_v57::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);
  if(variant==1)riley_decode_v57_1::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);
  if(variant==2)riley_decode_v57_2::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);
  if(variant==3)riley_decode_v57_3::enqueue(stream,q,k,v,b,sb,shape,shape+32,live,4096);

 };
 int cases=0;
 for(int seed:{1,17,99}){
  if(timing&&seed!=1)continue;
  for(int i=0;i<32*576;++i)q[i]=__float2bfloat16_rn(float((i*7+seed)%127-63)/128);
  for(int i=0;i<4096*32*192;++i){k[i]=__float2bfloat16_rn(float((i*17+seed)%251-125)/256);v[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/64);}
  for(int context:{0,1,31,128,398,1024,4096})for(int rows:{0,1,4,8,16,24,32,33})for(int exceptional:{0,1,2,3}){
   if(timing&&(exceptional||context<128||rows<1||rows>32||rows==24))continue;
   *live=rows;
   for(int row=0;row<32;++row){shape[row*416+1]=(context?context:counts[row%16])-1;for(int page=0;page<256;++page)shape[row*416+32+page]=row*256+(page*17+seed)%256;}
   int pos=shape[1],at=((shape[32+pos/16]*3)*16+pos%16)*64+seed%64;auto oldk=k[at],oldq=q[seed%576];
   if(exceptional==1)k[at]=__ushort_as_bfloat16(0x7fc1);if(exceptional==2)k[at]=__ushort_as_bfloat16(0x7f80);if(exceptional==3)q[seed%576]=__ushort_as_bfloat16(0x7fc1);
   for(int i=0;i<32*576+16;++i)a[i]=__ushort_as_bfloat16(0x4123);
   for(int i=0;i<32*9*4096+16;++i)sa[i]=-123.5F;
   riley_shared32_attention::enqueue(stream,q,k,v,a,sa,shape,shape+32,live,4096);OK(cudaDeviceSynchronize());
   for(int variant=1;variant<=3;++variant){
    for(int i=0;i<32*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
    for(int i=0;i<32*9*4096+16;++i)sb[i]=-123.5F;
    launch(variant);OK(cudaDeviceSynchronize());
    if(memcmp(a,b,(32*576+16)*2)||memcmp(sa,sb,(32*9*4096+16)*4)){fprintf(stderr,"mismatch seed=%d context=%d rows=%d variant=%d\n",seed,context,rows,variant);return 3;}
    ++cases;
   }
   if(timing){
    cudaGraphExec_t execs[4];
    for(int variant=0;variant<4;++variant){cudaGraph_t graph;OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(variant);OK(cudaStreamEndCapture(stream,&graph));OK(cudaGraphInstantiate(&execs[variant],graph,0,0,0));OK(cudaGraphDestroy(graph));}
    cudaEvent_t start,end;OK(cudaEventCreate(&start));OK(cudaEventCreate(&end));
    for(int pair=0;pair<4;++pair)for(int order=0;order<4;++order){int variant=pair%2?3-order:order;for(int i=0;i<10;++i)OK(cudaGraphLaunch(execs[variant],stream));OK(cudaDeviceSynchronize());OK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)OK(cudaGraphLaunch(execs[variant],stream));OK(cudaEventRecord(end,stream));OK(cudaEventSynchronize(end));float ms;OK(cudaEventElapsedTime(&ms,start,end));printf("{\"rows\":%d,\"context\":%d,\"pair\":%d,\"variant\":%d,\"us\":%.6f}\n",rows,context,pair,variant,ms*10);}
    for(auto graph_exec:execs)OK(cudaGraphExecDestroy(graph_exec));OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));
   }
   k[at]=oldk;q[seed%576]=oldq;
  }
 }
 fprintf(stderr,"PASS cases=%d exact_scores_outputs_padding=true finite_nonfinite_Q_K=true seeds=%d\n",cases,timing?1:3);
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)sa,(void*)sb,(void*)shape,(void*)live})OK(cudaFree(p));
}
