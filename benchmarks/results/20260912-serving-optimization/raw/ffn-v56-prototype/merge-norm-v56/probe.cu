#include "merge.cuh"
#include "decode_shared32.cuh"
#include "prefill_shape_pointwise.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#define CK(x) do{auto ck_error=(x);if(ck_error!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(ck_error));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(int argc,char**argv){
 bool timing=argc>1;constexpr int size=32*576+16;
 auto*parts=alloc<float>(5*32*576);auto*bf=alloc<__nv_bfloat16>(size);auto*fl=alloc<float>(size);auto*w=alloc<__nv_bfloat16>(576);auto*merged=alloc<__nv_bfloat16>(size);
 auto*a=alloc<__nv_bfloat16>(size);auto*b=alloc<__nv_bfloat16>(size);auto*ra=alloc<float>(size);auto*rb=alloc<float>(size);auto*shape=alloc<uint32_t>(3);shape[0]=shape[1]=0;
 cudaStream_t stream;CK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 auto launch=[&](int mode,bool fused,__nv_bfloat16* out,float* res){
  const void* residual=mode==1?static_cast<const void*>(bf):static_cast<const void*>(fl);
  if(fused)riley_merge_norm_v56::merge_norm<<<32,256,0,stream>>>(parts,residual,w,res,out,mode,shape,32);
  else{shared32_projection_merge<576,1536,320><<<(32*576+255)/256,256,0,stream>>>(parts,merged,shape+2);riley_prefill_pointwise::norm_rows<<<32,256,0,stream>>>(merged,residual,w,res,out,mode,shape,32);}
 };
 int cases=0;
 for(int seed:{1,17,99}){if(timing&&seed!=1)continue;
  for(int i=0;i<5*32*576;++i)parts[i]=__bfloat162float(__float2bfloat16_rn(float((i*7+seed)%251-125)/256));
  for(int i=0;i<size;++i){bf[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/128);fl[i]=float((i*17+seed)%97-48)/64;}
  for(int i=0;i<576;++i)w[i]=__float2bfloat16_rn(float((i+seed)%127)/128);
  for(int mode:{1,2})for(int rows=0;rows<=33;++rows)for(int exc:{0,1,2,3}){
   if(timing&&(exc||!(rows==1||rows==4||rows==8||rows==16||rows==24||rows==32)))continue;
   shape[2]=rows;int at=seed%576;auto old=parts[at];if(exc)parts[at]=exc==1?std::numeric_limits<float>::quiet_NaN():(exc==2?std::numeric_limits<float>::infinity():-std::numeric_limits<float>::infinity());
   for(int i=0;i<size;++i){a[i]=b[i]=__ushort_as_bfloat16(0x4123);ra[i]=rb[i]=-123.5F;}
   launch(mode,false,a,ra);launch(mode,true,b,rb);CK(cudaDeviceSynchronize());
   if(memcmp(a,b,size*2)||memcmp(ra,rb,size*4)){fprintf(stderr,"FAIL seed=%d mode=%d rows=%d exc=%d\n",seed,mode,rows,exc);return 3;}++cases;
   if(timing){cudaGraphExec_t es[2];for(int variant=0;variant<2;++variant){cudaGraph_t gr;CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));launch(mode,variant,b,rb);CK(cudaStreamEndCapture(stream,&gr));CK(cudaGraphInstantiate(&es[variant],gr,0,0,0));CK(cudaGraphDestroy(gr));}
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){int v=pair%2?1-order:order;for(int i=0;i<10;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaDeviceSynchronize());CK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)CK(cudaGraphLaunch(es[v],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));printf("{\"mode\":%d,\"rows\":%d,\"variant\":%d,\"pair\":%d,\"us\":%.6f}\n",mode,rows,v,pair,ms*10);}
    for(auto ge:es)CK(cudaGraphExecDestroy(ge));CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
   }
   parts[at]=old;
  }
 }
 fprintf(stderr,"PASS cases=%d norm_and_residual_bytes_exact=true inactive_guards=true finite_nonfinite=true\n",cases);
 for(void*p:{(void*)parts,(void*)bf,(void*)fl,(void*)w,(void*)merged,(void*)a,(void*)b,(void*)ra,(void*)rb,(void*)shape})CK(cudaFree(p));CK(cudaStreamDestroy(stream));
}
