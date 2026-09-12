from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');s=(r/'kernels/src/graph_numerics_precise.cu').read_text()
norm=s[s.index('__global__ void compiled_norm_rows'):s.index('__global__ void compiled_rope_rows')]
swig=s[s.index('__global__ void compiled_swiglu_rows'):s.index('__global__ void gemv_prefill_rows')]
newnorm=norm.replace('compiled_norm_rows','norm_rows').replace('int mode){','int mode,const uint32_t* shape,uint32_t capacity){').replace(' const uint32_t row=blockIdx.x;', ' const uint32_t row=blockIdx.x,live=shape[2];if(!live||live>capacity||row>=live)return;')
newswig=swig.replace('compiled_swiglu_rows','swiglu_rows').replace('__nv_bfloat16* out){','__nv_bfloat16* out,const uint32_t* shape,uint32_t capacity){').replace(' const uint32_t row=blockIdx.y;', ' const uint32_t row=blockIdx.y,live=shape[2];if(!live||live>capacity||row>=live)return;')
head='''#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
// Internal V3 pointwise operators; the retained owner validates parent extents.
namespace riley_prefill_pointwise {
'''
extra=r'''
__global__ void embedding_rows(const __nv_bfloat16* weights,const uint32_t* tokens,__nv_bfloat16* out,const uint32_t* shape,uint32_t capacity,uint32_t vocabulary,uint32_t* status){
 uint32_t row=blockIdx.x,live=shape[2];if(!live||live>capacity){if(threadIdx.x==0)atomicOr(status,2U);return;}if(row>=live)return;
 uint32_t token=tokens[row];if(token>=vocabulary){if(threadIdx.x==0)atomicOr(status,1U);return;}
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)out[row*576+i]=weights[token*576+i];
}
__global__ void select_hidden(const __nv_bfloat16* rows,__nv_bfloat16* selected,const uint32_t* shape,uint32_t capacity,const uint32_t* status,uint32_t* publish){
 uint32_t live=shape[2],row=shape[11];bool ready=*status==0&&live>0&&live<=capacity&&row<live;
 if(threadIdx.x==0)*publish=ready?1:0;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)selected[i]=ready?rows[row*576+i]:__float2bfloat16_rn(0.F);
}
} // namespace riley_prefill_pointwise
'''
(r/'kernels/src/prefill_shape_pointwise.cuh').write_text(head+newnorm+newswig+extra)
probe=r'''
#include "prefill_shape_pointwise.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* up(const std::vector<T>&a){T*p;ck(cudaMalloc(&p,a.size()*sizeof(T)));ck(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 constexpr int cap=1024,vocab=64;std::vector<unsigned short>w(vocab*576),n(576),gate(cap*1536),uv(gate.size()),init(cap*576,0x5555),si(cap*1536,0x5555),selected(640,0x5555);std::vector<float>fi(cap*576,0);
 unsigned rng=91;for(auto*a:{&w,&n,&gate,&uv})for(auto&v:*a){rng=rng*1664525+1013904223;v=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(rng%1025)-512)/512.f));}
 auto dw=up(w),dn=up(n),dg=up(gate),du=up(uv),de=up(init),d0=up(init),d1=up(init),d2=up(init),dr=up(init),ds=up(si),re=up(init),r0=up(init),r1=up(init),r2=up(init),rr=up(init),rs=up(si),sel=up(selected);auto df=up(fi),rf=up(fi);
 uint32_t *meta,*host,*status,*publish;ck(cudaMallocHost(&host,17536));ck(cudaMalloc(&meta,17536));ck(cudaMalloc(&status,4));ck(cudaMalloc(&publish,4));cudaStream_t stream;ck(cudaStreamCreate(&stream));
 auto shape=meta+32;auto tokens=meta+3360;
 cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaMemcpyAsync(meta,host,17536,cudaMemcpyHostToDevice,stream));ck(cudaMemsetAsync(status,0,4,stream));
 riley_prefill_pointwise::embedding_rows<<<cap,256,0,stream>>>((__nv_bfloat16*)dw,tokens,(__nv_bfloat16*)de,shape,cap,vocab,status);
 riley_prefill_pointwise::norm_rows<<<cap,256,0,stream>>>((__nv_bfloat16*)de,nullptr,(__nv_bfloat16*)dn,nullptr,(__nv_bfloat16*)d0,0,shape,cap);
 riley_prefill_pointwise::norm_rows<<<cap,256,0,stream>>>((__nv_bfloat16*)d0,de,(__nv_bfloat16*)dn,df,(__nv_bfloat16*)d1,1,shape,cap);
 riley_prefill_pointwise::norm_rows<<<cap,256,0,stream>>>((__nv_bfloat16*)d1,df,(__nv_bfloat16*)dn,dr,(__nv_bfloat16*)d2,2,shape,cap);
 riley_prefill_pointwise::swiglu_rows<<<dim3(6,cap),256,0,stream>>>((__nv_bfloat16*)dg,(__nv_bfloat16*)du,(__nv_bfloat16*)ds,shape,cap);
 riley_prefill_pointwise::select_hidden<<<1,256,0,stream>>>((__nv_bfloat16*)d2,(__nv_bfloat16*)sel,shape,cap,status,publish);
 ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 unsigned replays=0;uint64_t compared=0;
 for(int count:{1,17,73,128,398,1024,16})for(bool partial:{false,true}){
  std::memset(host,0,17536);host[34]=count;host[43]=partial?UINT32_MAX:count-1;
  auto embedding=init;for(int i=0;i<count;++i){host[3360+i]=(i*7+replays)%vocab;std::copy_n(w.data()+host[3360+i]*576,576,embedding.data()+i*576);}
  for(auto p:{de,d0,d1,d2,dr,r0,r1,r2,rr})ck(cudaMemsetAsync(p,0x55,init.size()*2,stream));for(auto p:{ds,rs})ck(cudaMemsetAsync(p,0x55,si.size()*2,stream));for(auto p:{df,rf})ck(cudaMemsetAsync(p,0,fi.size()*4,stream));ck(cudaMemcpyAsync(re,embedding.data(),embedding.size()*2,cudaMemcpyHostToDevice,stream));
  ck(cudaGraphLaunch(exec,stream));
  oracle::compiled_norm_rows<<<count,256,0,stream>>>((__nv_bfloat16*)re,nullptr,(__nv_bfloat16*)dn,nullptr,(__nv_bfloat16*)r0,0);
  oracle::compiled_norm_rows<<<count,256,0,stream>>>((__nv_bfloat16*)r0,re,(__nv_bfloat16*)dn,rf,(__nv_bfloat16*)r1,1);
  oracle::compiled_norm_rows<<<count,256,0,stream>>>((__nv_bfloat16*)r1,rf,(__nv_bfloat16*)dn,rr,(__nv_bfloat16*)r2,2);
  oracle::compiled_swiglu_rows<<<dim3(6,count),256,0,stream>>>((__nv_bfloat16*)dg,(__nv_bfloat16*)du,(__nv_bfloat16*)rs);
  ck(cudaStreamSynchronize(stream));
  for(int i=0;i<7;++i){void*a=i==0?de:i==1?d0:i==2?d1:i==3?d2:i==4?dr:i==5?ds:(void*)df;void*b=i==0?re:i==1?r0:i==2?r1:i==3?r2:i==4?rr:i==5?rs:(void*)rf;size_t bytes=i==6?fi.size()*4:(i==5?si.size():init.size())*2;std::vector<uint8_t>aa(bytes),bb(bytes);ck(cudaMemcpy(aa.data(),a,bytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bb.data(),b,bytes,cudaMemcpyDeviceToHost));if(aa!=bb){fprintf(stderr,"mismatch %d %d\n",count,i);exit(3);}compared+=bytes;}
  unsigned flag;ck(cudaMemcpy(&flag,publish,4,cudaMemcpyDeviceToHost));if(flag!=!partial)exit(4);std::vector<unsigned short>a(640),b(576);ck(cudaMemcpy(a.data(),sel,a.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(b.data(),r2+(count-1)*576,b.size()*2,cudaMemcpyDeviceToHost));for(int i=0;i<640;++i)if(a[i]!=(i>=576?0x5555:partial?0:b[i]))exit(5);++replays;
 }
 // Fresh replay with invalid token must suppress the published hidden state.
 host[3360]=vocab;ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));unsigned flag,error;ck(cudaMemcpy(&flag,publish,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));if(flag||!(error&1))exit(6);
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));for(void*p:{(void*)dw,(void*)dn,(void*)dg,(void*)du,(void*)de,(void*)d0,(void*)d1,(void*)d2,(void*)dr,(void*)ds,(void*)re,(void*)r0,(void*)r1,(void*)r2,(void*)rr,(void*)rs,(void*)sel,(void*)df,(void*)rf,(void*)meta,(void*)status,(void*)publish})ck(cudaFree(p));
 printf("pointwise_graph valid_replays=%u compared_bytes=%llu invalid_token_suppressed=true partial_hidden_zero=true\n",replays,(unsigned long long)compared);
}
'''
pre='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <vector>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <algorithm>\nnamespace oracle {\n'
(r/'kernels/tests/prefill_shape_pointwise_graph.cu').write_text(pre+norm+swig+'}\n'+probe)
