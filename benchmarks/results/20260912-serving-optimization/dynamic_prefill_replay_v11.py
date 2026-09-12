from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'kernels/src/prefill_shape_projection.cuh';s=p.read_text().replace('uint32_t rows){','uint32_t rows,const uint32_t* live_rows=nullptr){',1);s=s.replace(' static_assert(N%', ' if(live_rows){uint32_t live=*live_rows;if(!live||live>rows)return;rows=live;}\n static_assert(N%',1);p.write_text(s)
p=r/'kernels/src/prefill_shape_rope_kv.cuh';s=p.read_text().replace('const uint32_t* pages,uint32_t start,uint32_t rows){','const uint32_t* pages,uint32_t start,uint32_t rows,const uint32_t* shape=nullptr){',1).replace(' const uint32_t row=blockIdx.y;', ' if(shape){uint32_t live=shape[2];if(!live||live>rows)return;rows=live;start=shape[4];}\n const uint32_t row=blockIdx.y;',1);p.write_text(s)
p=r/'kernels/src/prefill_shape_attention.cuh';s=p.read_text().replace('const int* dynamic_n,const uint32_t* blocks){','const int* dynamic_n,const uint32_t* blocks,const uint32_t* live_rows=nullptr){',1).replace(' if(dynamic_n)n=*dynamic_n+1;',' if(live_rows){uint32_t live=*live_rows;if(!live||live>static_cast<uint32_t>(rows))return;rows=live;}\n if(blockIdx.x>=static_cast<uint32_t>(rows))return;\n if(dynamic_n)n=*dynamic_n+1;',1);p.write_text(s)
p=r/'kernels/tests/prefill_shape_graph_replay.cu';p.write_text(r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "prefill_shape_projection.cuh"
#include "prefill_shape_rope_kv.cuh"
#include "prefill_shape_attention.cuh"
#include "prefill_shape_packet.hpp"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* up(const std::vector<T>& a){T* p;ck(cudaMalloc(&p,a.size()*sizeof(T)));ck(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
int main(){
 constexpr int bucket=1024,context=4096,pool=context*192;std::vector<unsigned short>x(bucket*576),w(576*576),k(bucket*192),v(k.size()),zero(bucket*576),cache(pool);
 unsigned rng=131;for(auto* a:{&x,&w,&k,&v})for(auto& t:*a){rng=rng*1664525+1013904223;t=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(rng%257)-128)/1024.f));}
 std::vector<float>cos(context*32,1),sin(context*32,0);
 auto dx=up(x),dw=up(w),dk=up(k),dv=up(v),dq=up(zero),dqo=up(zero),dout=up(zero),dkeys=up(cache),dvalues=up(cache),rq=up(zero),rqo=up(zero),rout=up(zero),rkeys=up(cache),rvalues=up(cache);auto dc=up(cos),ds=up(sin);
 uint8_t *host,*meta;ck(cudaMallocHost(&host,17536));ck(cudaMalloc(&meta,17536));cudaStream_t stream;ck(cudaStreamCreate(&stream));
 auto u32=[&](int at,unsigned val){std::memcpy(host+at,&val,4);};auto u64=[&](int at,uint64_t val){std::memcpy(host+at,&val,8);};
 auto packet=[&](unsigned rows,unsigned start,unsigned replay){std::memset(host,0,17536);unsigned target=start+rows,live=(target+15)/16;
  u32(0,0x33444d52);u32(4,3);u32(8,17536);u32(12,1664);u32(20,1);u32(24,8);u32(28,256);u64(40,1);u64(48,replay);u64(56,replay);host[64]=1;
  u32(128,33);u32(132,target-1);u32(136,rows);u32(140,live);u32(144,start);u32(148,target);u32(156,32);u32(160,target);u32(164,context);u32(172,rows-1);u64(176,1);u64(184,replay);
  for(unsigned i=0;i<live;++i){u32(256+4*i,(i*5+17)%256);uint16_t valid=i+1<live?16:(target-1)%16+1;std::memcpy(host+1280+i*2,&valid,2);}
  for(unsigned i=0;i<rows;++i)u32(13440+i*4,33);
  if(!valid_prefill_shape_packet(host,17536,256,8))exit(3);
 };
 auto ops=[&](bool dynamic,int rows,int start){auto q=dynamic?dq:rq,qo=dynamic?dqo:rqo,out=dynamic?dout:rout,keys=dynamic?dkeys:rkeys,values=dynamic?dvalues:rvalues;auto shape=(const uint32_t*)(meta+128);auto pages=(const uint32_t*)(meta+256);
  gemm_prefill_shape_vector<576,576,192,2><<<dim3(36,(rows+15)/16),64,0,stream>>>((__nv_bfloat16*)dx,(__nv_bfloat16*)dw,(__nv_bfloat16*)q,rows,dynamic?shape+2:nullptr);
  prefill_shape_rope_kv<<<dim3(2,rows),256,0,stream>>>((__nv_bfloat16*)q,(__nv_bfloat16*)dk,(__nv_bfloat16*)dv,(__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,dc,ds,pages,start,rows,dynamic?shape:nullptr);
  riley_prefill_shape::attention_shape<<<dim3(rows,3),96,0,stream>>>((__nv_bfloat16*)qo,(__nv_bfloat16*)keys,(__nv_bfloat16*)values,(__nv_bfloat16*)out,rows,start+rows,dynamic?(const int*)(shape+1):nullptr,pages,dynamic?shape+2:nullptr);
 };
 packet(17,0,1);cudaGraph_t graph;cudaGraphExec_t exec;
 ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaMemcpyAsync(meta,host,17536,cudaMemcpyHostToDevice,stream));ops(true,bucket,0);ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 unsigned replay=0;uint64_t compared=0;
 for(int rows:{17,73,128,398,1024,1,129,16})for(int start:{0,128}){
  packet(rows,start,++replay);
  for(auto p:{dq,dqo,dout,rq,rqo,rout})ck(cudaMemsetAsync(p,0x55,zero.size()*2,stream));
  for(auto p:{dkeys,dvalues,rkeys,rvalues})ck(cudaMemsetAsync(p,0,cache.size()*2,stream));
  ck(cudaGraphLaunch(exec,stream));ops(false,rows,start);ck(cudaStreamSynchronize(stream));
  for(int i=0;i<5;++i){auto a=i==0?dq:i==1?dqo:i==2?dout:i==3?dkeys:dvalues;auto b=i==0?rq:i==1?rqo:i==2?rout:i==3?rkeys:rvalues;size_t count=i<3?zero.size():cache.size();std::vector<unsigned short>left(count),right(count);ck(cudaMemcpy(left.data(),a,count*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(right.data(),b,count*2,cudaMemcpyDeviceToHost));if(left!=right){fprintf(stderr,"replay mismatch %u rows%d start%d buffer%d\n",replay,rows,start,i);exit(4);}compared+=count;}
 }
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));ck(cudaFree(meta));for(void* p:{(void*)dx,(void*)dw,(void*)dk,(void*)dv,(void*)dq,(void*)dqo,(void*)dout,(void*)dkeys,(void*)dvalues,(void*)rq,(void*)rqo,(void*)rout,(void*)rkeys,(void*)rvalues,(void*)dc,(void*)ds})ck(cudaFree(p));
 printf("dynamic_prefill_graph replays=%u exact_elements=%llu single_capture=true\n",replay,(unsigned long long)compared);
}
''')
