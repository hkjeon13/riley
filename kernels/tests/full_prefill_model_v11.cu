#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "prefill_shape_model.cuh"
#include "prefill_shape_packet.hpp"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T read(std::ifstream& f){T x{};f.read((char*)&x,sizeof(x));if(!f)exit(8);return x;}
int main(int argc,char**argv){if(argc!=2)return 1;std::string root=argv[1];std::ifstream wf(root+"/weights.bin",std::ios::binary);std::vector<const void*>weights(273);std::vector<void*>owned;
 for(int i=0;i<273;++i){auto bytes=read<uint64_t>(wf);if(!bytes){weights[i]=weights[0];continue;}std::vector<uint8_t>v(bytes);wf.read((char*)v.data(),bytes);if(!wf)exit(8);void*p;ck(cudaMalloc(&p,bytes));ck(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));weights[i]=p;owned.push_back(p);}
 constexpr unsigned cap=512,physical=64,context=1024;size_t cachebytes=30ull*physical*16*192*2;std::vector<void*>scratch(12);unsigned widths[]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};for(int i=0;i<12;++i){ck(cudaMalloc(&scratch[i],cap*widths[i]));owned.push_back(scratch[i]);}
 void *keys,*values,*cos,*sin,*selected,*metadata;uint32_t *status,*publish;uint8_t* host;
 auto alloc=[&](void**p,size_t n){ck(cudaMalloc(p,n));owned.push_back(*p);};alloc(&keys,cachebytes);alloc(&values,cachebytes);alloc(&cos,context*32*4);alloc(&sin,context*32*4);alloc(&selected,1152);alloc(&metadata,17536);ck(cudaMalloc(&status,4));ck(cudaMalloc(&publish,4));owned.push_back(status);owned.push_back(publish);ck(cudaMallocHost(&host,17536));
 std::ifstream rf(root+"/rope.bin",std::ios::binary);std::vector<float>table(context*32);for(void*p:{cos,sin}){rf.read((char*)table.data(),table.size()*4);if(!rf)exit(8);ck(cudaMemcpy(p,table.data(),table.size()*4,cudaMemcpyHostToDevice));}
 cudaStream_t stream;ck(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaMemcpyAsync(metadata,host,17536,cudaMemcpyHostToDevice,stream));ck(enqueue_v3_prefill_model(stream,scratch.data(),weights.data(),metadata,keys,values,cos,sin,selected,status,publish,cap,physical));ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 auto u32=[&](int at,uint32_t value){std::memcpy(host+at,&value,4);};auto u64=[&](int at,uint64_t value){std::memcpy(host+at,&value,8);};unsigned replays=0;
 auto run=[&](const std::vector<uint32_t>&tokens,unsigned start,unsigned count){std::memset(host,0,17536);unsigned target=start+count,live=(target+15)/16;
  u32(0,0x33444d52);u32(4,3);u32(8,17536);u32(12,1664);u32(20,1);u32(24,8);u32(28,physical);u64(40,1);u64(48,++replays);u64(56,replays);host[64]=1;
  u32(128,tokens[start]);u32(132,target-1);u32(136,count);u32(140,live);u32(144,start);u32(148,target);u32(156,32);u32(160,tokens.size());u32(164,context);u32(172,target==tokens.size()?count-1:UINT32_MAX);u64(176,1);u64(184,replays);
  for(unsigned i=0;i<live;++i){u32(256+i*4,(i*5+17)%physical);uint16_t valid=i+1<live?16:(target-1)%16+1;std::memcpy(host+1280+i*2,&valid,2);}for(unsigned i=0;i<count;++i)u32(13440+i*4,tokens[start+i]);if(!valid_prefill_shape_packet(host,17536,physical,8))exit(3);
  ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));uint32_t error,flag;ck(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(&flag,publish,4,cudaMemcpyDeviceToHost));if(error||flag!=(target==tokens.size()))exit(4);if(!flag){std::vector<uint16_t>x(576);ck(cudaMemcpy(x.data(),selected,1152,cudaMemcpyDeviceToHost));for(auto v:x)if(v)exit(9);}
 };
 std::ifstream input(root+"/requests.bin",std::ios::binary);unsigned cases=read<uint32_t>(input);uint64_t compared=0;
 for(unsigned index=0;index<cases;++index){unsigned count=read<uint32_t>(input);std::vector<uint32_t>tokens(count);input.read((char*)tokens.data(),count*4);if(!input)exit(8);
  ck(cudaMemset(keys,0,cachebytes));ck(cudaMemset(values,0,cachebytes));run(tokens,0,count);std::vector<uint8_t>a(1152),ka(cachebytes),va(cachebytes),b(1152),kb(cachebytes),vb(cachebytes);ck(cudaMemcpy(a.data(),selected,a.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(ka.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(va.data(),values,cachebytes,cudaMemcpyDeviceToHost));
  ck(cudaMemset(keys,0,cachebytes));ck(cudaMemset(values,0,cachebytes));for(unsigned start=0;start<count;start+=73)run(tokens,start,std::min(73u,count-start));ck(cudaMemcpy(b.data(),selected,b.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(kb.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(vb.data(),values,cachebytes,cudaMemcpyDeviceToHost));if(a!=b||ka!=kb||va!=vb){fprintf(stderr,"checkpoint mismatch prompt%u hidden%d keys%d values%d\n",count,a!=b,ka!=kb,va!=vb);exit(5);}compared+=a.size()+ka.size()+va.size();std::ofstream dump(root+"/hidden-"+std::to_string(count)+".bf16",std::ios::binary);dump.write((char*)a.data(),a.size());printf("full_model_prefill prompt=%u hidden_exact=true full_kv_exact=true\n",count);fflush(stdout);
 }
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));for(void*p:owned)ck(cudaFree(p));printf("full_model_prefill cases=%u replays=%u compared_bytes=%llu\n",cases,replays,(unsigned long long)compared);
}
