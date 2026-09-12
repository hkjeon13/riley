#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "decode_shared_model.cuh"
#include "prefill_shape_packet.hpp"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T read(std::ifstream& f){T x{};f.read((char*)&x,sizeof(x));if(!f)exit(8);return x;}
int main(int argc,char**argv){if(argc!=2)return 1;std::string root=argv[1];std::ifstream wf(root+"/weights.bin",std::ios::binary);std::vector<const void*>weights(273);std::vector<void*>owned;
 for(int i=0;i<273;++i){auto bytes=read<uint64_t>(wf);if(!bytes){weights[i]=weights[0];continue;}std::vector<uint8_t>v(bytes);wf.read((char*)v.data(),bytes);if(!wf)exit(8);void*p;ck(cudaMalloc(&p,bytes));ck(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));weights[i]=p;owned.push_back(p);}
 constexpr unsigned cap=512,physical=512,context=1024;size_t cachebytes=30ull*physical*16*192*2;std::vector<void*>scratch(12);unsigned widths[]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};for(int i=0;i<12;++i){ck(cudaMalloc(&scratch[i],i==7?8*9*4096*4:cap*widths[i]));owned.push_back(scratch[i]);}
 void *keys,*values,*cos,*sin,*selected,*metadata;uint32_t *status,*publish;uint8_t* host;
 auto alloc=[&](void**p,size_t n){ck(cudaMalloc(p,n));owned.push_back(*p);};alloc(&keys,cachebytes);alloc(&values,cachebytes);alloc(&cos,context*32*4);alloc(&sin,context*32*4);alloc(&selected,1152);alloc(&metadata,17536);ck(cudaMalloc(&status,4));ck(cudaMalloc(&publish,4));owned.push_back(status);owned.push_back(publish);ck(cudaMallocHost(&host,17536));
 std::ifstream rf(root+"/rope.bin",std::ios::binary);std::vector<float>table(context*32);for(void*p:{cos,sin}){rf.read((char*)table.data(),table.size()*4);if(!rf)exit(8);ck(cudaMemcpy(p,table.data(),table.size()*4,cudaMemcpyHostToDevice));}

 cudaStream_t stream;ck(cudaStreamCreate(&stream));
 auto u32=[&](int at,uint32_t value){std::memcpy(host+at,&value,4);};
 std::ifstream input(root+"/requests.bin",std::ios::binary);unsigned count=read<uint32_t>(input);std::vector<std::vector<uint32_t>> corpus;
 for(unsigned i=0;i<count;++i){unsigned n=read<uint32_t>(input);corpus.emplace_back(n);input.read((char*)corpus.back().data(),n*4);if(!input)exit(8);}
 auto row=[&](int slot,int request,unsigned pos,uint32_t token){unsigned base=128+slot*1664;
  u32(base,token);u32(base+4,pos);u32(base+8,1);u32(base+16,pos);u32(base+44,0);
  for(int page=0;page<64;++page)u32(base+128+page*4,request*64+(page*17+3)%64);
 };
 ck(cudaMemset(keys,0,cachebytes));ck(cudaMemset(values,0,cachebytes));
 for(int request=0;request<8;++request){auto& tokens=corpus[request%corpus.size()];std::memset(host,0,17536);u32(20,1);row(0,request,tokens.size()-1,tokens[0]);u32(136,tokens.size());u32(144,0);u32(172,tokens.size()-1);
  for(unsigned j=0;j<tokens.size();++j)u32(13440+j*4,tokens[j]);
  ck(cudaMemcpyAsync(metadata,host,17536,cudaMemcpyHostToDevice,stream));ck(enqueue_v3_prefill_model(stream,scratch.data(),weights.data(),metadata,keys,values,cos,sin,selected,status,publish,cap,physical));ck(cudaStreamSynchronize(stream));
 }
 std::vector<uint8_t> initial_k(cachebytes),initial_v(cachebytes),expected_k(cachebytes),expected_v(cachebytes),actual_k(cachebytes),actual_v(cachebytes);
 ck(cudaMemcpy(initial_k.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(initial_v.data(),values,cachebytes,cudaMemcpyDeviceToHost));
 auto reset=[&](){ck(cudaMemcpy(keys,initial_k.data(),cachebytes,cudaMemcpyHostToDevice));ck(cudaMemcpy(values,initial_v.data(),cachebytes,cudaMemcpyHostToDevice));};
 cudaGraph_t graph;cudaGraphExec_t exec;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 ck(cudaMemcpyAsync(metadata,host,17536,cudaMemcpyHostToDevice,stream));ck(riley_shared_model::enqueue(stream,scratch.data(),weights.data(),metadata,keys,values,(float*)cos,(float*)sin,status,physical,context));
 ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 auto seed_k=initial_k,seed_v=initial_v;
 for(int active:{1,2,4,8}){
  initial_k=seed_k;initial_v=seed_v;
  for(int step=0;step<8;++step){
  reset();std::vector<uint8_t> expected(active*1152),actual(active*1152);
  for(int request=0;request<active;++request){auto& tokens=corpus[request%corpus.size()];std::memset(host,0,17536);u32(16,1);u32(20,1);row(0,request,tokens.size()+step,17+request+step);
   ck(cudaMemcpyAsync(metadata,host,17536,cudaMemcpyHostToDevice,stream));ck(enqueue_v3_prefill_model(stream,scratch.data(),weights.data(),metadata,keys,values,cos,sin,selected,status,publish,1,physical));ck(cudaStreamSynchronize(stream));
   ck(cudaMemcpy(expected.data()+request*1152,selected,1152,cudaMemcpyDeviceToHost));
  }
  ck(cudaMemcpy(expected_k.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(expected_v.data(),values,cachebytes,cudaMemcpyDeviceToHost));
  reset();std::memset(host,0,17536);u32(16,1);u32(20,active);
  for(int request=0;request<active;++request)row(request,request,corpus[request%corpus.size()].size()+step,17+request+step);
  ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
  ck(cudaMemcpy(actual.data(),scratch[1],actual.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual_k.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual_v.data(),values,cachebytes,cudaMemcpyDeviceToHost));
  uint32_t error;ck(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));
  if(error||actual!=expected||actual_k!=expected_k||actual_v!=expected_v){fprintf(stderr,"shared model mismatch active=%d status=%u hidden=%d keys=%d values=%d\n",active,error,actual!=expected,actual_k!=expected_k,actual_v!=expected_v);exit(5);}
  initial_k=actual_k;initial_v=actual_v;
  printf("shared_model active=%d step=%d real_prefill_contexts=16,128,398 hidden_exact=true full_kv_exact=true graph_replay=true\n",active,step);fflush(stdout);
  }
 }
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));for(void*p:owned)ck(cudaFree(p));
}
