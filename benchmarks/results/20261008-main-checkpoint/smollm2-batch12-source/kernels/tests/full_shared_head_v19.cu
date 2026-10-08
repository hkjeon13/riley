#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "decode_shared_model.cuh"
#include "decode_shared_result.cuh"
#include <cublasLt.h>
#include "prefill_shape_packet.hpp"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T read(std::ifstream& f){T x{};f.read((char*)&x,sizeof(x));if(!f)exit(8);return x;}

void cb(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cublas status %d\n",int(s));exit(12);}}
struct Head{
 cublasLtHandle_t handle; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t w,x,y;cublasLtMatmulAlgo_t algo;
 Head(int rows){cb(cublasLtCreate(&handle));cb(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));cublasOperation_t trans=CUBLAS_OP_T;cb(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
 cb(cublasLtMatrixLayoutCreate(&w,CUDA_R_16BF,576,49152,576));cb(cublasLtMatrixLayoutCreate(&x,CUDA_R_16BF,576,rows,576));cb(cublasLtMatrixLayoutCreate(&y,CUDA_R_16BF,49152,rows,49152));
 cublasLtMatmulPreference_t pref;cb(cublasLtMatmulPreferenceCreate(&pref));size_t workspace=0;cb(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));cublasLtMatmulHeuristicResult_t choices[32];int count=0;cb(cublasLtMatmulAlgoGetHeuristic(handle,op,w,x,y,y,pref,32,choices,&count));bool found=false;
 for(int i=0;i<count;++i){uint32_t split=0,reduce=0;size_t size;cb(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&size));cb(cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduce,sizeof(reduce),&size));if(choices[i].state==CUBLAS_STATUS_SUCCESS&&choices[i].workspaceSize==0&&split<=1&&reduce==0){algo=choices[i].algo;found=true;break;}}
 if(!found)exit(13);int id;size_t size;cb(cublasLtMatmulAlgoConfigGetAttribute(&algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&size));printf("head rows=%d algorithm=%d workspace=0\n",rows,id);cb(cublasLtMatmulPreferenceDestroy(pref));
 }
 void run(cudaStream_t stream,const void* weight,const void* input,void* output){float alpha=1,beta=0;cb(cublasLtMatmul(handle,op,&alpha,weight,w,input,x,&beta,output,y,output,y,&algo,nullptr,0,stream));}
 ~Head(){cublasLtMatrixLayoutDestroy(w);cublasLtMatrixLayoutDestroy(x);cublasLtMatrixLayoutDestroy(y);cublasLtMatmulDescDestroy(op);cublasLtDestroy(handle);}
};
int main(int argc,char**argv){if(argc!=2)return 1;std::string root=argv[1];std::ifstream wf(root+"/weights.bin",std::ios::binary);std::vector<const void*>weights(273);std::vector<void*>owned;
 for(int i=0;i<273;++i){auto bytes=read<uint64_t>(wf);if(!bytes){weights[i]=weights[0];continue;}std::vector<uint8_t>v(bytes);wf.read((char*)v.data(),bytes);if(!wf)exit(8);void*p;ck(cudaMalloc(&p,bytes));ck(cudaMemcpy(p,v.data(),bytes,cudaMemcpyHostToDevice));weights[i]=p;owned.push_back(p);}
 constexpr unsigned cap=512,physical=512,context=1024;size_t cachebytes=30ull*physical*16*192*2;std::vector<void*>scratch(12);unsigned widths[]={1152,1152,1152,1152,1152,384,384,384,3072,3072,2304,3072};for(int i=0;i<12;++i){ck(cudaMalloc(&scratch[i],i==7?8*9*4096*4:cap*widths[i]));owned.push_back(scratch[i]);}
 void *keys,*values,*cos,*sin,*selected,*metadata;uint32_t *status,*publish;uint8_t* host;
 auto alloc=[&](void**p,size_t n){ck(cudaMalloc(p,n));owned.push_back(*p);};alloc(&keys,cachebytes);alloc(&values,cachebytes);alloc(&cos,context*32*4);alloc(&sin,context*32*4);alloc(&selected,1152);alloc(&metadata,17536);ck(cudaMalloc(&status,4));ck(cudaMalloc(&publish,4));owned.push_back(status);owned.push_back(publish);ck(cudaMallocHost(&host,17536));
 std::ifstream rf(root+"/rope.bin",std::ios::binary);std::vector<float>table(context*32);for(void*p:{cos,sin}){rf.read((char*)table.data(),table.size()*4);if(!rf)exit(8);ck(cudaMemcpy(p,table.data(),table.size()*4,cudaMemcpyHostToDevice));}

 cudaStream_t stream;ck(cudaStreamCreate(&stream));
 Head single(1),shared(8);void *logits,*records;alloc(&logits,8*98304);alloc(&records,riley_shared_result::batch_bytes);
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
 shared.run(stream,weights[2],scratch[1],logits);ck(riley_shared_result::enqueue(stream,metadata,(const __nv_bfloat16*)logits,status,records));
 ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 auto seed_k=initial_k,seed_v=initial_v;
 for(int active:{1,2,4,8}){
  initial_k=seed_k;initial_v=seed_v;
  for(int step=0;step<8;++step){
  reset();std::vector<uint8_t> expected(active*1152),actual(active*1152),expected_logits(active*98304),actual_records(riley_shared_result::batch_bytes);
  for(int request=0;request<active;++request){auto& tokens=corpus[request%corpus.size()];std::memset(host,0,17536);u32(16,1);u32(20,1);row(0,request,tokens.size()+step,17+request+step);
   ck(cudaMemcpyAsync(metadata,host,17536,cudaMemcpyHostToDevice,stream));ck(enqueue_v3_prefill_model(stream,scratch.data(),weights.data(),metadata,keys,values,cos,sin,selected,status,publish,1,physical));ck(cudaStreamSynchronize(stream));
   ck(cudaMemcpy(expected.data()+request*1152,selected,1152,cudaMemcpyDeviceToHost));
   single.run(stream,weights[2],selected,logits);ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(expected_logits.data()+request*98304,logits,98304,cudaMemcpyDeviceToHost));
  }
  ck(cudaMemcpy(expected_k.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(expected_v.data(),values,cachebytes,cudaMemcpyDeviceToHost));
  reset();std::memset(host,0,17536);u32(16,1);u32(20,active);
  for(int request=0;request<active;++request)row(request,request,corpus[request%corpus.size()].size()+step,17+request+step);
  ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
  ck(cudaMemcpy(actual.data(),scratch[1],actual.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual_k.data(),keys,cachebytes,cudaMemcpyDeviceToHost));ck(cudaMemcpy(actual_v.data(),values,cachebytes,cudaMemcpyDeviceToHost));
  uint32_t error;ck(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));
  if(error||actual!=expected||actual_k!=expected_k||actual_v!=expected_v){fprintf(stderr,"shared model mismatch active=%d status=%u hidden=%d keys=%d values=%d\n",active,error,actual!=expected,actual_k!=expected_k,actual_v!=expected_v);exit(5);}
  ck(cudaMemcpy(actual_records.data(),records,actual_records.size(),cudaMemcpyDeviceToHost));
  for(int request=0;request<active;++request){auto* record=actual_records.data()+request*98432;auto* expected_row=expected_logits.data()+request*98304;
   if(std::memcmp(record+128,expected_row,98304)){fprintf(stderr,"head logits differ active=%d step=%d row=%d\n",active,step,request);exit(15);}
   uint32_t token=0;float maximum=-INFINITY;for(uint32_t i=0;i<49152;++i){uint16_t bits;std::memcpy(&bits,expected_row+2*i,2);uint32_t fbits=uint32_t(bits)<<16;float value;std::memcpy(&value,&fbits,4);if(value>maximum){maximum=value;token=i;}}
   auto* words=reinterpret_cast<const uint32_t*>(record);if(words[0]||words[1]!=1||words[2]!=token||words[3])exit(16);
  }
  for(size_t i=active*98432;i<actual_records.size();++i)if(actual_records[i])exit(17);
  initial_k=actual_k;initial_v=actual_v;
  printf("shared_model active=%d step=%d real_prefill_contexts=16,128,398 hidden_exact=true full_kv_exact=true logits_exact=true result_argmax_exact=true inactive_records_zero=true graph_replay=true\n",active,step);fflush(stdout);
  }
 }
 ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));for(void*p:owned)ck(cudaFree(p));
}
