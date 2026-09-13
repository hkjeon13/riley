#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <cstring>
extern "C" uint64_t riley_flashinfer_prefill_workspace_bytes() noexcept;
extern "C" int riley_flashinfer_prefill_prepare(void*,const void*,void*,uint64_t,unsigned,unsigned,void*) noexcept;
extern "C" int riley_flashinfer_prefill_only_prepare(void*,const void*,void*,uint64_t,unsigned,unsigned,void*) noexcept;
extern "C" int riley_flashinfer_prefill_run(void*,const void*,const void*,const void*,void*,void*,uint64_t) noexcept;
#define CHECK(x) do {auto e=(x);if(int(e)!=0){fprintf(stderr,"line%d %s code%d\n",__LINE__,#x,int(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
int main(int argc,char** argv){
 bool mixed=argc>2 && std::strcmp(argv[2],"--prefill-only")==0;
 auto prepare=mixed?riley_flashinfer_prefill_only_prepare:riley_flashinfer_prefill_prepare;
 FILE* dump=argc>1?std::fopen(argv[1],"wb"):nullptr;if(argc>1&&!dump)return 9;
 constexpr unsigned Physical=512,Capacity=1024,Stride=576;
 constexpr size_t Cache=Physical*3*16*64, Query=Capacity*Stride, Packet=32+32*416;
 auto q=alloc<__nv_bfloat16>(Query),k=alloc<__nv_bfloat16>(Cache),v=alloc<__nv_bfloat16>(Cache),o=alloc<__nv_bfloat16>(Query);
 auto packet=alloc<unsigned>(Packet),status=alloc<unsigned>(1);auto bytes=riley_flashinfer_prefill_workspace_bytes();auto ws=alloc<unsigned char>(bytes);
 std::vector<__nv_bfloat16> hq(Query),hk(Cache),hv(Cache),ho(Query);
 std::vector<unsigned> hp(Packet);uint64_t seed=9130401;
 auto random=[&](){seed=seed*6364136223846793005ULL+1;return float(int(seed>>48)-32768)/32768.F;};
 for(auto& x:hq)x=__float2bfloat16_rn(random());for(auto& x:hk)x=__float2bfloat16_rn(random());for(auto& x:hv)x=__float2bfloat16_rn(random()*.25F);
 CHECK(cudaMemcpy(q,hq.data(),Query*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(k,hk.data(),Cache*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(v,hv.data(),Cache*2,cudaMemcpyHostToDevice));
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 // Set function attributes before capture via a zero-work initialized packet.
 CHECK(cudaMemset(packet,0,Packet*4));CHECK(cudaMemset(status,0,4));
 CHECK(prepare(stream,packet,ws,bytes,Physical,Capacity,status));
 CHECK(riley_flashinfer_prefill_run(stream,q,k,v,o,ws,bytes));CHECK(cudaStreamSynchronize(stream));
 CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 CHECK(prepare(stream,packet,ws,bytes,Physical,Capacity,status));
 CHECK(riley_flashinfer_prefill_run(stream,q,k,v,o,ws,bytes));
 CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 std::vector<std::vector<unsigned>> qcases={{1,17,33},{16,32},{127,128,129},{1,17,33}};
 std::vector<std::vector<unsigned>> kcases={{16,129,511},{4096,128},{128,256,398},{16,129,511}};
 qcases.emplace_back(32,1);qcases.back().back()=993;
 kcases.emplace_back(32,16);kcases.back().back()=1024;
 if(mixed){
  qcases.push_back({1,1});kcases.push_back({16,128}); // all decode
  qcases.push_back({1,1,1});kcases.push_back({16,129,511}); // one-token prefill between decode rows
  qcases.push_back({17,1});kcases.push_back({32,16}); // invalid stage suffix
  qcases.push_back({17,2});kcases.push_back({32,16}); // decode requires exactly one query
  qcases.push_back({1,1,1});kcases.push_back({16,129,511}); // recovery after invalid metadata
 }
 for(unsigned test=0;test<qcases.size();++test){
  std::fill(hp.begin(),hp.end(),0);unsigned cursor=0,pages=0;hp[5]=qcases[test].size();
  for(unsigned r=0;r<hp[5];++r){auto s=hp.data()+32+r*416;s[1]=kcases[test][r]-1;s[2]=qcases[test][r];s[16]=cursor;cursor+=s[2];
   for(unsigned p=0;p<(kcases[test][r]+15)/16;++p)s[32+p]=((pages++)*17+7)%Physical;}
  if(mixed){
   for(unsigned r=0;r<hp[5];++r)hp[32+r*416+18]=qcases[test][r]==1?1:0;
   if(test==6||test==9)hp[32+416+18]=0;
   if(test==7)hp[32+416+18]=2;
   if(test==8)hp[32+416+18]=1;
  }
  hp[9]=cursor;if(test==3)hp[32+2*416+32]=Physical; // invalid suffix must invalidate all work
  std::fill(ho.begin(),ho.end(),__float2bfloat16_rn(-99.F));
  CHECK(cudaMemcpy(o,ho.data(),Query*2,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(packet,hp.data(),Packet*4,cudaMemcpyHostToDevice));CHECK(cudaMemset(status,0,4));
  CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));unsigned error;CHECK(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(ho.data(),o,Query*2,cudaMemcpyDeviceToHost));
  if(test==3||(mixed&&(test==7||test==8))){unsigned wanted=test==3?4:2;if(error!=wanted)return 3;for(auto x:ho)if(__bfloat162float(x)!=-99.F)return 4;printf("case=%u invalid_suffix_status=%u all_outputs_untouched=true\n",test,error);continue;}
  if(error)return 5;double max_error=0,square=0;size_t values=0;
  for(unsigned r=0;r<hp[5];++r){auto s=hp.data()+32+r*416;unsigned n=s[1]+1,rows=s[2],offset=s[16];
   if(mixed && s[18]==1){for(unsigned i=offset*Stride;i<(offset+rows)*Stride;++i)if(__bfloat162float(ho[i])!=-99.F)return 12;continue;}
   for(unsigned row=0;row<rows;++row)for(unsigned head=0;head<9;++head){unsigned count=n-rows+row+1;std::vector<double> scores(count);double maximum=-1e30;
    for(unsigned t=0;t<count;++t){double dot=0;unsigned base=((s[32+t/16]*3+head/3)*16+t%16)*64;
     for(unsigned d=0;d<64;++d)dot+=double(__bfloat162float(hq[(offset+row)*Stride+head*64+d]))*__bfloat162float(hk[base+d]);scores[t]=dot*.125;maximum=std::max(maximum,scores[t]);}
    double denominator=0;for(auto& z:scores){z=std::exp(z-maximum);denominator+=z;}
    for(unsigned d=0;d<64;++d){double expected=0;for(unsigned t=0;t<count;++t){unsigned base=((s[32+t/16]*3+head/3)*16+t%16)*64;expected+=scores[t]*__bfloat162float(hv[base+d]);}expected/=denominator;
     double actual=__bfloat162float(ho[(offset+row)*Stride+head*64+d]);if(!std::isfinite(actual))return 6;double error=std::abs(actual-expected);max_error=std::max(max_error,error);square+=error*error;++values;}
   }
  }
  for(size_t i=cursor*Stride;i<Query;++i)if(__bfloat162float(ho[i])!=-99.F)return 7;
  printf("case=%u query_rows=%u values=%zu max_abs=%.9g rmse=%.9g inactive_untouched=true\n",test,cursor,values,max_error,values?std::sqrt(square/values):0.0);fflush(stdout);
  // Predeclared synthetic functional tolerance only, not a model-quality gate.
  if(max_error>.01)return 8;
  if(dump&&std::fwrite(ho.data(),2,cursor*Stride,dump)!=cursor*Stride)return 10;
 }
 if(dump&&std::fclose(dump)!=0)return 11;
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
 for(void* p:{(void*)q,(void*)k,(void*)v,(void*)o,(void*)packet,(void*)status,(void*)ws})CHECK(cudaFree(p));
 return 0;
}
