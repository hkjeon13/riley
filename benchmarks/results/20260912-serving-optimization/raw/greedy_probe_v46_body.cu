#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
float value(uint16_t b){uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,4);return f;}
RileyCudaBf16ArgmaxResult reference(const uint16_t* x,int n){
 float best=-INFINITY;uint32_t id=UINT32_MAX;bool invalid=false;
 for(int i=0;i<n;++i){float f=value(x[i]);if(!std::isfinite(f)){invalid=true;continue;}if(id==UINT32_MAX||f>best){best=f;id=i;}}
 return invalid?RileyCudaBf16ArgmaxResult{UINT32_MAX,1}:RileyCudaBf16ArgmaxResult{id,0};
}
void check(const std::vector<uint16_t>& x,int rows,int vocab){
 __nv_bfloat16* d;RileyCudaBf16ArgmaxResult* result;CK(cudaMalloc(&d,x.size()*2));CK(cudaMalloc(&result,(rows+16)*8));CK(cudaMemcpy(d,x.data(),x.size()*2,cudaMemcpyHostToDevice));CK(cudaMemset(result,0x51,(rows+16)*8));
 bf16_argmax_kernel<<<std::min(rows,65535),256>>>(d,result,rows,vocab);CK(cudaDeviceSynchronize());std::vector<RileyCudaBf16ArgmaxResult> actual(rows+16);CK(cudaMemcpy(actual.data(),result,(rows+16)*8,cudaMemcpyDeviceToHost));
 for(int row=0;row<rows;++row){auto expected=reference(x.data()+row*vocab,vocab);if(expected.token_id!=actual[row].token_id||expected.status!=actual[row].status){fprintf(stderr,"argmax mismatch row=%d vocab=%d expected=%u/%u actual=%u/%u\n",row,vocab,expected.token_id,expected.status,actual[row].token_id,actual[row].status);exit(3);}}
 for(int row=rows;row<rows+16;++row)if(actual[row].token_id!=0x51515151||actual[row].status!=0x51515151)exit(4);
 CK(cudaFree(result));CK(cudaFree(d));printf("CHECK rows=%d vocab=%d exact=true guard=true\n",rows,vocab);
}
void correctness(){
 std::vector<uint16_t> exhaustive(65536*2);
 for(int order=0;order<2;++order){for(int i=0;i<65536;++i){exhaustive[i*2+order]=i;exhaustive[i*2+1-order]=uint16_t(i^0x8000);}check(exhaustive,65536,2);}
 for(int rows:{1,4,8,16})for(int vocab:{257,49152}){
  std::vector<uint16_t> x(rows*vocab);
  for(int row=0;row<rows;++row)for(int i=0;i<vocab;++i){uint16_t b=(i*97+row*7919)&65535;if((b&0x7f80)==0x7f80)b=0;x[row*vocab+i]=b;}check(x,rows,vocab);
  for(int pos:{0,31,32,255,256}){std::fill(x.begin(),x.end(),0xbf80);for(int row=0;row<rows;++row){x[row*vocab+pos]=0x4000;x[row*vocab+vocab-1]=0x4000;}check(x,rows,vocab);}
  for(uint16_t b:{uint16_t(0x7f80),uint16_t(0xff80),uint16_t(0x7fc1),uint16_t(0x7f81),uint16_t(0xffc1)})for(int pos:{0,31,32,255,256,vocab-1}){std::fill(x.begin(),x.end(),0);for(int row=0;row<rows;++row)x[row*vocab+pos]=b;check(x,rows,vocab);}
  for(int order=0;order<2;++order){for(size_t i=0;i<x.size();++i)x[i]=(i+order)%2?0:0x8000;check(x,rows,vocab);}
 }
}
void timing(){
 constexpr int vocab=49152;
 for(int rows:{1,4,8,16}){
  std::vector<uint16_t>x(rows*vocab);for(size_t i=0;i<x.size();++i)x[i]=__bfloat16_as_ushort(__float2bfloat16_rn(float(int(i*97%4096)-2048)/2048));
  __nv_bfloat16* d;RileyCudaBf16ArgmaxResult* result;uint16_t* full;RileyCudaBf16ArgmaxResult* compact;cudaStream_t stream;CK(cudaStreamCreate(&stream));CK(cudaMalloc(&d,x.size()*2));CK(cudaMalloc(&result,rows*8));CK(cudaMallocHost(&full,x.size()*2));CK(cudaMallocHost(&compact,rows*8));CK(cudaMemcpy(d,x.data(),x.size()*2,cudaMemcpyHostToDevice));
  cudaGraph_t graph[2];cudaGraphExec_t exec[2];
  for(int mode=0;mode<2;++mode){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));if(mode){bf16_argmax_kernel<<<rows,256,0,stream>>>(d,result,rows,vocab);CK(cudaMemcpyAsync(compact,result,rows*8,cudaMemcpyDeviceToHost,stream));}else CK(cudaMemcpyAsync(full,d,x.size()*2,cudaMemcpyDeviceToHost,stream));CK(cudaStreamEndCapture(stream,&graph[mode]));CK(cudaGraphInstantiate(&exec[mode],graph[mode],0));}
  cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
  for(int pair=0;pair<4;++pair)for(int order=0;order<2;++order){int mode=(pair+order)%2;for(int i=0;i<30;++i)CK(cudaGraphLaunch(exec[mode],stream));CK(cudaStreamSynchronize(stream));auto wall=std::chrono::steady_clock::now();CK(cudaEventRecord(begin,stream));for(int i=0;i<100;++i)CK(cudaGraphLaunch(exec[mode],stream));CK(cudaEventRecord(end,stream));CK(cudaStreamSynchronize(stream));auto finished=std::chrono::steady_clock::now();float ms;CK(cudaEventElapsedTime(&ms,begin,end));
   printf("{\"rows\":%d,\"pair\":%d,\"compact\":%s,\"gpu_us\":%.6f,\"host_us\":%.6f}\n",rows,pair,mode?"true":"false",ms*10.,std::chrono::duration<double,std::micro>(finished-wall).count()/100.);
   if(mode){for(int row=0;row<rows;++row){auto e=reference(x.data()+row*vocab,vocab);if(compact[row].token_id!=e.token_id||compact[row].status!=e.status)exit(5);}}else if(std::memcmp(full,x.data(),x.size()*2))exit(6);
  }
  CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));for(int mode=0;mode<2;++mode){CK(cudaGraphExecDestroy(exec[mode]));CK(cudaGraphDestroy(graph[mode]));}CK(cudaStreamDestroy(stream));CK(cudaFree(d));CK(cudaFree(result));CK(cudaFreeHost(full));CK(cudaFreeHost(compact));
 }
}
int main(int argc,char**argv){if(argc>1&&std::strcmp(argv[1],"time")==0)timing();else correctness();}
