#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "../../kernels/optional/future_token.cuh"
using namespace riley_future_token;
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line=%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T* p;CHECK(cudaMalloc(&p,n*sizeof(T)));return p;}
// A device producer changes tokens on each replay. The consumer cannot obtain
// these values from host placeholders or its expected-identity sidecar.
__global__ void produce(unsigned* previous,const unsigned* epoch){
 unsigned row=threadIdx.x;if(previous[row*32+2]!=UINT32_MAX)previous[row*32+2]=100+row+*epoch;
}
int main(){
 auto* packet=alloc<unsigned>(PacketWords);auto* previous=alloc<unsigned>(1024);
 auto* refs=alloc<Reference>(32);auto* status=alloc<unsigned>(1);auto* epoch=alloc<unsigned>(1);
 cudaStream_t stream;CHECK(cudaStreamCreate(&stream));cudaGraph_t graph;cudaGraphExec_t exec;
 CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
 produce<<<1,32,0,stream>>>(previous,epoch);CHECK(cudaGetLastError());
 CHECK(enqueue(stream,packet,refs,previous,status));
 CHECK(cudaStreamEndCapture(stream,&graph));CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
 unsigned passed=0;
 for(unsigned rows:{1u,16u,32u})for(bool mixed:{false,true})for(unsigned fault=0;fault<48;++fault){
  std::vector<unsigned> m(PacketWords),r(1024),out(PacketWords);std::vector<Reference> f(32);
  m[0]=0x37444d52;m[1]=7;m[2]=PacketWords*4;m[3]=1664;m[4]=mixed?2:1;m[5]=rows;m[6]=32;m[8]=0;m[9]=mixed?rows:0;
  m[10]=7;m[12]=11;m[14]=21;for(unsigned i=0;i<8;++i)m[16+i]=1000+i;
  for(unsigned row=0;row<32;++row){
   unsigned source=31-row;auto* s=m.data()+32+row*416;auto* p=r.data()+source*32;
   s[1]=16;s[2]=1;s[4]=16;s[5]=17;s[6]=1;s[7]=32;s[8]=16;s[9]=4096;s[12]=700+row;s[14]=900+row;s[16]=mixed?row:0;s[18]=1;
   p[1]=1;p[4]=7;p[6]=10;p[8]=20;p[10]=700+row;p[12]=900+row;p[14]=16;p[15]=16;p[17]=16;p[18]=4096;p[28]=15;p[29]=row;p[31]=0xb7524d52;
   for(unsigned i=0;i<8;++i)p[20+i]=m[16+i];
   f[row].source_row=source;std::copy(p,p+32,f[row].expected);
  }
  auto* suffix=m.data()+32+(rows-1)*416;auto* old=r.data()+f[rows-1].source_row*32;
  bool fail=fault!=0 && fault!=47;
  if(fault>=1 && fault<=32){unsigned word=fault-1;if(word==2)old[word]=UINT32_MAX;else old[word]^=1;}
  if(fault==33)f[rows-1].source_row=32;
  if(fault==34)suffix[12]^=1;
  if(fault==35)suffix[14]^=1;
  if(fault==36)m[10]^=1;
  if(fault==37)m[12]+=1;
  if(fault==38)m[14]+=1;
  if(fault==39)suffix[4]+=1;
  if(fault==40)suffix[6]+=1;
  if(fault==41)suffix[0]=100;
  if(fault==42)suffix[18]=0;
  if(fault==43)m[5]=33;
  if(fault==44)m[0]=0;
  if(fault==45){suffix[16]=1024;if(!mixed)suffix[2]=2;}
  if(fault==47){f[rows-1].source_row=HostToken;suffix[0]=777;if(mixed)m[13344+rows-1]=777;}
  unsigned initial=fault==46?128:0,value=passed;
  CHECK(cudaMemcpy(packet,m.data(),PacketWords*4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(previous,r.data(),4096,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(refs,f.data(),sizeof(Reference)*32,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(status,&initial,4,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(epoch,&value,4,cudaMemcpyHostToDevice));
  CHECK(cudaGraphLaunch(exec,stream));CHECK(cudaStreamSynchronize(stream));unsigned error;
  CHECK(cudaMemcpy(&error,status,4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(out.data(),packet,PacketWords*4,cudaMemcpyDeviceToHost));
  if(fail){if(!error || out!=m){fprintf(stderr,"failure rows=%u mixed=%d fault=%u error=%u\n",rows,mixed,fault,error);return 3;}}
  else {if(error)return 4;for(unsigned row=0;row<rows;++row)if(f[row].source_row!=HostToken){unsigned token=100+f[row].source_row+value;m[32+row*416]=token;if(mixed)m[13344+row]=token;}if(out!=m)return 5;}
  ++passed;
 }
 printf("cases=%u graph_device_token=true reordered_rows=true invalid_suffix_no_mutation=true\n",passed);
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));CHECK(cudaStreamDestroy(stream));
 for(void* p:{(void*)packet,(void*)previous,(void*)refs,(void*)status,(void*)epoch})CHECK(cudaFree(p));
}
