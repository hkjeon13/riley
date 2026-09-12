#include "compact_shared_result.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
float fp(uint16_t b){uint32_t bits=uint32_t(b)<<16;float x;std::memcpy(&x,&bits,4);return x;}
template<uint32_t Rows>void test(){
 uint32_t *m,*output,*status;riley_compact_result::GreedyPartial* partial;__nv_bfloat16* logits;
 const size_t words=(128+Rows*1664+4096)/4;
 CK(cudaMalloc(&m,words*4));CK(cudaMalloc(&output,(Rows*32+32)*4));CK(cudaMalloc(&status,4));CK(cudaMalloc(&partial,Rows*24*sizeof(*partial)));CK(cudaMalloc(&logits,Rows*49152*2));
 int cases=0;
 for(uint32_t active=0;active<=Rows+1;++active)for(int kind=0;kind<7;++kind){
  if(kind==5&&active!=1)continue;
  std::vector<uint32_t> meta(words,0),expect(Rows*32+32,0),actual(expect.size());std::vector<uint16_t> x(Rows*49152,0x7fc1);
  meta[4]=kind==5?0:1;meta[5]=active;meta[8]=0;
  for(int i=0;i<6;++i)meta[10+i]=17+i*37;for(int i=0;i<8;++i)meta[16+i]=111+i*19;
  uint32_t fault=kind==6?1:0;
  for(uint32_t row=0;row<Rows&&row<active;++row){
   uint32_t* s=meta.data()+32+row*416;bool published=kind!=5;
   s[1]=published?128+row:15;s[2]=published?1:16;s[5]=s[1]+1;s[6]=published?row+1:0;s[8]=128;s[9]=1024;s[10]=active-1-row;s[11]=published?0:UINT32_MAX;
   for(int i=0;i<4;++i)s[12+i]=1001+row*31+i;
   for(int i=0;i<49152;++i){uint16_t b=uint16_t(i*97+row*197);if((b&0x7f80)==0x7f80)b=0;x[row*49152+i]=kind==1?(i%2?0x8000:0):b;}
   if(kind==2){std::fill(x.begin()+row*49152,x.begin()+(row+1)*49152,0xbf80);x[row*49152+255]=x[row*49152+49151]=0x4000;}
   if(kind==3)x[row*49152+49151]=0x7fc1;if(kind==4)x[row*49152]=0x7f80;
   if(active>Rows)continue;
   float maximum=-INFINITY;uint32_t token=UINT32_MAX,bad=0;
   for(int i=0;i<49152;++i){float v=fp(x[row*49152+i]);if(!std::isfinite(v))bad=1;else if(token==UINT32_MAX||v>maximum){maximum=v;token=i;}}
   auto* e=expect.data()+row*32;e[0]=fault;e[1]=published;e[2]=published?token:0;e[3]=published?bad:0;
   for(int i=0;i<6;++i)e[4+i]=meta[10+i];for(int i=0;i<4;++i)e[10+i]=s[12+i];
   e[14]=s[5];e[15]=s[8];e[16]=s[6];e[17]=s[2];e[18]=s[9];e[19]=meta[4];for(int i=0;i<8;++i)e[20+i]=meta[16+i];
   e[28]=s[1];e[29]=s[10];e[30]=0;e[31]=(Rows==8?0x33524d52U:0x34524d52U)^0x80000000U;
  }
  for(size_t i=Rows*32;i<expect.size();++i)expect[i]=0x51515151;
  CK(cudaMemcpy(m,meta.data(),words*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(logits,x.data(),x.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(status,&fault,4,cudaMemcpyHostToDevice));CK(cudaMemset(output,0x51,expect.size()*4));CK(cudaMemset(partial,0x51,Rows*24*sizeof(*partial)));
  CK(riley_compact_result::enqueue<Rows>(0,m,logits,status,partial,output));CK(cudaDeviceSynchronize());CK(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
  if(actual!=expect){for(size_t i=0;i<actual.size();++i)if(actual[i]!=expect[i]){fprintf(stderr,"rows%u active%u kind%d word%zu got%u expected%u\n",Rows,active,kind,i,actual[i],expect[i]);break;}exit(3);}++cases;
 }
 for(void*p:{(void*)m,(void*)output,(void*)status,(void*)partial,(void*)logits})CK(cudaFree(p));printf("compact capacity=%u cases=%d identities_status_tokens_inactive_guard=true\n",Rows,cases);
}
int main(){test<8>();test<16>();}
