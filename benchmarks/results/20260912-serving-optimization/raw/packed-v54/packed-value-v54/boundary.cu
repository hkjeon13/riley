#include "prefill_query_tile_attention.cuh"
#include "baseline.cuh"
#include "packed.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#define CK(e) do{auto z=(e);if(z!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(z));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int packed_at(int i){int t=(i%1024)/64,d=i%64;return(i/1024)*1024+(d/8)*128+(t/8)*64+(d%8)*8+t%8;}
int main(int argc,char**argv){bool timing=argc>1;constexpr int capacity=1024,physical=8192;auto*q=alloc<__nv_bfloat16>(capacity*576);auto*k=alloc<__nv_bfloat16>(physical*16*192);auto*v=alloc<__nv_bfloat16>(physical*16*192);auto*vp=alloc<__nv_bfloat16>(physical*16*192);auto*a=alloc<__nv_bfloat16>(capacity*576+16);auto*b=alloc<__nv_bfloat16>(capacity*576+16);auto*m=alloc<uint32_t>(32+32*416+2048);int cases=0;
 for(int seed:{99}){for(int i=0;i<capacity*576;++i)q[i]=__float2bfloat16_rn(float((i*7+seed)%127-63)/128);for(int i=0;i<physical*16*192;++i){k[i]=__float2bfloat16_rn(float((i*17+seed)%251-125)/256);v[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/64);}for(int i=0;i<physical*16*192;++i)vp[packed_at(i)]=v[i];
 for(int pattern:{2,3,4})for(int owners:{1,4,32})for(int exceptional=0;exceptional<4;++exceptional){if(timing&&(seed!=1||exceptional!=0))continue;std::memset(m,0,(32+32*416+2048)*4);unsigned total=0,tiles=0;m[5]=owners;
  for(int i=0;i<owners;++i){auto*s=m+32+i*416;unsigned n=1;if(pattern==0)n=i==0?128:1;if(pattern==1)n=i<4?128:1;if(pattern==3){unsigned list[4]={1,7,31,33};n=list[i%4];}if(pattern==4)n=i==0?398:1;if(pattern==5)n=31;unsigned prefix=4096-n;s[1]=prefix+n-1;s[2]=n;s[4]=prefix;s[16]=total;s[17]=tiles;for(int p=0;p<256;++p)s[32+p]=i*256+(p*5+17)%256;total+=n;unsigned nt=n<32?n:(n+7)/8;for(unsigned j=0;j<nt;++j)m[14368+tiles+j]=(i<<16)|j;tiles+=nt;}m[9]=total;m[24]=tiles;if(total>capacity)exit(5);
  int owner=owners>1?1:0;auto*s=m+32+owner*416;int pos=exceptional==3?0:s[1];int at=((s[32+pos/16]*3)*16+pos%16)*64;auto old=v[at];if(exceptional)v[at]=__ushort_as_bfloat16(exceptional==2?0x7f80:0x7fc1);vp[packed_at(at)]=v[at];
  for(int i=0;i<capacity*576+16;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
  for(int i=0;i<owners;++i){auto*t=m+32+i*416;unsigned count=t[2];riley_prefill_query_tile::attention<8><<<dim3(max((count+7)/8,min(count,31U)),9),32>>>(q+t[16]*576,k,v,a+t[16]*576,count,t[1]+1,nullptr,t+32);}
  for(unsigned variant:{0,1,2}){for(int i=0;i<capacity*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
  if(variant==0)riley_mixed_attention::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,b,capacity,m);
  if(variant==1)riley_attention_v54::mapped_attention<false><<<dim3(capacity,9),32>>>(q,k,vp,b,capacity,m);
  if(variant==2)riley_attention_v54::mapped_attention<true><<<dim3(capacity,9),32>>>(q,k,vp,b,capacity,m);
  CK(cudaDeviceSynchronize());for(int i=0;i<capacity*576+16;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch seed%d pattern%d owners%d exceptional%d variant%u at%d\n",seed,pattern,owners,exceptional,variant,i);exit(3);}}
  if(timing){unsigned benchmark_capacity=total<=512?512:1024;cudaGraph_t graph[3];cudaGraphExec_t exec[3];cudaStream_t stream;CK(cudaStreamCreate(&stream));for(int variant=0;variant<3;++variant){CK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));if(variant==0)riley_mixed_attention::mapped_attention<<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,v,b,benchmark_capacity,m);
if(variant==1)riley_attention_v54::mapped_attention<false><<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,vp,b,benchmark_capacity,m);
if(variant==2)riley_attention_v54::mapped_attention<true><<<dim3(benchmark_capacity,9),32,0,stream>>>(q,k,vp,b,benchmark_capacity,m);
CK(cudaStreamEndCapture(stream,&graph[variant]));CK(cudaGraphInstantiate(&exec[variant],graph[variant],0));}
   cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));for(int pair=0;pair<4;++pair)for(int order=0;order<3;++order){int variant=pair%2?2-order:order;for(int i=0;i<10;++i)CK(cudaGraphLaunch(exec[variant],stream));CK(cudaEventRecord(start,stream));for(int i=0;i<100;++i)CK(cudaGraphLaunch(exec[variant],stream));CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,start,end));printf("{\"pattern\":%d,\"owners\":%d,\"tokens\":%u,\"tiles\":%u,\"pair\":%d,\"variant\":%d,\"us\":%.6f}\n",pattern,owners,total,tiles,pair,variant,ms*10);}
   CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));for(int i=0;i<3;++i){CK(cudaGraphExecDestroy(exec[i]));CK(cudaGraphDestroy(graph[i]));}CK(cudaStreamDestroy(stream));
  }
  v[at]=old;vp[packed_at(at)]=old;++cases;
 }}
 for(int owners:{0,33}){m[5]=owners;for(int i=0;i<capacity*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);riley_attention_v54::mapped_attention<true><<<dim3(capacity,9),32>>>(q,k,vp,b,capacity,m);CK(cudaDeviceSynchronize());for(int i=0;i<capacity*576+16;++i)if(__bfloat16_as_ushort(b[i])!=0x4123)exit(4);++cases;}
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)vp,(void*)a,(void*)b,(void*)m})CK(cudaFree(p));fprintf(stderr,"mixed attention cases=%d three_variants_exact=true packed_value=true finite_nonfinite_causal_owner_isolation=true inactive_guards=true\n",cases);
}
