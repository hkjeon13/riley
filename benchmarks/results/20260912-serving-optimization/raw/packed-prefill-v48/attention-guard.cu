#include "prefill_query_tile_attention.cuh"
#include "packed_prefill_attention_v48.cuh"
#include <cstdio>
#include <cstdlib>
#define CK(e) do{auto z=(e);if(z!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(z));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(){auto*q=alloc<__nv_bfloat16>(512*576);auto*k=alloc<__nv_bfloat16>(256*16*192);auto*v=alloc<__nv_bfloat16>(256*16*192);auto*a=alloc<__nv_bfloat16>(512*576+16);auto*b=alloc<__nv_bfloat16>(512*576+16);auto*m=alloc<uint32_t>(32+32*416);int patterns[3][4]={{1,7,31,33},{16,128,129,239},{128,128,128,128}};int prefix[4]={0,17,127,511};int cases=0;
 for(int seed:{1,17}){for(int i=0;i<512*576;++i)q[i]=__float2bfloat16_rn(float((i*7+seed)%127-63)/128);for(int i=0;i<256*16*192;++i){k[i]=__float2bfloat16_rn(float((i*17+seed)%251-125)/256);v[i]=__float2bfloat16_rn(float((i*13+seed)%127-63)/64);}
 for(int pattern=0;pattern<3;++pattern)for(int owners=1;owners<=4;++owners)for(int exceptional=0;exceptional<4;++exceptional){for(int i=0;i<32+32*416;++i)m[i]=0;unsigned total=0;m[5]=owners;for(int i=0;i<owners;++i){auto*s=m+32+i*416;s[1]=prefix[i]+patterns[pattern][i]-1;s[2]=patterns[pattern][i];s[4]=prefix[i];s[16]=total;for(int p=0;p<64;++p)s[32+p]=i*64+(p*5+17)%64;total+=s[2];}m[9]=total;
  int owner=owners>1?1:0;auto*s=m+32+owner*416;int pos=exceptional==3?0:s[1];int at=((s[32+pos/16]*3)*16+pos%16)*64;auto old=v[at];if(exceptional)v[at]=__ushort_as_bfloat16(exceptional==2?0x7f80:0x7fc1);
  for(int i=0;i<512*576+16;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
  for(int i=0;i<owners;++i){auto*t=m+32+i*416;unsigned count=t[2];riley_prefill_query_tile::attention<8><<<dim3(max((count+7)/8,min(count,31U)),9),32>>>(q+t[16]*576,k,v,a+t[16]*576,count,t[1]+1,nullptr,t+32);}
  riley_packed_prefill_attention::attention<<<dim3(64,9,4),32>>>(q,k,v,b,512,m);CK(cudaDeviceSynchronize());for(int i=0;i<512*576+16;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"attention mismatch seed%d pattern%d owners%d exceptional%d at%d\n",seed,pattern,owners,exceptional,i);exit(3);}v[at]=old;++cases;
 }}
 for(int owners:{0,5}){m[5]=owners;for(int i=0;i<512*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);riley_packed_prefill_attention::attention<<<dim3(64,9,4),32>>>(q,k,v,b,512,m);CK(cudaDeviceSynchronize());for(int i=0;i<512*576+16;++i)if(__bfloat16_as_ushort(b[i])!=0x4123)exit(4);++cases;}
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)m})CK(cudaFree(p));printf("packed attention cases=%d finite_nonfinite_causal_owner_isolation=true inactive_guards=true\n",cases);
}
