#include "variant3.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do{auto err=(x);if(err!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(err));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(){constexpr int capacity=512,physical=2048;auto*q=alloc<__nv_bfloat16>(capacity*576);auto*k=alloc<__nv_bfloat16>(physical*16*192);auto*v=alloc<__nv_bfloat16>(physical*16*192);auto*out=alloc<__nv_bfloat16>(capacity*576);auto*m=alloc<uint32_t>(15392);memset(m,0,61568);
 for(int i=0;i<capacity*576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);
 for(int i=0;i<physical*16*192;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}
 m[5]=32;unsigned total=0,tiles=0;
 for(int i=0;i<32;++i){auto*s=m+32+i*416;unsigned n=i==0?398:1,prefix=i==0?0:128+(i%3)*128;s[1]=prefix+n-1;s[2]=n;s[4]=prefix;s[16]=total;s[17]=tiles;for(int p=0;p<64;++p)s[32+p]=i*64+(p*5+17)%64;unsigned nt=n<32?n:(n+7)/8;for(unsigned j=0;j<nt;++j)m[14368+tiles+j]=(i<<16)|j;tiles+=nt;total+=n;}
 m[9]=total;m[24]=tiles;
 cudaFuncAttributes attr;CK(cudaFuncGetAttributes(&attr,riley_attention_v52_3::mapped_attention));int blocks=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,riley_attention_v52_3::mapped_attention,32,0));printf("registers=%d static_shared_bytes=%zu max_active_blocks_per_sm=%d active_warps_per_sm=%d\n",attr.numRegs,attr.sharedSizeBytes,blocks,blocks);
 riley_attention_v52_3::mapped_attention<<<dim3(capacity,9),32>>>(q,k,v,out,capacity,m);CK(cudaDeviceSynchronize());
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)out,(void*)m})CK(cudaFree(p));
}
