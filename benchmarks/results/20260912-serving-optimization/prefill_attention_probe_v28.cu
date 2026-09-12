#define riley_prefill_shape prefill_reference
#include "prefill_attention_v27_reference.cuh"
#undef riley_prefill_shape
#include "prefill_shape_attention.cuh"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <initializer_list>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
int main(){
 auto*q=alloc<__nv_bfloat16>(1024*576);auto*k=alloc<__nv_bfloat16>(4096*192);auto*v=alloc<__nv_bfloat16>(4096*192);auto*a=alloc<__nv_bfloat16>(1024*576+16);auto*b=alloc<__nv_bfloat16>(1024*576+16);auto*pages=alloc<uint32_t>(256);auto*live=alloc<uint32_t>(1);
 for(int i=0;i<1024*576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);for(int i=0;i<4096*192;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}for(int i=0;i<256;++i)pages[i]=(i*17+3)%256;
 for(bool paged:{false,true})for(auto shape:{std::pair{0,16},{1,1},{7,17},{16,16},{17,129},{73,398},{128,4096},{398,525},{1024,1024}}){
 *live=shape.first;for(int i=0;i<1024*576+16;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
 prefill_reference::attention_shape<<<dim3(1024,3),96>>>(q,k,v,a,1024,shape.second,nullptr,paged?pages:nullptr,live);
 riley_prefill_shape::attention_shape<<<dim3(1024,3),96>>>(q,k,v,b,1024,shape.second,nullptr,paged?pages:nullptr,live);OK(cudaDeviceSynchronize());
 for(int i=0;i<1024*576+16;++i){unsigned expected=i<shape.first*576?__bfloat16_as_ushort(a[i]):0x4123;if(__bfloat16_as_ushort(b[i])!=expected){fprintf(stderr,"mismatch paged%d rows%d context%d i%d\n",paged,shape.first,shape.second,i);exit(3);}}
 printf("prefill_attention paged%d rows%d context%d exact=true guards=true\n",paged,shape.first,shape.second);fflush(stdout);
 }
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)pages,(void*)live})OK(cudaFree(p));
}
