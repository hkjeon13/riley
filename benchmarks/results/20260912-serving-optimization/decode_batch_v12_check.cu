#include "prefill_shape_projection.cuh"
#include "decode_shape.cuh"
#include <cstdio>
#include <cstdlib>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
void compare(const __nv_bfloat16*a,const __nv_bfloat16*b,int n){for(int i=0;i<n;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch %d %g %g\n",i,__bfloat162float(a[i]),__bfloat162float(b[i]));exit(3);}}
template<int N,int K,int I,int W>void projection(int seed){
 auto*x=alloc<__nv_bfloat16>(K);auto*w=alloc<__nv_bfloat16>(N*K);auto*a=alloc<__nv_bfloat16>(N);auto*b=alloc<__nv_bfloat16>(N);auto*p=alloc<float>(9*4096);
 for(int i=0;i<K;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 for(int i=0;i<N*K;++i)w[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);
 gemm_prefill_shape_vector<N,K,I,W><<<dim3(N/(8*W),1),32*W>>>(x,w,a,1);
 enqueue_decode_projection<N,K,I>(0,x,w,b,p);OK(cudaDeviceSynchronize());compare(a,b,N);
 OK(cudaFree(x));OK(cudaFree(w));OK(cudaFree(a));OK(cudaFree(b));OK(cudaFree(p));
}
int main(){
 for(int seed:{1,17,99}){projection<576,576,192,2>(seed);projection<192,576,192,1>(seed);projection<576,576,128,2>(seed);projection<1536,576,0,4>(seed);projection<576,1536,320,2>(seed);}
 auto*q=alloc<__nv_bfloat16>(576);auto*k=alloc<__nv_bfloat16>(256*3*16*64);auto*v=alloc<__nv_bfloat16>(256*3*16*64);auto*a=alloc<__nv_bfloat16>(576);auto*b=alloc<__nv_bfloat16>(576);auto*pages=alloc<uint32_t>(256);auto*shape=alloc<uint32_t>(4);auto*scratch=alloc<float>(9*4096);
 for(int i=0;i<576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);
 for(int i=0;i<256*3*16*64;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}
 for(int i=0;i<256;++i)pages[i]=(i*17+3)%256;
 for(int n:{1,16,17,127,128,129,160,525,1024,2049,4096}){
  shape[1]=n-1;riley_prefill_shape::attention_shape<<<dim3(1,3),96>>>(q,k,v,a,1,n,nullptr,pages);
  riley_decode_shape::enqueue(0,q,k,v,b,scratch,shape,pages,256);OK(cudaDeviceSynchronize());compare(a,b,576);
  printf("attention context=%d exact=true\n",n);
 }
 for(void*p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)pages,(void*)shape,(void*)scratch})OK(cudaFree(p));
 printf("projection_cases=15 attention_cases=11 exact=true\n");
}
