#include "prefill_shape_projection.cuh"
#include "decode_projection_probe_v13.cuh"
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

template<int N,int K,int I,int Warps,bool Packed>void bench(){
 auto*x=alloc<__nv_bfloat16>(K);auto*w=alloc<__nv_bfloat16>(N*K);auto*a=alloc<__nv_bfloat16>(N);auto*b=alloc<__nv_bfloat16>(N);auto*p=alloc<float>(9*4096);
 for(int i=0;i<K;++i)x[i]=__float2bfloat16_rn(float((i*17+1)%127-63)/128);
 for(int i=0;i<N*K;++i)w[i]=__float2bfloat16_rn(float((i*13+1)%251-125)/256);
 enqueue_decode_projection<N,K,I>(0,x,w,a,p);enqueue_probe_projection<N,K,I,Warps,Packed>(0,x,w,b,p);OK(cudaDeviceSynchronize());compare(a,b,N);
 cudaEvent_t start,end;OK(cudaEventCreate(&start));OK(cudaEventCreate(&end));
 for(int j=0;j<50;++j)enqueue_probe_projection<N,K,I,Warps,Packed>(0,x,w,b,p);
 OK(cudaEventRecord(start));for(int j=0;j<500;++j)enqueue_probe_projection<N,K,I,Warps,Packed>(0,x,w,b,p);OK(cudaEventRecord(end));OK(cudaEventSynchronize(end));float ms;OK(cudaEventElapsedTime(&ms,start,end));
 printf("N=%d K=%d I=%d warps=%d packed=%d us=%f exact=true\n",N,K,I,Warps,Packed,ms*2);
 OK(cudaEventDestroy(start));OK(cudaEventDestroy(end));for(void*z:{(void*)x,(void*)w,(void*)a,(void*)b,(void*)p})OK(cudaFree(z));
}
template<int W,bool P>void shapes(){bench<1536,576,0,W,P>();bench<576,1536,320,W,P>();bench<192,576,192,W,P>();}
int main(){shapes<1,false>();shapes<2,false>();shapes<4,false>();shapes<8,false>();shapes<1,true>();shapes<2,true>();shapes<4,true>();shapes<8,true>();}
