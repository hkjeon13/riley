#define gemm_prefill_shape_vector reference_projection
#define launch_prefill_shape reference_launch
#include "prefill_projection_v26_reference.cuh"
#undef gemm_prefill_shape_vector
#undef launch_prefill_shape
#include "prefill_shape_projection.cuh"
#include <cstdio>
#include <cstdlib>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line%d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<int N,int K,int I,int W,bool Tiled>void check(){
 auto*x=alloc<__nv_bfloat16>(1024*K);auto*w=alloc<__nv_bfloat16>(N*K);auto*tw=alloc<__nv_bfloat16>(N*K);auto*a=alloc<__nv_bfloat16>(1024*N+16);auto*b=alloc<__nv_bfloat16>(1024*N+16);auto*live=alloc<uint32_t>(1);
 for(int i=0;i<1024*K;++i)x[i]=__float2bfloat16_rn(float((i*17)%127-63)/128);for(int i=0;i<N*K;++i)w[i]=__float2bfloat16_rn(float((i*13)%251-125)/256);
 for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)tw[((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z]=w[(n+lane/4)*K+k+hi*8+2*(lane%4)+z];
 for(int rows:{0,1,7,15,16,17,73,128,398,1024}){
 *live=rows;for(int i=0;i<1024*N+16;++i)a[i]=b[i]=__ushort_as_bfloat16(0x4123);
 reference_projection<N,K,I,W><<<dim3(N/(8*W),64),32*W>>>(x,w,a,1024,live);
 gemm_prefill_shape_vector<N,K,I,W,Tiled><<<dim3(N/(8*W),64),32*W>>>(x,Tiled?tw:w,b,1024,live);OK(cudaDeviceSynchronize());
 for(int i=0;i<1024*N+16;++i)if(__bfloat16_as_ushort(a[i])!=__bfloat16_as_ushort(b[i])){fprintf(stderr,"mismatch N%d K%d rows%d i%d\n",N,K,rows,i);exit(3);}
 }
 printf("prefill N%d K%d packed%d rows0..1024 exact=true guards=true\n",N,K,Tiled);for(void*p:{(void*)x,(void*)w,(void*)tw,(void*)a,(void*)b,(void*)live})OK(cudaFree(p));
}
int main(){check<576,576,192,2,false>();check<192,576,192,1,false>();check<576,576,128,2,false>();check<1536,576,0,4,true>();check<576,1536,320,2,true>();}
