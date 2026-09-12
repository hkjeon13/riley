#include "decode_shared.cuh"
#include "decode_shared_attention.cuh"
#include <cstdio>
#include <cstdlib>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
template<int N,int K,int I,bool Tiled>void check(){
 auto*x=alloc<__nv_bfloat16>(8*K);auto*w=alloc<__nv_bfloat16>(N*K);auto*tw=alloc<__nv_bfloat16>(N*K);auto*a=alloc<__nv_bfloat16>(8*N);auto*b=alloc<__nv_bfloat16>(8*N+16);auto*p=alloc<float>(8*8*N);auto*live=alloc<uint32_t>(1);
 for(int seed:{1,17,99}){
 for(int i=0;i<8*K;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 for(int i=0;i<N*K;++i)w[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);
 for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z)
 tw[((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z]=w[(n+lane/4)*K+k+hi*8+2*(lane%4)+z];
 for(int rows=0;rows<=9;++rows){
 *live=rows;for(int i=0;i<8*N+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
 if(rows>=1&&rows<=8)for(int row=0;row<rows;++row){
 if constexpr(Tiled)enqueue_tile_projection<N,K,I>(0,x+row*K,tw,a+row*N,p);
 else enqueue_decode_projection<N,K,I>(0,x+row*K,w,a+row*N,p);
 }
 enqueue_shared_projection<N,K,I,Tiled>(0,x,Tiled?tw:w,b,p,live);OK(cudaDeviceSynchronize());
 int valid=rows>=1&&rows<=8?rows*N:0;
 for(int i=0;i<8*N+16;++i){unsigned expected=i<valid?__bfloat16_as_ushort(a[i]):0x4123;
 if(__bfloat16_as_ushort(b[i])!=expected){fprintf(stderr,"mismatch N=%d K=%d I=%d seed=%d rows=%d i=%d\n",N,K,I,seed,rows,i);exit(3);}}
 }
 }
 printf("N=%d K=%d I=%d tiled=%d seeds=3 active_rows=1..8 invalid_rows=0,9 exact=true inactive_guard=true\n",N,K,I,Tiled);
 for(void*z:{(void*)x,(void*)w,(void*)tw,(void*)a,(void*)b,(void*)p,(void*)live})OK(cudaFree(z));
}

void attention(){
 auto*q=alloc<__nv_bfloat16>(8*576);auto*k=alloc<__nv_bfloat16>(2048*16*192);auto*v=alloc<__nv_bfloat16>(2048*16*192);auto*a=alloc<__nv_bfloat16>(8*576);auto*b=alloc<__nv_bfloat16>(8*576+16);auto*scratch=alloc<float>(8*9*4096);auto*shape=alloc<uint32_t>(8*416);auto*live=alloc<uint32_t>(1);
 int counts[8]={1,16,17,127,128,129,2049,4096};
 for(int i=0;i<8*576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);
 for(int i=0;i<2048*16*192;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}
 for(int row=0;row<8;++row){shape[row*416+1]=counts[row]-1;for(int page=0;page<256;++page)shape[row*416+32+page]=row*256+(page*17+3)%256;}
 for(int rows=0;rows<=9;++rows){
  *live=rows;for(int i=0;i<8*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
  if(rows>=1&&rows<=8)for(int row=0;row<rows;++row)riley_decode_shape::enqueue(0,q+row*576,k,v,a+row*576,scratch,shape+row*416,shape+row*416+32,2048);
  riley_shared_attention::enqueue(0,q,k,v,b,scratch,shape,shape+32,live,4096);OK(cudaDeviceSynchronize());
  int valid=rows>=1&&rows<=8?rows*576:0;
  for(int i=0;i<8*576+16;++i)if(__bfloat16_as_ushort(b[i])!=(i<valid?__bfloat16_as_ushort(a[i]):0x4123)){fprintf(stderr,"attention mismatch rows=%d i=%d\n",rows,i);exit(4);}
 }
 printf("attention active_rows=1..8 contexts=1,16,17,127,128,129,2049,4096 disjoint_permuted_pages=true exact=true inactive_guard=true\n");
 for(void*z:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)scratch,(void*)shape,(void*)live})OK(cudaFree(z));
}
int main(){check<576,576,192,false>();check<192,576,192,false>();check<576,576,128,false>();check<1536,576,0,true>();check<576,1536,320,true>();attention();}
