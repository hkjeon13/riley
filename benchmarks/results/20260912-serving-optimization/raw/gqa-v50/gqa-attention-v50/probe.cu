#include "gqa_attention_v50.cuh"
#include "decode_shared32_attention.cuh"
#include <cstdio>
#include <cstdlib>
#define OK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"line %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;OK(cudaMallocManaged(&p,n*sizeof(T)));return p;}
void attention(){
 auto*q=alloc<__nv_bfloat16>(32*576);auto*k=alloc<__nv_bfloat16>(4096*32*192);auto*v=alloc<__nv_bfloat16>(4096*32*192);auto*a=alloc<__nv_bfloat16>(32*576);auto*b=alloc<__nv_bfloat16>(32*576+16);auto*scratch=alloc<float>(32*9*4096);auto*shape=alloc<uint32_t>(32*416);auto*live=alloc<uint32_t>(1);
 int counts[16]={1,16,17,127,128,129,2049,4096,2,15,73,255,256,398,1024,4095};
 for(int i=0;i<32*576;++i)q[i]=__float2bfloat16_rn(float(i*7%127-63)/128);
 for(int i=0;i<4096*32*192;++i){k[i]=__float2bfloat16_rn(float(i*17%251-125)/256);v[i]=__float2bfloat16_rn(float(i*13%127-63)/64);}
 for(int row=0;row<32;++row){shape[row*416+1]=counts[row%16]-1;for(int page=0;page<256;++page)shape[row*416+32+page]=row*256+(page*17+3)%256;}
 for(int rows=0;rows<=33;++rows){
  *live=rows;for(int i=0;i<32*576+16;++i)b[i]=__ushort_as_bfloat16(0x4123);
  if(rows>=1&&rows<=32)for(int row=0;row<rows;++row)riley_decode_shape::enqueue(0,q+row*576,k,v,a+row*576,scratch,shape+row*416,shape+row*416+32,4096);
  riley_gqa50_attention::enqueue(0,q,k,v,b,scratch,shape,shape+32,live,4096);OK(cudaDeviceSynchronize());
  int valid=rows>=1&&rows<=32?rows*576:0;
  for(int i=0;i<32*576+16;++i)if(__bfloat16_as_ushort(b[i])!=(i<valid?__bfloat16_as_ushort(a[i]):0x4123)){fprintf(stderr,"attention mismatch rows=%d i=%d\n",rows,i);exit(4);}
 }
 printf("attention active_rows=1..32 invalid_rows=0,33 contexts=1..4096 mixed disjoint_permuted_pages=true exact=true inactive_guard=true\n");
 for(void*z:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)scratch,(void*)shape,(void*)live})OK(cudaFree(z));
}
int main(){attention();}
