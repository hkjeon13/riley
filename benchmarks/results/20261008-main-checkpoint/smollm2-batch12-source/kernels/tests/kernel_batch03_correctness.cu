// Exact comparison against the preserved V52 operators; no timing claim.
#include "decode_shared32.cuh"
// Distinct symbols prevent CUDA argument-dependent lookup from considering
// the global candidate overloads when compiling the unchanged V52 body.
#define shared32_projection_compute shared32_projection_compute_v52
#define shared32_projection_parts shared32_projection_parts_v52
#define shared32_gate_up shared32_gate_up_v52
#define shared32_qkv_parts shared32_qkv_parts_v52
#define shared32_qkv_merge shared32_qkv_merge_v52
#define enqueue_shared32_qkv enqueue_shared32_qkv_v52
#define shared32_projection_merge shared32_projection_merge_v52
#define enqueue_shared32_projection enqueue_shared32_projection_v52
#define shared32_gate_up_swiglu shared32_gate_up_swiglu_v52
namespace v52_reference {
#include "reference/decode_shared32_v52.cuh"
}
#undef shared32_projection_compute
#undef shared32_projection_parts
#undef shared32_gate_up
#undef shared32_qkv_parts
#undef shared32_qkv_merge
#undef enqueue_shared32_qkv
#undef shared32_projection_merge
#undef enqueue_shared32_projection
#undef shared32_gate_up_swiglu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define OK(call) do { auto e=(call); if(e!=cudaSuccess) { \
 std::fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); std::exit(2); } } while(0)
template<class T> T* alloc(size_t n) { T* p; OK(cudaMallocManaged(&p,n*sizeof(T))); return p; }
template<class T> void compare(const T* a,const T* b,size_t n,const char* label,int rows) {
 for(size_t i=0;i<n;++i) if(std::memcmp(a+i,b+i,sizeof(T))) {
  std::fprintf(stderr,"%s rows=%d mismatch=%zu\n",label,rows,i); std::exit(3);
 }
}
void fill_bf16(__nv_bfloat16* p,size_t n) { for(size_t i=0;i<n;++i)p[i]=__ushort_as_bfloat16(0x4123); }
void fill_fp32(float* p,size_t n) { const uint32_t bits=0x7f123456; for(size_t i=0;i<n;++i)std::memcpy(p+i,&bits,4); }
template<int N,int K> void weights(__nv_bfloat16* linear,__nv_bfloat16* packed,int seed) {
 for(int i=0;i<N*K;++i)linear[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/256);
 for(int n=0;n<N;n+=8)for(int k=0;k<K;k+=16)for(int lane=0;lane<32;++lane)
  for(int hi=0;hi<2;++hi)for(int z=0;z<2;++z) {
   int dst=((n/8)*(K/16)+k/16)*128+hi*64+lane*2+z;
   int src=(n+lane/4)*K+k+hi*8+2*(lane%4)+z; packed[dst]=linear[src];
  }
}
template<int N,int K,int Interval,bool Tiled> int projection(int seed) {
 constexpr int Chunks=Interval>0?(K+Interval-1)/Interval:1;
 constexpr size_t Output=32*N,Parts=Chunks*32*N;
 auto* x=alloc<__nv_bfloat16>(32*K);auto* linear=alloc<__nv_bfloat16>(N*K);auto* packed=alloc<__nv_bfloat16>(N*K);
 auto* a=alloc<__nv_bfloat16>(Output+32);auto* b=alloc<__nv_bfloat16>(Output+32);
 auto* pa=alloc<float>(Parts+32);auto* pb=alloc<float>(Parts+32);auto* live=alloc<uint32_t>(1);
 for(int i=0;i<32*K;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 weights<N,K>(linear,packed,seed);auto* w=Tiled?packed:linear;
 for(int rows=0;rows<=33;++rows) {
  *live=rows;fill_bf16(a,Output+32);fill_bf16(b,Output+32);fill_fp32(pa,Parts+32);fill_fp32(pb,Parts+32);
  v52_reference::enqueue_shared32_projection_v52<N,K,Interval,Tiled>(0,x,w,a+16,pa+16,live);
  enqueue_shared32_projection<N,K,Interval,Tiled>(0,x,w,b+16,pb+16,live);
  OK(cudaGetLastError());OK(cudaDeviceSynchronize());
  compare(a,b,Output+32,"projection output including guards/inactive",rows);
  compare(pa,pb,Parts+32,"FP32 partials including guards/inactive",rows);
  for(int i=0;i<16;++i)if(__bfloat16_as_ushort(b[i])!=0x4123||__bfloat16_as_ushort(b[Output+16+i])!=0x4123)std::exit(4);
  for(size_t i=(rows>=1&&rows<=32?rows*N:0);i<Output;++i)if(__bfloat16_as_ushort(b[16+i])!=0x4123)std::exit(4);
  if(rows>=1&&rows<=32) { bool written=false;for(int i=0;i<rows*N;++i)written|=__bfloat16_as_ushort(a[16+i])!=0x4123;if(!written)std::exit(5); }
 }
 for(void* p:{(void*)x,(void*)linear,(void*)packed,(void*)a,(void*)b,(void*)pa,(void*)pb,(void*)live})OK(cudaFree(p));
 std::printf("projection N=%d K=%d interval=%d tiled=%d seed=%d active0..33 exact\n",N,K,Interval,Tiled,seed);return 34;
}
template<bool Tiled> int gate(int seed) {
 constexpr size_t Output=32*1536;
 auto* x=alloc<__nv_bfloat16>(32*576);auto* g=alloc<__nv_bfloat16>(1536*576);auto* u=alloc<__nv_bfloat16>(1536*576);
 auto* gt=alloc<__nv_bfloat16>(1536*576);auto* ut=alloc<__nv_bfloat16>(1536*576);
 auto* a=alloc<__nv_bfloat16>(Output+32);auto* b=alloc<__nv_bfloat16>(Output+32);auto* live=alloc<uint32_t>(1);
 for(int i=0;i<32*576;++i)x[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 weights<1536,576>(g,gt,seed);weights<1536,576>(u,ut,seed+11);
 for(int rows=0;rows<=33;++rows) {
  *live=rows;fill_bf16(a,Output+32);fill_bf16(b,Output+32);
  v52_reference::shared32_gate_up_swiglu_v52<Tiled><<<192,64>>>(x,Tiled?gt:g,Tiled?ut:u,a+16,live);
  shared32_gate_up_swiglu<Tiled><<<192,64>>>(x,Tiled?gt:g,Tiled?ut:u,b+16,live);
  OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,Output+32,"gate BF16 output including guards/inactive",rows);
  for(int i=0;i<16;++i)if(__bfloat16_as_ushort(b[i])!=0x4123||__bfloat16_as_ushort(b[Output+16+i])!=0x4123)std::exit(4);
  for(size_t i=(rows>=1&&rows<=32?rows*1536:0);i<Output;++i)if(__bfloat16_as_ushort(b[16+i])!=0x4123)std::exit(4);
  if(rows>=1&&rows<=32) { bool written=false;for(int i=0;i<rows*1536;++i)written|=__bfloat16_as_ushort(a[16+i])!=0x4123;if(!written)std::exit(5); }
 }
 for(void* p:{(void*)x,(void*)g,(void*)u,(void*)gt,(void*)ut,(void*)a,(void*)b,(void*)live})OK(cudaFree(p));
 std::printf("gate tiled=%d seed=%d active0..33 exact\n",Tiled,seed);return 34;
}
int main() {
 int cases=0;
 for(int seed:{1,17,99}) {
  // Actual V52 graph routes: down K1536/chunk320 (last chunk256),
  // output K576/chunk128 (last chunk64), including untiled down fallback.
  cases+=projection<576,1536,320,true>(seed);cases+=projection<576,1536,320,false>(seed);
  cases+=projection<576,576,128,false>(seed);
  // Additional direct and divisible-chunk routes exercise the shared helper.
  cases+=projection<576,1536,512,true>(seed);cases+=projection<576,576,0,true>(seed);
  cases+=projection<192,576,192,false>(seed);
  cases+=gate<true>(seed);cases+=gate<false>(seed);
 }
 OK(cudaDeviceSynchronize());std::printf("{\"passed\":true,\"cases\":%d,\"seeds\":3,\"active_rows\":\"0..33\",\"performance_claim_eligible\":false}\n",cases); return 0;
}
