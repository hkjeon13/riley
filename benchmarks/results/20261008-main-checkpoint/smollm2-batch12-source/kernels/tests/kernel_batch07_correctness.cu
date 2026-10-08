// Batch07: preserved Batch05 oracle, expanded paired-query boundaries; exact bytes only.
#define main kernel_batch03_matrix_main
#include "kernel_batch03_correctness.cu"
#undef main
#include "prefill_shape_pointwise.cuh"
#include "decode_merge_norm_v56.cuh"
int fusion_case(int seed,int mode) {
 constexpr size_t N=32*576,Guard=16;
 auto* parts=alloc<float>(5*N+2*Guard);
 auto* merged=alloc<__nv_bfloat16>(N+2*Guard);
 auto* skip_bf=alloc<__nv_bfloat16>(N);
 auto* skip_fp=alloc<float>(N);
 auto* weight=alloc<__nv_bfloat16>(576);
 auto* a=alloc<__nv_bfloat16>(N+2*Guard);auto* b=alloc<__nv_bfloat16>(N+2*Guard);
 auto* ra=alloc<float>(N+2*Guard);auto* rb=alloc<float>(N+2*Guard);
 auto* ba=alloc<__nv_bfloat16>(N+2*Guard);auto* bb=alloc<__nv_bfloat16>(N+2*Guard);
 auto* shape=alloc<uint32_t>(3);shape[0]=shape[1]=0;
 fill_fp32(parts,5*N+2*Guard);
 for(size_t i=0;i<5*N;++i) {
  float x=seed==0?0.F:float((i*13+seed)%251)/257.F;
  if(seed==99)x=(i/N==0?65536.F:i/N==1?-65536.F:x);
  parts[Guard+i]=x;
 }
 for(size_t i=0;i<N;++i){skip_fp[i]=float(int((i*17+seed)%127)-63)/128;skip_bf[i]=__float2bfloat16_rn(skip_fp[i]);}
 for(size_t i=0;i<576;++i)weight[i]=__float2bfloat16_rn(float((i+seed)%17+1)/16);
 for(int rows=0;rows<=33;++rows) {
  shape[2]=rows;fill_bf16(merged,N+2*Guard);fill_bf16(a,N+2*Guard);fill_bf16(b,N+2*Guard);
  fill_fp32(ra,N+2*Guard);fill_fp32(rb,N+2*Guard);fill_bf16(ba,N+2*Guard);fill_bf16(bb,N+2*Guard);
  const void* skip=mode==1?static_cast<void*>(skip_bf):static_cast<void*>(skip_fp);
  void* ar=mode==1?static_cast<void*>(ra+Guard):static_cast<void*>(ba+Guard);
  void* br=mode==1?static_cast<void*>(rb+Guard):static_cast<void*>(bb+Guard);
  v52_reference::shared32_projection_merge_v52<576,576,128><<<72,256>>>(parts+Guard,merged+Guard,shape+2);
  riley_prefill_pointwise::norm_rows<<<32,256>>>(merged+Guard,skip,weight,ar,a+Guard,mode,shape,32);
  riley_merge_norm_v56::merge_norm<<<32,256>>>(parts+Guard,skip,weight,br,b+Guard,mode,shape,32);
  OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,N+2*Guard,"fused normalized BF16 including guards/inactive",rows);
  if(mode==1)compare(ra,rb,N+2*Guard,"fused FP32 residual including guards/inactive",rows);
  else compare(ba,bb,N+2*Guard,"fused BF16 residual including guards/inactive",rows);
 }
 for(void* p:{(void*)parts,(void*)merged,(void*)skip_bf,(void*)skip_fp,(void*)weight,(void*)a,(void*)b,(void*)ra,(void*)rb,(void*)ba,(void*)bb,(void*)shape})OK(cudaFree(p));
 return 34;
}

#include "prefill_fused_gate_v51.cuh"
#define riley_prefill51 riley_prefill51_reference
#include "reference/prefill_fused_gate_v51_reference.cuh"
#undef riley_prefill51
int prefill_gate_case(int seed){
 constexpr uint32_t Capacity=512;constexpr size_t Output=Capacity*1536;
 auto* x=alloc<__nv_bfloat16>(Capacity*576);auto* g=alloc<__nv_bfloat16>(1536*576);auto* u=alloc<__nv_bfloat16>(1536*576);
 auto* gt=alloc<__nv_bfloat16>(1536*576);auto* ut=alloc<__nv_bfloat16>(1536*576);
 auto* a=alloc<__nv_bfloat16>(Output+32);auto* b=alloc<__nv_bfloat16>(Output+32);auto* live=alloc<uint32_t>(1);
 for(size_t i=0;i<Capacity*576;++i)x[i]=__float2bfloat16_rn(float((int(i)*17+seed)%127-63)/128);
 weights<1536,576>(g,gt,seed);weights<1536,576>(u,ut,seed+11);
 int count=0;
 for(uint32_t rows:{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,33,127,128,129,255,256,257,511,512,513}){
  *live=rows;fill_bf16(a,Output+32);fill_bf16(b,Output+32);
  riley_prefill51_reference::gate_up<4><<<dim3(48,Capacity/16),128>>>(x,gt,ut,a+16,Capacity,live);
  riley_prefill51::gate_up<4><<<dim3(48,Capacity/16),128>>>(x,gt,ut,b+16,Capacity,live);
  OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,Output+32,"prefill pipelined gate BF16 including guards/inactive",rows);
  for(int i=0;i<16;++i)if(__bfloat16_as_ushort(b[i])!=0x4123||__bfloat16_as_ushort(b[Output+16+i])!=0x4123)std::exit(4);
  for(size_t i=(rows>=1&&rows<=Capacity?rows*1536:0);i<Output;++i)if(__bfloat16_as_ushort(b[16+i])!=0x4123)std::exit(4);
  ++count;
 }
 for(void* p:{(void*)x,(void*)g,(void*)u,(void*)gt,(void*)ut,(void*)a,(void*)b,(void*)live})OK(cudaFree(p));
 std::printf("prefill gate seed=%d rows0..512/tails/invalid513 exact\n",seed);return count;
}

#include "decode_gqa_attention_v50.cuh"
#include "mixed_attention_v49.cuh"
#define riley_gqa50_attention riley_gqa50_attention_batch05_reference
#include "reference/decode_gqa_attention_v50_batch05_reference.cuh"
#undef riley_gqa50_attention
#define riley_mixed_attention riley_mixed_attention_batch05_reference
#include "reference/mixed_attention_v49_batch05_reference.cuh"
#undef riley_mixed_attention

int decode_values_cases(int seed){
 constexpr int N=32*576,KV=256*3*16*64,Guard=16;
 auto* scores=alloc<float>(32*9*4096);auto* v=alloc<__nv_bfloat16>(KV);
 auto* a=alloc<__nv_bfloat16>(N+2*Guard);auto* b=alloc<__nv_bfloat16>(N+2*Guard);
 auto* shape=alloc<uint32_t>(32*416);auto* pages=alloc<uint32_t>(32*416);auto* live=alloc<uint32_t>(1);
 for(int i=0;i<32*9*4096;++i)scores[i]=float((i*7+seed)%131-65)/16;
 for(int i=0;i<KV;++i)v[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/128);
 int cases=0;
 for(int count:{1,15,16,17,63,64,65,127,128,129,255,256,257,4095,4096}){
  for(int r=0;r<32;++r){shape[r*416+1]=count-1;for(int p=0;p<256;++p)pages[r*416+p]=(p*5+13+r)%256;}
  for(int rows:{0,1,2,4,8,16,32,33}){
   *live=rows;fill_bf16(a,N+2*Guard);fill_bf16(b,N+2*Guard);
   riley_gqa50_attention_batch05_reference::independent_values<<<dim3(8,96),96>>>(scores,v,a+Guard,shape,pages,live);
   riley_gqa50_attention::independent_values<<<dim3(8,96),96>>>(scores,v,b+Guard,shape,pages,live);
   OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,N+2*Guard,"decode AV guards/inactive/tails/paged exact",rows);
   for(int i=0;i<Guard;++i)if(__bfloat16_as_ushort(b[i])!=0x4123||__bfloat16_as_ushort(b[Guard+N+i])!=0x4123)std::exit(4);
   for(int i=(rows>=1&&rows<=32?rows*576:0);i<N;++i)if(__bfloat16_as_ushort(b[Guard+i])!=0x4123)std::exit(4);
   ++cases;
  }
 }
 for(void* p:{(void*)scores,(void*)v,(void*)a,(void*)b,(void*)shape,(void*)pages,(void*)live})OK(cudaFree(p));return cases;
}
int mapped_values_cases(int seed){
 constexpr int Capacity=256,N=Capacity*576,KV=256*3*16*64,Guard=16,Meta=32+32*416+1024+1024;
 auto* q=alloc<__nv_bfloat16>(N);auto* k=alloc<__nv_bfloat16>(KV);auto* v=alloc<__nv_bfloat16>(KV);
 auto* a=alloc<__nv_bfloat16>(N+2*Guard);auto* b=alloc<__nv_bfloat16>(N+2*Guard);auto* meta=alloc<uint32_t>(Meta);
 for(int i=0;i<N;++i)q[i]=__float2bfloat16_rn(float((i*17+seed)%127-63)/128);
 for(int i=0;i<KV;++i){k[i]=__float2bfloat16_rn(float((i*7+seed)%131-65)/128);v[i]=__float2bfloat16_rn(float((i*13+seed)%251-125)/128);}
 int cases=0;
 for(int count:{1,17,31,32,33,47,48,49,63,64,65,79,80,81,95,96,97,111,112,113,127,128})for(int owners:{1,2})for(int prefix:{0,127,257,3968})for(int exceptional:{0,1,2}){
  std::memset(meta,0,Meta*4);int total=count*owners,tiles=0;
  meta[5]=owners;meta[9]=total;
  for(int owner=0;owner<owners;++owner){auto* shape=meta+32+owner*416;shape[1]=prefix+count-1;shape[2]=count;shape[16]=owner*count;for(int p=0;p<256;++p)shape[32+p]=(p*5+13+owner)%256;
   int nt=count<32?count:(count+7)/8;for(int t=0;t<nt;++t)meta[32+32*416+1024+tiles++]=(owner<<16)|t;
  }
  meta[24]=tiles;const int changed=((13*3)*16)*64+64;auto saved=v[changed];if(exceptional)v[changed]=__ushort_as_bfloat16(exceptional==1?0x7fc1:0x7f80);
  fill_bf16(a,N+2*Guard);fill_bf16(b,N+2*Guard);
  riley_mixed_attention_batch05_reference::mapped_attention<<<dim3(tiles+1,9),32>>>(q,k,v,a+Guard,Capacity,meta);
  riley_mixed_attention::mapped_attention<<<dim3(tiles+1,9),32>>>(q,k,v,b+Guard,Capacity,meta);
  OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,N+2*Guard,"mapped AV owners/causal/nonfinite/paged exact",count);
  for(int i=0;i<Guard;++i)if(__bfloat16_as_ushort(b[i])!=0x4123||__bfloat16_as_ushort(b[Guard+N+i])!=0x4123)std::exit(4);
  for(int i=total*576;i<N;++i)if(__bfloat16_as_ushort(b[Guard+i])!=0x4123)std::exit(4);
  v[changed]=saved;++cases;
 }
 // Invalid active/total/tile count must leave all output unchanged.
 for(int invalid:{0,1,2}){std::memset(meta,0,Meta*4);meta[5]=invalid==0?33:1;meta[9]=invalid==1?Capacity+1:32;meta[24]=invalid==2?33:4;fill_bf16(a,N+2*Guard);fill_bf16(b,N+2*Guard);
  riley_mixed_attention_batch05_reference::mapped_attention<<<dim3(4,9),32>>>(q,k,v,a+Guard,Capacity,meta);riley_mixed_attention::mapped_attention<<<dim3(4,9),32>>>(q,k,v,b+Guard,Capacity,meta);OK(cudaGetLastError());OK(cudaDeviceSynchronize());compare(a,b,N+2*Guard,"invalid metadata exact",invalid);for(int i=0;i<N+2*Guard;++i)if(__bfloat16_as_ushort(b[i])!=0x4123)std::exit(4);++cases;
 }
 for(void* p:{(void*)q,(void*)k,(void*)v,(void*)a,(void*)b,(void*)meta})OK(cudaFree(p));return cases;
}
int main(){kernel_batch03_matrix_main();for(int seed:{0,1,17,99})for(int mode:{1,2})fusion_case(seed,mode);for(int seed:{0,1,17,99})prefill_gate_case(seed);int decode=0,mapped=0;for(int seed:{1,17,99}){decode+=decode_values_cases(seed);mapped+=mapped_values_cases(seed);}OK(cudaDeviceSynchronize());std::printf("{\"passed\":true,\"cases\":%d,\"decode_value_cases\":%d,\"mapped_value_cases\":%d,\"BF16_and_FP32_exact\":true,\"performance_claim_eligible\":false}\n",1208+decode+mapped,decode,mapped);return 0;}
