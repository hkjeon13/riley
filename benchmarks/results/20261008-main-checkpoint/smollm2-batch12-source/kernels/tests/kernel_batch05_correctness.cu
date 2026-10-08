// Preserve the complete Batch04 byte-exact matrix, then qualify pipelined
// prefill gate/up against the byte-preserved pre-candidate V51 implementation.
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
int main(){
 kernel_batch03_matrix_main();int cases=816;
 for(int seed:{0,1,17,99})for(int mode:{1,2})cases+=fusion_case(seed,mode);
 int prefill_cases=0;for(int seed:{0,1,17,99})prefill_cases+=prefill_gate_case(seed);cases+=prefill_cases;
 OK(cudaDeviceSynchronize());std::printf("{\"passed\":true,\"cases\":%d,\"fusion_cases\":272,\"prefill_gate_cases\":%d,\"BF16_and_FP32_exact\":true,\"performance_claim_eligible\":false}\n",cases,prefill_cases);
}
