// Retain all 816 projection/gate cases; compare fused residual and norm bytes
// directly against the preserved V52 two-kernel sequence, including inactive
// rows, invalid live counts, and both BF16/FP32 residual representations.
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
int main(){kernel_batch03_matrix_main();int cases=816;for(int seed:{0,1,17,99})for(int mode:{1,2})cases+=fusion_case(seed,mode);OK(cudaDeviceSynchronize());std::printf("{\"passed\":true,\"cases\":%d,\"fusion_cases\":272,\"BF16_and_FP32_exact\":true,\"performance_claim_eligible\":false}\n",cases);}
