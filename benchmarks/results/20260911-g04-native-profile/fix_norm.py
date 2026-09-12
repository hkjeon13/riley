from pathlib import Path
r=Path('/tmp/riley-g04-native-profile-source-260911')
p=r/'kernels/src/graph_numerics_precise.cu';s=p.read_text();a=s.index('__global__ void compiled_norm');b=s.index('__global__ void compiled_rope')
s=s[:a]+'''__global__ void compiled_norm(const __nv_bfloat16* a,const void* b,const __nv_bfloat16* w,void* residual,__nv_bfloat16* out,int mode){
 __shared__ float sums[8];int tid=threadIdx.x,lane=tid%32;float x[4]={};
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576){x[j]=__bfloat162float(a[i]);
  if(mode==1)x[j]+=__bfloat162float(((const __nv_bfloat16*)b)[i]);if(mode==2)x[j]+=((const float*)b)[i];
  if(mode==1)((float*)residual)[i]=x[j];if(mode==2)((__nv_bfloat16*)residual)[i]=__float2bfloat16_rn(x[j]);}}
 float sum=x[1]*x[1];sum=fmaf(x[0],x[0],sum);sum=fmaf(x[2],x[2],sum);sum=fmaf(x[3],x[3],sum);
 for(int step=16;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);
 if(lane==0)sums[tid/32]=sum;__syncthreads();
 if(tid<32){sum=lane<8?sums[lane]:0.;for(int step=4;step;step>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,step);if(lane==0)sums[0]=sum;}__syncthreads();
 float mean,inv;asm("div.full.f32 %0, %1, %2;":"=f"(mean):"f"(sums[0]),"f"(576.F));mean=__fadd_rn(mean,1e-5F);
 asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inv):"f"(mean));
 for(int j=0;j<4;++j){int i=tid*4+j;if(i<576)out[i]=__float2bfloat16_rn((x[j]*inv)*__bfloat162float(w[i]));}
}

'''+s[b:];s=s.replace('compiled_norm<<<1,1024','compiled_norm<<<1,256');p.write_text(s)
p=r/'kernels/src/graph_resources.cu';s=p.read_text().replace('native-trace-tail.bin','native-trace-normfix.bin');p.write_text(s)
