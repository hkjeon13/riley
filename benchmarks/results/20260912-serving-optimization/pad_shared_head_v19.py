from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/kernels/src/decode_shared_model.cuh');s=p.read_text();a='inline cudaError_t enqueue(';b='''// Fixed-capacity GEMM also reads inactive rows; initialize those inputs.
__global__ void clear_inactive_hidden(__nv_bfloat16* hidden,const uint32_t* active){
 uint32_t rows=*active,row=blockIdx.x;if(rows<1||rows>8||row<rows)return;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)hidden[row*576+i]=__float2bfloat16_rn(0.F);
}
'''+a;assert a in s;s=s.replace(a,b).replace(' return cudaGetLastError();',' clear_inactive_hidden<<<8,256,0,stream>>>(b(1),active);\n return cudaGetLastError();');p.write_text(s)
