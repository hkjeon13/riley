from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/kernels/src')
s=(r/'decode_shape.cuh').read_text().split('namespace riley_decode_shape {')[1].split('inline void enqueue(')[0]
s=s.replace('const uint32_t* shape,const uint32_t* pages){','const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows){\n int row=blockIdx.y/9;uint32_t active=*live_rows;if(active<1||active>8||row>=active)return;\n shape+=row*416;pages+=row*416;')
s=s.replace('head=blockIdx.y','head=blockIdx.y%9')
s=s.replace(' int count=shape[1]+1,token=', ' q+=row*576;result+=row*9*4096;\n int count=shape[1]+1,token=')
s=s.replace(' int count=shape[1]+1,head=', ' scores+=row*9*4096;out+=row*576;\n int count=shape[1]+1,head=')
s='#pragma once\n#include "decode_shape.cuh"\nnamespace riley_shared_attention {\n'+s+'''
inline void enqueue(cudaStream_t stream,const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,float* scratch,const uint32_t* shape,const uint32_t* pages,const uint32_t* live_rows,uint32_t context){
 scores<<<dim3((context+7)/8,72),32,0,stream>>>(q,k,scratch,shape,pages,live_rows);
 values<<<dim3(8,72),32,0,stream>>>(scratch,v,out,shape,pages,live_rows);
}
}
'''
(r/'decode_shared_attention.cuh').write_text(s)
