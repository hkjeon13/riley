from pathlib import Path
p=Path('/tmp/prefill_query_tile_v44.cuh');s=p.read_text();old=Path('/tmp/prefill_attention_v42_source.cuh').read_text()
b=old[old.index(' int lane=threadIdx.x%32'):old.index('\ninline cudaError_t launch')]
b=b.replace(' int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,t=lane%4;\n int row=blockIdx.x,kvh=blockIdx.y,qh=kvh*3+warp,count=n-rows+row+1;',' int lane=threadIdx.x%32,warp=0,group=lane/4,t=lane%4;\n int kvh=qh/3,count=n-rows+row+1;')
for line in [' __shared__ float scores[3][128];\n',' __shared__ __nv_bfloat16 probs[3][128];\n',' __shared__ float exponentials[3][128];\n']:b=b.replace(line,'')
f='''// Exceptional nonfinite V reuses the original per-query arithmetic and causal mask.
__device__ void single_query(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,int rows,int n,const uint32_t* blocks,int row,int qh,float (*scores)[128],__nv_bfloat16 (*probs)[128],float (*exponentials)[128]){
'''+b
s=s.replace('template<int TileRows>',f+'\ntemplate<int TileRows>',1)
s=s.replace('   #pragma unroll\n   for(int b=0;b<8;++b){\n    const int dim=', '   bool nonfinite_value=false;\n   #pragma unroll\n   for(int b=0;b<8;++b){\n    const int dim=',1)
s=s.replace('    float next[4];','''    nonfinite_value=nonfinite_value||((vb&0x7f80U)==0x7f80U)||((vb&0x7f800000U)==0x7f800000U)||((vhi&0x7f80U)==0x7f80U)||((vhi&0x7f800000U)==0x7f800000U);
    float next[4];''',1)
old='''   }
  }
  __syncwarp();
 }
 #pragma unroll
 for(int h=0;h<2;++h){'''
new='''   }
   if(__any_sync(0xffffffff,nonfinite_value)){
    __syncwarp();
    for(int row=first;row<min(first+TileRows,rows);++row){
     single_query(q,k,v,out,rows,n,blocks,row,qh,scores,probs,exps);
     __syncwarp();
    }
    return;
   }
  }
  __syncwarp();
 }
 #pragma unroll
 for(int h=0;h<2;++h){'''
assert old in s;s=s.replace(old,new,1);p.write_text(s)
