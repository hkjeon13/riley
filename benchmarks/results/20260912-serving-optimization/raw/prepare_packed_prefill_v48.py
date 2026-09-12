from pathlib import Path
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';out=r/'packed-prefill-v48';out.mkdir()
s=(src/'kernels/src/prefill_query_tile_attention.cuh').read_text().replace('namespace riley_prefill_query_tile','namespace riley_packed_prefill_attention').replace('__global__ void attention(','__device__ __forceinline__ void attention_body(')
pos=s.rfind('\n}')
s=s[:pos]+'''
// Each CTA belongs to exactly one owner's query/head tile. Keys and causal
// positions stay owner-local even though projection rows are densely packed.
__global__ void attention(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* out,uint32_t capacity,const uint32_t* meta){
 uint32_t owner=blockIdx.z,active=meta[5],total=meta[9];if(!active||active>4||owner>=active||!total||total>capacity)return;
 const uint32_t* shape=meta+32+owner*416;uint32_t offset=shape[16],count=shape[2];
 if(!count||offset>total||count>total-offset)return;
 attention_body<8>(q+offset*576,k,v,out+offset*576,capacity,0,reinterpret_cast<const int*>(shape+1),shape+32,shape+2);
}
'''+s[pos:];(out/'packed_prefill_attention_v48.cuh').write_text(s)
s=(src/'kernels/src/prefill_shape_rope_kv.cuh').read_text();a=s.index('__global__ void prefill_shape_rope_kv');b=s.index('\ncudaError_t launch_prefill_shape_rope_kv',a);s=s[:b];s=s.replace('prefill_shape_rope_kv','packed_prefill_rope_kv');a=s.index(' const uint32_t row=blockIdx.y');start=s.index(' const uint32_t* pages,uint32_t start');s=s[:start]+''' const uint32_t* meta,uint32_t capacity){
 const uint32_t row=blockIdx.y,active=meta[5],total=meta[9];if(!active||active>4||!total||total>capacity||row>=total)return;
 uint32_t owner=0;for(;owner<active;++owner){const auto* s=meta+32+owner*416;if(row>=s[16]&&row-s[16]<s[2])break;}if(owner==active)return;
 const auto* shape=meta+32+owner*416;const auto* pages=shape+32;const uint32_t start=shape[4]-shape[16];
'''+s[s.index(' int i=threadIdx.x',a):]
# start can wrap modulo u32 when offset exceeds committed; start+row still
# equals committed+(row-offset), but use the explicit expression for clarity.
s=s.replace('const uint32_t start=shape[4]-shape[16];','const uint32_t pos=shape[4]+row-shape[16];').replace(' const uint32_t pos=start+row;int head',' int head');(out/'packed_prefill_rope_v48.cuh').write_text(s)
s=(src/'kernels/src/prefill_shape_model.cuh').read_text();s=s.replace('#include "prefill_query_tile_attention.cuh"','#include "packed_prefill_attention_v48.cuh"\n#include "packed_prefill_rope_v48.cuh"');s=s.replace('enqueue_v3_prefill_model','enqueue_packed_prefill_model');s=s.replace(' auto shape=reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(metadata)+128);',' const auto* meta=static_cast<const uint32_t*>(metadata);\n auto shape=meta+7; // shape[2] is total packed token count, not one owner count.')
s=s.replace('status,reinterpret_cast<const uint32_t*>(metadata)+4);','status,nullptr);')
a=s.index('  prefill_shape_rope_kv<<<');b=s.index('  if(capacity==1)enqueue_decode_projection<576,576,128>',a)
s=s[:a]+'''  packed_prefill_rope_kv<<<dim3(2,capacity),256,0,stream>>>(b(2),b(5),b(6),b(3),lk,lv,static_cast<const float*>(cos),static_cast<const float*>(sin),meta,capacity);
  riley_packed_prefill_attention::attention<<<dim3(max((capacity+7)/8,min(capacity,31U)),9,4),32,0,stream>>>(b(3),lk,lv,b(4),capacity,meta);
'''+s[b:]
s=s.replace(' riley_prefill_pointwise::select_hidden<<<1,256,0,stream>>>(b(1),static_cast<__nv_bfloat16*>(selected),shape,capacity,status,publish);',' packed_prefill_select_hidden<<<32,256,0,stream>>>(b(1),static_cast<__nv_bfloat16*>(selected),meta,capacity,status,publish);')
pos=s.index('template<uint32_t WireRows=8>');s=s[:pos]+'''__global__ void packed_prefill_select_hidden(const __nv_bfloat16* rows,__nv_bfloat16* selected,const uint32_t* meta,uint32_t capacity,const uint32_t* status,uint32_t* publish){
 uint32_t owner=blockIdx.x,active=meta[5],total=meta[9];bool ready=false;uint32_t at=0;
 if(active>=1&&active<=4&&owner<active&&total>0&&total<=capacity){const auto* shape=meta+32+owner*416;uint32_t n=shape[2],offset=shape[16],row=shape[11];ready=*status==0&&n>0&&offset<=total&&n<=total-offset&&row<n;at=offset+row;}
 if(threadIdx.x==0)publish[owner]=ready?1:0;
 for(uint32_t i=threadIdx.x;i<576;i+=blockDim.x)selected[owner*576+i]=ready?rows[at*576+i]:__float2bfloat16_rn(0.F);
}
'''+s[pos:];(out/'packed_prefill_model_v48.cuh').write_text(s)
print('packed prefill prototype headers prepared')
