#include "decode_gqa_attention_v50.cuh"
extern "C" int riley_attention(void* stream, const void* q,const void* k,const void* v,void* out,void* scratch,const void* shape,const void* live,unsigned context) {
 riley_gqa50_attention::enqueue((cudaStream_t)stream,(const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,(float*)scratch,(const uint32_t*)shape,((const uint32_t*)shape)+32,(const uint32_t*)live,context);
 return (int)cudaGetLastError();
}
