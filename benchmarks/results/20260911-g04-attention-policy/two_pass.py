from pathlib import Path
p=Path('/tmp/riley-g04-attention-policy-source-260911/kernels/src/batch_primitives.cu')
s=p.read_text();name='void ragged_paged_attention_gqa_shared_kv_kernel('
a=s.index(name);start=s.index('{',a);depth=1;end=start+1
while depth:
 if s[end]=='{':depth+=1
 if s[end]=='}':depth-=1
 end+=1
body='''{
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t warp = threadIdx.x / 32;
  const uint64_t row = blockIdx.x;
  const uint64_t kvh = blockIdx.y;
  const uint64_t qh = kvh * (blockDim.x / 32) + warp;
  const uint64_t base = (row * query_head_count + qh) * 64;
  if (row >= batch.active_row_count) {
    output[base+lane]=__float2bfloat16_rn(0.0F);
    output[base+lane+32]=__float2bfloat16_rn(0.0F); return;
  }
  const float q0=__bfloat162float(query[base+lane]);
  const float q1=__bfloat162float(query[base+lane+32]);
  const uint64_t count=static_cast<uint64_t>(batch.row_positions[row])+1;
  float maximum=-CUDART_INF_F;
  for(uint64_t t=0;t<count;++t) {
    uint64_t kb=0;
    if(!resolve_row_cache_base(batch,row,t,kvh,key_value_head_count,64,&kb)) return;
    float score=fmaf(q0,__bfloat162float(key_pool[kb+lane]),q1*__bfloat162float(key_pool[kb+lane+32]));
    score=warp_sum(score);score=__shfl_sync(kFullWarpMask,score,0)*scale;
    maximum=fmaxf(maximum,score);
  }
  float denominator=0.0F,n0=0.0F,n1=0.0F;
  for(uint64_t t=0;t<count;++t) {
    uint64_t kb=0;
    if(!resolve_row_cache_base(batch,row,t,kvh,key_value_head_count,64,&kb)) return;
    float score=fmaf(q0,__bfloat162float(key_pool[kb+lane]),q1*__bfloat162float(key_pool[kb+lane+32]));
    score=warp_sum(score);score=__shfl_sync(kFullWarpMask,score,0)*scale;
    float p=exp2f((score-maximum)*1.4426950408889634F);
    denominator+=p;
    p=__bfloat162float(__float2bfloat16_rn(p));
    n0=fmaf(p,__bfloat162float(value_pool[kb+lane]),n0);
    n1=fmaf(p,__bfloat162float(value_pool[kb+lane+32]),n1);
  }
  output[base+lane]=__float2bfloat16_rn(n0/denominator);
  output[base+lane+32]=__float2bfloat16_rn(n1/denominator);
}'''
p.write_text(s[:start]+body+s[end:])
