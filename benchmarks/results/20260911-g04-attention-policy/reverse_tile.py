from pathlib import Path
p=Path('/tmp/riley-g04-attention-policy-source-260911/kernels/src/batch_primitives.cu')
s=p.read_text();a=s.index('void ragged_paged_attention_gqa_shared_kv_kernel(');start=s.index('  float maximum=-CUDART_INF_F;',a);end=s.index('  output[base+lane]=',start)
s=s[:start]+'''  float maximum=-CUDART_INF_F,denominator=0.0F,n0=0.0F,n1=0.0F;
  for(int64_t tile=(count-1)/128;tile>=0;--tile) {
    const uint64_t begin=tile*128,end=min(begin+128,count);
    float next_maximum=maximum;
    for(uint64_t t=begin;t<end;++t) {
      uint64_t kb=0;
      if(!resolve_row_cache_base(batch,row,t,kvh,key_value_head_count,64,&kb)) return;
      float score=fmaf(q0,__bfloat162float(key_pool[kb+lane]),q1*__bfloat162float(key_pool[kb+lane+32]));
      score=warp_sum(score);score=__shfl_sync(kFullWarpMask,score,0)*scale;
      next_maximum=fmaxf(next_maximum,score);
    }
    const float alpha=exp2f((maximum-next_maximum)*1.4426950408889634F);
    n0*=alpha;n1*=alpha;denominator*=alpha;maximum=next_maximum;
    for(uint64_t t=begin;t<end;++t) {
      uint64_t kb=0;
      if(!resolve_row_cache_base(batch,row,t,kvh,key_value_head_count,64,&kb)) return;
      float score=fmaf(q0,__bfloat162float(key_pool[kb+lane]),q1*__bfloat162float(key_pool[kb+lane+32]));
      score=warp_sum(score);score=__shfl_sync(kFullWarpMask,score,0)*scale;
      float p=exp2f((score-maximum)*1.4426950408889634F);
      denominator+=p;p=__bfloat162float(__float2bfloat16_rn(p));
      n0=fmaf(p,__bfloat162float(value_pool[kb+lane]),n0);
      n1=fmaf(p,__bfloat162float(value_pool[kb+lane+32]),n1);
    }
  }
'''+s[end:]
p.write_text(s)
