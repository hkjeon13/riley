#!/usr/bin/env python3
"""Derive one isolated F8 candidate; no build, GPU, or production edits."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

BASELINE_SHA = 'edb7990577187d9574bf6d7c751713581529e4da1b835d92ca4a012518de5552'
ORACLE_SHA = 'a1bb90862e9eb6bb378ac36843c1d94b05b59f078d4a1f4b9de8c2018a1f666e'
HERE = Path(__file__).resolve().parent
KERNEL = '__global__ void attention_packed_decode_rope(\n'
LOOP = ' for(int tile=(count-1)/128;tile>=0;--tile){\n'
END = '\n if(warp==0){\n  den+=__shfl_xor_sync'
LOADS = '''// All callers below prove four-byte alignment and preserve adjacent BF16 bits.
__device__ __forceinline__ uint32_t fused_history_shared_pair(const __nv_bfloat16* p){
 const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(p));
 uint32_t bits;asm volatile("ld.shared.b32 %0, [%1];" : "=r"(bits) : "r"(address) : "memory");return bits;
}
__device__ __forceinline__ uint32_t fused_history_global_pair(const __nv_bfloat16* p){
 const uint64_t address=static_cast<uint64_t>(__cvta_generic_to_global(p));
 uint32_t bits;asm volatile("ld.global.b32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");return bits;
}
'''
SIGNATURE = '''// Exactly two tiles for positions128..159: tail first, then full history.
// History specialization has no current-token test, raw-V bypass or tail mask.
template<bool History>
__device__ __forceinline__ void fused_history_tile(
 const __nv_bfloat16* local_q,const __nv_bfloat16* local_k,const __nv_bfloat16* raw_v,
 const __nv_bfloat16* k,const __nv_bfloat16* v,const uint32_t* blocks,uint32_t position,
 float* scores,__nv_bfloat16* probs,float* alpha_by_lane,
 int lane,int warp,int group,int t,int kvh,int first_block,
 float& maximum,float& den,float (&accum)[2][4]){
 const int begin=History?0:128,end=History?128:static_cast<int>(position)+1;
 const __nv_bfloat16 zero=__float2bfloat16_rn(0.);
'''
QK = '''  for(int token=begin+8*warp;token<end;token+=16){
   float d[4]={};
   const int key_token=token+group;
   int key_base=0;bool current=false;
   if constexpr(History){
    key_base=((blocks[token/16]*3+kvh)*16+token%16+group)*64;
   }else{
    current=key_token==static_cast<int>(position);
    if(key_token<end&&!current)key_base=cache_index(key_token,kvh,0,blocks);
   }
   for(int depth=0;depth<64;depth+=16){
    uint32_t a=fused_history_shared_pair(local_q+depth+2*t);
    uint32_t aa=fused_history_shared_pair(local_q+depth+2*t+8);
    uint32_t b=0,bb=0;
    if constexpr(History){
     b=fused_history_global_pair(k+key_base+depth+2*t);
     bb=fused_history_global_pair(k+key_base+depth+2*t+8);
    }else if(key_token<end){
     if(current){
      b=fused_history_shared_pair(local_k+depth+2*t);
      bb=fused_history_shared_pair(local_k+depth+2*t+8);
     }else{
      b=fused_history_global_pair(k+key_base+depth+2*t);
      bb=fused_history_global_pair(k+key_base+depth+2*t+8);
     }
    }
    mma(d,a,a,aa,aa,b,bb);
   }
   if(group==0){for(int j=0;j<2;++j)if(token+2*t+j<end)scores[token-begin+2*t+j]=d[j]*.125F;}
  }
'''
PV = '''  for(int block=0;block<2;++block){
   for(int j=0;j<4;++j)accum[block][j]*=alpha;
   const int dim=(first_block+block)*8+group;
   for(int token=begin;token<end;token+=16){
    int pi=token-begin;
    uint32_t a=fused_history_shared_pair(probs+pi+2*t);
    uint32_t aa=fused_history_shared_pair(probs+pi+2*t+8);
    uint32_t b,bb;
    if constexpr(History){
     // Every 16-token historical iteration stays within one physical page.
     const int page=((blocks[token/16]*3+kvh)*16)*64+dim;
     b=pair(v[page+(2*t)*64],v[page+(2*t+1)*64]);
     bb=pair(v[page+(2*t+8)*64],v[page+(2*t+9)*64]);
    }else{
     auto val=[&](int pos){
      if(pos>=end)return zero;
      return pos==static_cast<int>(position)?raw_v[kvh*64+dim]:v[cache_index(pos,kvh,dim,blocks)];
     };
     b=pair(val(token+2*t),val(token+2*t+1));
     bb=pair(val(token+2*t+8),val(token+2*t+9));
    }
    mma(accum[block],a,a,aa,aa,b,bb);
   }
  }
'''
CALLS = ''' // The validated range guarantees tail128..position followed by history0..127.
 fused_history_tile<false>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);
 fused_history_tile<true>(local_q,local_k,raw_v,k,v,blocks,position,scores,probs,alpha_by_lane,
     lane,warp,group,t,kvh,first_block,maximum,den,accum);
'''


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def transform(raw):
    require(digest(raw) == BASELINE_SHA, 'requires exact frozen F8 full TU')
    source=raw.decode();start=source.index(KERNEL);tail=source[start:]
    loop_start=tail.index(LOOP);loop_end=tail.index(END,loop_start)
    loop=tail[loop_start:loop_end]
    qk_start=loop.index('  for(int token=begin+8*warp;');qk_end=loop.index('  // Includes the idle',qk_start)
    pv_start=loop.index('  for(int block=0;block<2;++block){');pv_end=loop.index('  __syncthreads(); // Finish both PV',pv_start)
    softmax=loop[qk_end:pv_start]
    finish=loop[pv_end:]
    require(finish.endswith('\n }'), 'tile loop terminator differs')
    helper=LOADS+SIGNATURE+QK+softmax+PV+finish[:-2]+'}\n'
    changed=tail[:loop_start]+CALLS+tail[loop_end:]
    changed=changed.replace(' const int count=static_cast<int>(position)+1;\n','',1)
    changed=changed.replace(' __shared__ __nv_bfloat16 probs[128];',' __shared__ __align__(4) __nv_bfloat16 probs[128];',1)
    changed=changed.replace(' __shared__ __nv_bfloat16 local_q[64],local_k[64];',' __shared__ __align__(4) __nv_bfloat16 local_q[64],local_k[64];',1)
    changed=changed.replace(' const __nv_bfloat16 zero=__float2bfloat16_rn(0.);\n','',1)
    generated=(source[:start]+helper+changed).encode()
    oracle_end=source.index('\n// Packed M1 decode only. Fold precise RoPE')
    require(digest(raw[:oracle_end])==ORACLE_SHA, 'accepted7 prefix differs')
    require(generated.startswith(raw[:oracle_end]), 'accepted7 prefix changed')
    require(source[:start] in generated.decode(), 'explicit RoPE helpers changed')
    require(tail[loop_end:]==changed[changed.index(END):], 'final denominator/output or wrapper changed')
    require(softmax in generated.decode(), 'MODE6 softmax/denominator order changed')
    return generated, {'baseline_sha256':BASELINE_SHA,'candidate_sha256':digest(generated),
        'accepted7_prefix_sha256':ORACLE_SHA,'preserved_softmax_bytes_sha256':digest(softmax.encode()),
        'preserved_finalization_bytes_sha256':digest(tail[loop_end:].encode()),
        'changes':['compile-specialized tail then full historical tile','hoisted key bases and historical PV page arithmetic',
                   'aligned raw32 shared Q/prob and K pair loads'],
        'softmax_and_finalization_text_unchanged':True,
        'mma_operand_mapping_and_chain_order':'source contract; GPU qualification pending',
        'tile_order':['tail128..position','history0..127'],
        'cuda_compile_executed':False,'gpu_executed':False,'performance_claim':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=HERE/'baseline_graph_numerics.cu')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();raw=args.baseline.read_bytes();generated,proof=transform(raw)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    patch=''.join(difflib.unified_diff(raw.decode().splitlines(keepends=True),generated.decode().splitlines(keepends=True),
        fromfile='a/kernels/src/graph_numerics.cu',tofile='b/kernels/src/graph_numerics.cu'))
    for name,contents in [('graph_numerics.cu',generated),('candidate.patch',patch.encode()),
                          ('invariants.json',(json.dumps(proof,indent=2)+'\n').encode())]:
        with (args.output_dir/name).open('xb') as f:f.write(contents)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
