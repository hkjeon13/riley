"""Diagnostic register conversion; BF16 storage and FP32 accumulation remain."""
import hashlib
import re

BASE = '996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975'
PATCHED = '70452db62dc984d5be3155f97f5f7b369e4098e6617be938a0d2b8a705edcbb9'

def close_paren(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == '(') - (text[i] == ')')
        if depth == 0:
            return i
    raise ValueError('unbalanced function/call')

def transform(text):
    if hashlib.sha256(text.encode()).hexdigest() != BASE:
        raise ValueError('expected pinned warp-synchronized BF16 prefill header')
    for name in ['compute_qk', 'compute_sfm_v']:
        start = text.index('__device__ __forceinline__ void '+name+'(')
        opening = text.index('(', start)
        closing = close_paren(text, opening)
        text = text[:opening+1] + 'uint32_t** riley_status, ' + text[opening+1:]
        body = text.index('{', closing)
        # These functions are followed by another top-level template definition.
        end = text.index('\n}\n', body) + 2
        section = text[start:end]
        section = section.replace('typename KTraits::DTypeQ, MMAMode', 'half, MMAMode')
        section = section.replace('f16f16f32<typename KTraits::DTypeQ>', 'f16f16f32<half>')
        if name == 'compute_qk':
            for anchor, fragment in [
                ('q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);', 'a_frag[mma_q]'),
                ('k_smem->ldmatrix_m8n8x4(*k_smem_offset_r, b_frag);', 'b_frag')]:
                if section.count(anchor) != 1: raise ValueError('Q/K fragment anchor changed')
                section = section.replace(anchor, anchor+'\n        riley_half_fragment('+fragment+', riley_status);')
        else:
            declaration = 'typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];'
            cast = 'vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],\n                                                           s_frag[mma_q][mma_kv]);'
            load = 'v_smem->ldmatrix_m8n8x4_trans(*v_smem_offset_r, b_frag);'
            if any(section.count(x) != 1 for x in [declaration, cast, load]):
                raise ValueError('probability/V fragment anchor changed')
            section = section.replace(declaration, declaration.replace('typename KTraits::DTypeQ', 'half'))
            section = section.replace(cast, 'for(unsigned i=0;i<8;++i)s_frag_f16[mma_q][mma_kv][i]=riley_to_half(s_frag[mma_q][mma_kv][i],riley_status);')
            section = section.replace(load, load+'\n          riley_half_fragment(b_frag, riley_status);')
        brace = section.index('{')
        section = section[:brace+1]+'''
  static_assert(std::is_same_v<typename KTraits::DTypeQ, __nv_bfloat16> &&
                std::is_same_v<typename KTraits::DTypeKV, __nv_bfloat16> &&
                std::is_same_v<typename KTraits::DTypeQKAccum, float>,
                "Riley half diagnostic only accepts BF16 storage / FP32 accumulation");
'''+section[brace+1:]
        text = text[:start]+section+text[end:]
    calls = list(re.finditer(r'\bcompute_(?:qk|sfm_v)<KTraits[^>]*>\s*\(', text))
    if len(calls) != 10: raise ValueError(f'expected ten prefill call sites, got {len(calls)}')
    for match in reversed(calls):
        closing = close_paren(text, match.end()-1)
        text = text[:match.end()]+'params.riley_status_slot, '+text[match.end():]
    helper = '''
// Explicit numerical profile: finite overflow is rejected through the retained
// status owner. FP16 subnormals are deliberately flushed to signed zero before
// MMA, including rounded probabilities, rather than relying on GPU behavior.
__device__ __forceinline__ half riley_to_half(float x, uint32_t** status) {
  if(!isfinite(x) || fabsf(x)>65504.F){atomicOr(*status,32u);return __float2half_rn(0.F);}
  if(fabsf(x)<0x1p-14F)x=copysignf(0.F,x);
  return __float2half_rn(x);
}
__device__ __forceinline__ void riley_half_fragment(uint32_t (&fragment)[4], uint32_t** status){
  #pragma unroll
  for(unsigned i=0;i<4;++i){
    uint32_t bits=fragment[i];
    half lo=riley_to_half(__bfloat162float(__ushort_as_bfloat16(bits&65535)),status);
    half hi=riley_to_half(__bfloat162float(__ushort_as_bfloat16(bits>>16)),status);
    fragment[i]=uint32_t(__half_as_ushort(lo))|(uint32_t(__half_as_ushort(hi))<<16);
  }
}
'''
    anchor = 'template <typename KTraits, bool REPACK_BF16 = false>\n__device__ __forceinline__ void compute_qk('
    if text.count(anchor) != 1: raise ValueError('helper insertion anchor changed')
    result = text.replace(anchor, helper+anchor)
    if hashlib.sha256(result.encode()).hexdigest() != PATCHED:
        raise ValueError('half-register diagnostic differs from verified header')
    return result
