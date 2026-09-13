"""Diagnostic-only, hash-verified BF16 probability compensation overlay."""
import hashlib

den_anchor = 'mma::m16k16_rowsum_f16f16f32(d[mma_q], s_frag_f16[mma_q][mma_kv]);'
den_code = '// Riley diagnostic: retain FP32 probabilities in the denominator.\n          #pragma unroll\n          for(unsigned row=0;row<2;++row){\n            const float* p=s_frag[mma_q][mma_kv];\n            float sum=(p[2*row]+p[2*row+1])+(p[2*row+4]+p[2*row+5]);\n            sum+=__shfl_xor_sync(0xffffffff,sum,1);\n            sum+=__shfl_xor_sync(0xffffffff,sum,2);\n            d[mma_q][row]+=sum;\n          }'
declaration = 'typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];'
cast = 'vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_frag_f16[mma_q][mma_kv],\n                                                           s_frag[mma_q][mma_kv]);'
mma = 'mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(\n                o_frag[mma_q][mma_d_local], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag);'
extra = '\n        float residual[8];\n        #pragma unroll\n        for(unsigned i=0;i<8;++i)residual[i]=s_frag[mma_q][mma_kv][i]-float(s_frag_f16[mma_q][mma_kv][i]);\n        vec_cast<typename KTraits::DTypeQ, float>::cast<8>(s_residual[mma_q][mma_kv],residual);'

def transform(text, residual=True, synchronize_kv_warps=False):
    if hashlib.sha256(text.encode()).hexdigest() != '996253b7c64caaa3ed86540fb5d5ca44482298c9e8c9e3665b91ba5310b0e975':
        raise ValueError('expected the pinned warp-synchronized prefill header')
    if text.count(den_anchor) != 1:
        raise ValueError('denominator context mismatch')
    text = text.replace(den_anchor, den_code)
    if residual:
        if any(text.count(x) != 1 for x in [declaration, cast, mma]):
            raise ValueError('residual context mismatch')
        text = text.replace(declaration, declaration + '\n  typename KTraits::DTypeQ s_residual[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][8];')
        text = text.replace(cast, cast + extra)
        text = text.replace(mma, mma + '\n            ' + mma.replace('s_frag_f16', 's_residual'))
    expected = ('e467d96c13eb05688488e68d5b408d5c37f6ed9251af15484080d2cb86c47e1e' if residual else
                '62026012e8afc44eb30691f07b60df928e1390a8632f53142d604810a402e68e')
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise ValueError('compensated prefill header differs from the tested transformation')
    if synchronize_kv_warps:
        if not residual:
            raise ValueError('KV-warp synchronization experiment requires residual arithmetic')
        anchor = 'const uint_fastdiv group_size, const dim3 tid = threadIdx) {\n  using DTypeO = typename KTraits::DTypeO;'
        replacement = ('const uint_fastdiv group_size, const dim3 tid = threadIdx) {\n'
                       '  // Finish KV-warp reduction reads before reusing shared storage for output.\n'
                       '  if constexpr (KTraits::NUM_WARPS_KV > 1) __syncthreads();\n'
                       '  using DTypeO = typename KTraits::DTypeO;')
        if text.count(anchor) != 1:
            raise ValueError('output shared-storage boundary mismatch')
        text = text.replace(anchor, replacement)
        if hashlib.sha256(text.encode()).hexdigest() != 'd01f73f64e51a95748a2034bf8d15890ccffe366301192458af48db074bfcb6a':
            raise ValueError('KV-warp synchronization overlay differs from tested header')
    return text
