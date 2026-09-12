"""CPU source/address contracts only; these tests do not simulate GPU arithmetic."""
import difflib
import importlib.util
import json
from pathlib import Path
import struct
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fusion_history_generator', HERE/'generate.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
BASE = (HERE/'baseline_graph_numerics.cu').read_bytes()
NEW, PROOF = gen.transform(BASE)
OLD_KERNEL = BASE.decode().split(gen.KERNEL, 1)[1]
NEW_KERNEL = NEW.decode().split(gen.KERNEL, 1)[1]
MAPS = [list(range(16)), list(reversed(range(16))), [(7*i+3)%16 for i in range(16)]]


def original_index(token, head, dim, blocks):
    return ((blocks[token//16]*3+head)*16+token%16)*64+dim


class CandidateContract(unittest.TestCase):
    def test_requires_exact_frozen_source(self):
        self.assertEqual(gen.digest(BASE), gen.BASELINE_SHA)
        for changed in (BASE+b'\n', BASE.replace(b'MODE 6', b'MODE 5', 1)):
            with self.assertRaisesRegex(ValueError, 'exact frozen'):
                gen.transform(changed)

    def test_generated_artifacts_are_reproducible(self):
        self.assertEqual(NEW, (HERE/'graph_numerics.cu').read_bytes())
        self.assertEqual(PROOF, json.loads((HERE/'invariants.json').read_text()))
        expected = ''.join(difflib.unified_diff(BASE.decode().splitlines(keepends=True),
            NEW.decode().splitlines(keepends=True), fromfile='a/kernels/src/graph_numerics.cu',
            tofile='b/kernels/src/graph_numerics.cu'))
        self.assertEqual(expected, (HERE/'candidate.patch').read_text())
        self.assertFalse(PROOF['cuda_compile_executed'])
        self.assertFalse(PROOF['gpu_executed'])
        self.assertFalse(PROOF['performance_claim'])

    def test_accepted_prefix_and_rope_publication_unchanged(self):
        prefix = BASE.split(b'\n// Packed M1 decode only. Fold precise RoPE', 1)[0]
        self.assertEqual(gen.digest(prefix), gen.ORACLE_SHA)
        self.assertTrue(NEW.startswith(BASE.split(gen.KERNEL.encode(), 1)[0]))
        old = OLD_KERNEL.split(gen.LOOP, 1)[0]
        old = old.replace(' const int count=static_cast<int>(position)+1;\n', '')
        old = old.replace(' const __nv_bfloat16 zero=__float2bfloat16_rn(0.);\n', '')
        old = old.replace('__shared__ __nv_bfloat16', '__shared__ __align__(4) __nv_bfloat16')
        new = NEW_KERNEL.split(' // The validated range', 1)[0]
        self.assertEqual(old, new)

    def test_softmax_finalization_mma_and_barriers(self):
        loop = OLD_KERNEL.split(gen.LOOP, 1)[1].split(gen.END, 1)[0]
        softmax = loop[loop.index('  // Includes the idle'):loop.index('  for(int block=0;block<2;++block){')]
        helper = NEW.decode().split('template<bool History>', 1)[1].split(gen.KERNEL, 1)[0]
        self.assertEqual(helper.count(softmax), 1)
        self.assertEqual(OLD_KERNEL[OLD_KERNEL.index(gen.END):], NEW_KERNEL[NEW_KERNEL.index(gen.END):])
        self.assertEqual(helper.count('__syncthreads();'), 3)
        self.assertEqual(helper.count('__syncwarp();'), 1)
        self.assertEqual(NEW_KERNEL.count('__syncthreads();'), 2)
        for statement in ('mma(d,a,a,aa,aa,b,bb);', 'mma(accum[block],a,a,aa,aa,b,bb);',
                          'for(int depth=0;depth<64;depth+=16)',
                          'for(int j=0;j<4;++j)accum[block][j]*=alpha;'):
            self.assertEqual(loop.count(statement), 1)
            self.assertEqual(helper.count(statement), 1)

    def test_two_tiles_and_per_output_order(self):
        self.assertEqual(NEW_KERNEL.count('fused_history_tile<false>'), 1)
        self.assertEqual(NEW_KERNEL.count('fused_history_tile<true>'), 1)
        self.assertLess(NEW_KERNEL.index('fused_history_tile<false>'), NEW_KERNEL.index('fused_history_tile<true>'))
        for position in range(128, 160):
            original = [(tile*128, min(tile*128+128, position+1))
                        for tile in range(position//128, -1, -1)]
            specialized = [(128, position+1), (0, 128)]
            self.assertEqual(original, specialized)
            for warp in range(2):
                # Depth order is the complete per-score MMA chain. Output block
                # order retains each separate accumulator's tail then history.
                qk = lambda tiles: [(begin, token, depth) for begin,end in tiles
                    for token in range(begin+8*warp, end, 16) for depth in range(0,64,16)]
                pv = lambda tiles: [(begin, block, token) for begin,end in tiles
                    for block in range(2) for token in range(begin,end,16)]
                self.assertEqual(qk(original), qk(specialized))
                self.assertEqual(pv(original), pv(specialized))

    def test_qk_addresses_tail_bypass_and_alignment(self):
        for blocks in MAPS:
            for position in range(128, 160):
                for head in range(3):
                    for begin,end,history in ((128,position+1,False),(0,128,True)):
                        for warp in range(2):
                            for token in range(begin+8*warp,end,16):
                                for lane in range(32):
                                    group,t = divmod(lane,4)
                                    key_token=token+group
                                    for depth in range(0,64,16):
                                        for shift in (0,8):
                                            dim=depth+2*t+shift
                                            self.assertEqual(dim%2,0)
                                            self.assertLess(dim+1,64)
                                            if key_token>=end:
                                                self.assertFalse(history)
                                                continue  # Both implementations feed raw32 positive zero.
                                            if key_token==position:
                                                self.assertFalse(history)
                                                continue  # Both implementations read rounded local K.
                                            old=original_index(key_token,head,dim,blocks)
                                            base=(((blocks[token//16]*3+head)*16+token%16+group)*64
                                                  if history else original_index(key_token,head,0,blocks))
                                            new=base+dim
                                            self.assertEqual(old,new)
                                            self.assertEqual((2*new)%4,0)
                                            self.assertLess(new+1,16*3*16*64)
                                            self.assertNotEqual(key_token,position)

    def test_pv_addresses_and_probability_load_bounds(self):
        for blocks in MAPS:
            for position in range(128,160):
                for head in range(3):
                    for begin,end,history in ((128,position+1,False),(0,128,True)):
                        for half in range(2):
                            for warp in range(2):
                                for lane in range(32):
                                    group,t=divmod(lane,4)
                                    for block in range(2):
                                        dim=(half*4+warp*2+block)*8+group
                                        for token in range(begin,end,16):
                                            pi=token-begin
                                            for shift in (0,8):
                                                start=pi+2*t+shift
                                                self.assertEqual((start*2)%4,0)
                                                self.assertLess(start+1,128)
                                            for offset in (2*t,2*t+1,2*t+8,2*t+9):
                                                pos=token+offset
                                                if pos>=end or pos==position:
                                                    self.assertFalse(history)
                                                    continue  # Exact unchanged zero/raw-V tail branches.
                                                old=original_index(pos,head,dim,blocks)
                                                new=(((blocks[token//16]*3+head)*16)*64+dim+offset*64
                                                     if history else old)
                                                self.assertEqual(old,new)
                                                self.assertLess(new,16*3*16*64)
        # V pairs intentionally remain strided gathers, not adjacent raw32 loads.
        self.assertNotIn('fused_history_global_pair(v', NEW.decode())

    def test_raw32_pair_preserves_every_bf16_bit_pattern(self):
        # Includes signed zeros, both subnormal signs, infinities and NaN payloads.
        # This checks byte mapping, not arithmetic or device load code generation.
        for low in range(65536):
            high=low^0xffff
            loaded=struct.unpack('<I',struct.pack('<HH',low,high))[0]
            self.assertEqual(loaded,low|(high<<16))
        self.assertIn('__shared__ __align__(4) __nv_bfloat16 probs[128]', NEW_KERNEL)
        self.assertIn('__shared__ __align__(4) __nv_bfloat16 local_q[64],local_k[64]', NEW_KERNEL)
        self.assertEqual(gen.LOADS.count(': "memory"'),2)

    def test_full_output_coverage_and_unique_publication(self):
        outputs=[];q_writes=[];kv_writes=[]
        for head in range(9):
            for half in range(2):
                for warp in range(2):
                    for lane in range(32):
                        group,t=divmod(lane,4)
                        if group==0:
                            for block in range(2):
                                for j in range(2):outputs.append(head*64+(half*4+warp*2+block)*8+2*t+j)
                        if warp==0 and half==0:q_writes.extend([head*64+lane,head*64+lane+32])
                        if warp==1 and head%3==0 and half==0:
                            kv_writes.extend([(head//3)*64+lane,(head//3)*64+lane+32])
        self.assertEqual(sorted(outputs),list(range(576)))
        self.assertEqual(sorted(q_writes),list(range(576)))
        self.assertEqual(sorted(kv_writes),list(range(192)))


if __name__=='__main__':unittest.main()
