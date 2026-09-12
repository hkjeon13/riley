from pathlib import Path
import subprocess
src=Path('/tmp/riley-g04-attention-policy-source-260911')
assert not src.exists()
subprocess.run(['git','clone','--no-hardlinks','/tmp/riley-g04-followup-source-260911',str(src)],check=True)
p=src/'kernels/src/batch_primitives.cu'
s=p.read_text();old='''  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(staged_dot) * scale));'''
assert s.count(old)==1
s=s.replace(old,'  return dot_product * scale; // DIAGNOSTIC ONLY: FP32 QK score')
p.write_text(s)
(src/'crates/riley-cuda/tests/serving_attention_replay_gpu.rs').write_bytes(Path('/tmp/riley-g04-prefix-source-260911/crates/riley-cuda/tests/serving_attention_replay_gpu.rs').read_bytes())
