"""Correctness-only numerical diagnosis; never collects performance samples."""
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  VLLM_DO_NOT_TRACK="1", VLLM_NO_USAGE_STATS="1")
import torch
from transformers import AutoModelForCausalLM

root = Path(__file__).resolve().parent
previous = Path('/tmp/riley-g04-readiness-260911')
env = json.loads((previous / 'environment.json').read_text())
request = json.loads((previous / 'request.json').read_text())
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
model = AutoModelForCausalLM.from_pretrained(
    env['vllm_checkpoint'], dtype=torch.bfloat16, attn_implementation='eager',
    local_files_only=True).to('cuda').eval()
prompt = torch.tensor([request['prompt_token_ids']], device='cuda')
from vllm import _custom_ops as ops
with torch.inference_mode():
    x=model.model.embed_tokens(prompt)
    w=model.model.layers[0].input_layernorm.weight
    def expression(x,w):
        y=x.float()
        y=y*torch.rsqrt(y.pow(2).mean(-1,keepdim=True)+1e-5)
        return y.to(torch.bfloat16)*w
    eager=expression(x,w)
    compiled=torch.compile(expression,fullgraph=True)(x,w)
    native=torch.empty_like(x)
    ops.rms_norm(native,x,w,1e-5)
    def cmp(a,b):
        return {'unequal':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
    result={'correctness_only':True,'performance_trials':0,'elements':x.numel(),
            'compiled_vs_eager':cmp(compiled,eager),'compiled_vs_native':cmp(compiled,native),'native_vs_eager':cmp(native,eager)}
    (root/'norm-tensor.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
