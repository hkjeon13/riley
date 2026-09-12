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
captured = []
def capture(module, inputs):
    x = inputs[0]
    captured.append((module.gate_proj(x).detach(), module.up_proj(x).detach()))
handle = model.model.layers[0].mlp.register_forward_pre_hook(capture)
with torch.inference_mode():
    model(input_ids=prompt, use_cache=False)
    handle.remove()
    gate, up = captured[0]
    joined = torch.cat([gate,up],dim=-1).reshape(-1,2*gate.shape[-1])
    native = torch.empty_like(gate.reshape(-1,gate.shape[-1]))
    torch.ops._C.silu_and_mul(native, joined)
    native = native.reshape_as(gate)
    def expression(g,u):
        return torch.nn.functional.silu(g)*u
    eager = expression(gate,up)
    compiled = torch.compile(expression, fullgraph=True)(gate,up)
    single_round = (torch.nn.functional.silu(gate.float())*up.float()).to(torch.bfloat16)
    def comparison(a,b):
        return {'unequal':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
    result={'correctness_only':True,'performance_trials':0,'tensor':'HF layer0 MLP input projections p128',
            'elements':gate.numel(),'compiled_vs_eager':comparison(compiled,eager),
            'native_vs_eager':comparison(native,eager),'compiled_vs_single_round':comparison(compiled,single_round),
            'native_vs_single_round':comparison(native,single_round)}
    (root/'swiglu-tensor.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
