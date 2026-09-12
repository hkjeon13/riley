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
import types
# Diagnostic only: keep the residual sum in FP32 through normalization,
# while materializing its BF16 residual for the following addition.
def residual_forward(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
    residual = hidden_states.to(torch.bfloat16)
    h = self.input_layernorm(hidden_states)
    h, _ = self.self_attn(hidden_states=h, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
    h = residual.float() + h.float()
    residual = h.to(torch.bfloat16)
    h = self.post_attention_layernorm(h)
    h = self.mlp(h)
    return residual.float() + h.float()
def norm_forward(self, x):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return x.to(torch.bfloat16) * self.weight
def mlp_forward(self, x):
    g, u = self.gate_proj(x), self.up_proj(x)
    return self.down_proj((torch.nn.functional.silu(g.float())*u.float()).to(torch.bfloat16))
for m in model.modules():
    if m.__class__.__name__ == 'LlamaDecoderLayer':
        m.forward = types.MethodType(residual_forward, m)
    elif m.__class__.__name__ == 'LlamaMLP':
        m.forward = types.MethodType(mlp_forward, m)
    elif m.__class__.__name__ == 'LlamaRMSNorm':
        m.forward = types.MethodType(norm_forward, m)
results = {'diagnostic_fp32_residual': True, 'performance_trials': 0, 'torch': torch.__version__, 'cuda': torch.version.cuda}

with torch.inference_mode():
    from vllm import _custom_ops as ops
    x = model.model.embed_tokens(prompt[:, :1])
    norm = model.model.layers[0].input_layernorm
    hf = norm(x)
    native = torch.empty_like(x)
    ops.rms_norm(native, x, norm.weight, norm.variance_epsilon)
    results['first_embedding_norm'] = {
        'unequal_elements': int((hf != native).sum().item()),
        'elements': x.numel(), 'max_abs_error': (hf.float()-native.float()).abs().max().item()}
    for chunk in (1, 128):
        cache = None
        for offset in range(0, 128, chunk):
            out = model(input_ids=prompt[:, offset:offset+chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        tokens = []
        ranks = []
        for index in range(32):
            logits = out.logits[0, -1].float()
            values, ids = logits.topk(5)
            ranks.append({'index': index, 'ids': ids.tolist(), 'logits': values.tolist()})
            token = int(logits.argmax().item())
            tokens.append(token)
            if index != 31:
                out = model(input_ids=torch.tensor([[token]], device='cuda'), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
        results[f'hf_prefill_chunk_{chunk}'] = {'tokens': tokens, 'top5': ranks}
        print('HF_CORRECTNESS', chunk, tokens, flush=True)
        (root / 'residual-swiglu-probe.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results['first_embedding_norm']), flush=True)
