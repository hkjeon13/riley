"""Local diagnostic worker extension; no callable deserialization."""
def capture(model):
    import json
    from pathlib import Path
    import torch
    target = Path('/tmp/riley-g04-prefix-trace-260911')
    cases = json.loads((target/'prefixes.json').read_text())
    m = model.model
    first = m.layers[0]
    def prefix(ids):
        x = m.embed_tokens(ids)
        y = first.input_layernorm(x)
        qkv, _ = first.self_attn.qkv_proj(y)
        return x, y, qkv
    def opaque_prefix(ids):
        return prefix(ids)[2]
    def split_prefix(ids):
        x = m.embed_tokens(ids)
        y = first.input_layernorm(x)
        wq,wk,wv = first.self_attn.qkv_proj.weight.split([576,192,192],dim=0)
        qkv = torch.cat([torch.nn.functional.linear(y,w) for w in (wq,wk,wv)],dim=-1)
        return x,y,qkv
    def fp32(fn, ids):
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            return fn(ids)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    def full_precision_split(ids):
        return fp32(split_prefix, ids)
    def full_precision_prefix(ids):
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            return prefix(ids)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    compiled = torch.compile(prefix, fullgraph=True, dynamic=True)
    opaque = torch.compile(opaque_prefix, fullgraph=True, dynamic=True)
    records = []
    from vllm.ir.op import enable_torch_wrap
    with torch.inference_mode(), enable_torch_wrap(False):
        for name, tokens in cases.items():
            ids = torch.tensor(tokens, dtype=torch.int32, device='cuda')
            for mode, fn in [('eager',prefix), ('compiled',compiled), ('eager_fp32',full_precision_prefix), ('split',split_prefix), ('split_fp32',full_precision_split)]:
                x,y,qkv = fn(ids)
                if mode == 'compiled':
                    assert torch.equal(qkv, opaque(ids)), 'exposing norm changed QKV'
                q,k,v = qkv.split([576,192,192],dim=-1)
                for label,tensor in [('embedding',x),('layer0.input_norm',y),
                                     ('layer0.q_proj',q),('layer0.k_proj',k),('layer0.v_proj',v)]:
                    t = tensor.detach().contiguous()
                    name_on_disk=f'{name}-{mode}-{label}.bf16'
                    (target/name_on_disk).write_bytes(t.view(torch.uint8).cpu().numpy().tobytes())
                    records.append({'file':name_on_disk,'shape':list(t.shape),'dtype':str(t.dtype)})
    return {'records':records,'exposed_vs_opaque_qkv_exact':True,
            'scope':'isolated loaded-model prefix replay, before RoPE/attention; not whole-graph capture'}


class PrefixTrace:
    def export_prefix(self):
        return capture(self.get_model())
