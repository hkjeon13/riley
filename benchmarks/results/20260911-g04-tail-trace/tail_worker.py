"""Controlled-input MLP diagnosis; never modifies serving model weights or methods."""
def capture(model):
    import hashlib
    import json
    from pathlib import Path
    import torch
    from vllm.ir.op import enable_torch_wrap
    root = Path('/tmp/riley-g04-tail-trace-260911')
    cases = json.loads((root/'prefixes.json').read_text())
    layer = model.model.layers[0]
    mlp = layer.mlp
    widths = {'after_attention_residual':576, 'post_attention_norm':576,
              'gate_proj':1536, 'up_proj':1536, 'gated':1536,
              'down_proj':576, 'output':576}
    records = {}
    def save(case, name, value, reference):
        value = value.contiguous()
        assert value.shape == reference.shape and value.dtype == reference.dtype
        assert torch.isfinite(value).all()
        raw = value.view(torch.uint8).cpu().numpy().tobytes()
        filename = f'{case}-{name}.bf16'
        (root/filename).write_bytes(raw)
        records[f'{case}/{name}'] = {'file':filename, 'elements':value.numel(),
            'unequal':int((value != reference).sum().item()),
            'max_abs':float((value.float()-reference.float()).abs().max().item()),
            'sha256':hashlib.sha256(raw).hexdigest()}
    def norm(x): return layer.post_attention_layernorm(x)
    def packed(x): return mlp.gate_up_proj(x)[0]
    def split(x):
        g,u = mlp.gate_up_proj.weight.chunk(2,dim=0)
        return torch.cat([torch.nn.functional.linear(x,g),torch.nn.functional.linear(x,u)],dim=-1)
    def activation(x): return mlp.act_fn(x)
    def down(x): return mlp.down_proj(x)[0]
    def staged(x):
        g,u=x.chunk(2,dim=-1)
        return torch.nn.functional.silu(g)*u
    def fused_formula(x):
        g,u=x.float().chunk(2,dim=-1)
        return (torch.nn.functional.silu(g)*u).to(torch.bfloat16)
    funcs = {name:torch.compile(fn,fullgraph=True,dynamic=True) for name,fn in
             [('norm',norm),('activation',activation)]}
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        with torch.inference_mode(), enable_torch_wrap(False):
            for case,ids in cases.items():
                t={name:torch.frombuffer(bytearray((root/'riley'/f'{case}-layer0.{name}.bf16').read_bytes()),
                    dtype=torch.bfloat16).reshape(len(ids),width).to('cuda') for name,width in widths.items()}
                for mode,fn in [('eager',norm),('compiled',funcs['norm'])]:
                    save(case,'norm_'+mode,fn(t['after_attention_residual']),t['post_attention_norm'])
                for reduced in [True,False]:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
                    for name,fn in [('packed',packed),('split',split)]:
                        g,u=fn(t['post_attention_norm']).chunk(2,dim=-1)
                        save(case,f'gate_{name}_reduced{reduced}',g,t['gate_proj'])
                        save(case,f'up_{name}_reduced{reduced}',u,t['up_proj'])
                    save(case,f'down_reduced{reduced}',down(t['gated']),t['down_proj'])
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
                gu=torch.cat([t['gate_proj'],t['up_proj']],dim=-1)
                for name,fn in [('eager',activation),('compiled',funcs['activation']),
                                ('staged',staged),('fp32_formula',fused_formula)]:
                    save(case,'swiglu_'+name,fn(gu),t['gated'])
                save(case,'residual_add',t['after_attention_residual']+t['down_proj'],t['output'])
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    result={'scope':'fixed Riley operation inputs; independent operation replay, not a serving graph tap',
            'performance_trials':0,'original_reduced_precision_flag':original,
            'restored_reduced_precision_flag':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'records':records}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

class TailTrace:
    def export_tail(self): return capture(self.get_model())
