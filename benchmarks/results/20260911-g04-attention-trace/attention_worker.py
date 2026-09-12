"""Fixed-input attention formula and fused residual normalization probes."""
def capture(model):
    import json,hashlib
    from pathlib import Path
    import torch
    from vllm.ir.op import enable_torch_wrap
    root=Path('/tmp/riley-g04-attention-trace-260911')
    layer=model.model.layers[0]; attn=layer.self_attn; norm=layer.post_attention_layernorm
    cases=json.loads((root/'prefixes.json').read_text()); records={}
    def save(case,name,x,ref):
        x=x.contiguous();ref=ref.contiguous()
        assert x.shape==ref.shape and x.dtype==ref.dtype and torch.isfinite(x).all()
        raw=x.view(torch.uint8).cpu().numpy().tobytes()
        target=ref.view(torch.uint8).cpu().numpy().tobytes()
        (root/f'{case}-{name}.bf16').write_bytes(raw)
        (root/f'{case}-{name}.reference.bf16').write_bytes(target)
        records[f'{case}/{name}']={'elements':x.numel(),'unequal':int((x!=ref).sum().item()),
            'max_abs':float((x.float()-ref.float()).abs().max().item()),
            'sha256':hashlib.sha256(raw).hexdigest(),'reference_sha256':hashlib.sha256(target).hexdigest()}
    def rope(pos,q,k): return attn.rotary_emb(pos,q,k)
    def fused(x,r): return norm(x,r)
    compiled_rope=torch.compile(rope,fullgraph=True,dynamic=True)
    compiled_fused=torch.compile(fused,fullgraph=True,dynamic=True)
    original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    with torch.inference_mode(),enable_torch_wrap(False):
        for case,ids in cases.items():
            n=len(ids)
            def read(name,shape):
                raw=bytearray((root/'riley'/f'{case}-{name}.bf16').read_bytes())
                return torch.frombuffer(raw,dtype=torch.bfloat16).reshape(shape).to('cuda')
            q=read('layer0.q_proj',(n,576));k=read('layer0.k_proj',(n,192));v=read('layer0.v_proj',(n,192))
            probs=read('layer0.attention_probs',(9,n,n));context=read('layer0.attention_context',(n,576))
            emb=read('embedding',(n,576));res=read('layer0.after_attention_residual',(n,576))
            normref=read('layer0.post_attention_norm',(n,576))
            pos=torch.arange(n,device='cuda')
            qe,ke=rope(pos,q.clone(),k.clone());qc,kc=compiled_rope(pos,q.clone(),k.clone())
            save(case,'rope_compiled_q_vs_selected',qc,qe);save(case,'rope_compiled_k_vs_selected',kc,ke)
            # Formula replay uses loaded vLLM RoPE; it is NOT its paged attention backend.
            for mode,qr,kr in [('selected',qe,ke),('compiled',qc,kc)]:
                Q=qr.reshape(n,9,64).transpose(0,1)
                K=kr.reshape(n,3,64).transpose(0,1).repeat_interleave(3,dim=0)
                V=v.reshape(n,3,64).transpose(0,1).repeat_interleave(3,dim=0)
                scores=(Q@K.transpose(-1,-2))*0.125
                scores.masked_fill_(torch.ones(n,n,device='cuda',dtype=torch.bool).triu(1),float('-inf'))
                P=torch.softmax(scores.float(),dim=-1).to(torch.bfloat16)
                C=(P@V).transpose(0,1).reshape(n,576)
                save(case,'attention_probs_'+mode,P,probs);save(case,'attention_context_'+mode,C,context)
                sdpa=torch.nn.functional.scaled_dot_product_attention(Q,K,V,is_causal=True)
                save(case,'sdpa_context_'+mode,sdpa.transpose(0,1).reshape(n,576),context)
            projection=attn.o_proj(context)[0]
            save(case,'projection_residual_add',projection+emb,res)
            # Exact reference context feeds o_proj; verify residual above before attributing norm errors.
            for mode,fn in [('selected',fused),('compiled',compiled_fused)]:
                y,r=fn(projection.clone(),emb.clone())
                save(case,'fused_norm_'+mode,y,normref);save(case,'fused_residual_'+mode,r,res)
            z=projection.float()+emb.float()
            weight=norm.weight.float();eps=norm.variance_epsilon
            y=(z*torch.rsqrt((z*z).mean(-1,keepdim=True)+eps)*weight).to(torch.bfloat16)
            save(case,'fused_norm_fp32_formula',y,normref)
            rounded=z.to(torch.bfloat16).float()
            yr=(rounded*torch.rsqrt((rounded*rounded).mean(-1,keepdim=True)+eps)*weight).to(torch.bfloat16)
            save(case,'rounded_sum_fp32_norm',yr,normref)
    result={'performance_trials':0,'scope':'fixed input formula replay, not paged attention or full serving graph',
            'reduced_precision_flag':original,'records':records}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

class AttentionTrace:
    def export_attention(self): return capture(self.get_model())
