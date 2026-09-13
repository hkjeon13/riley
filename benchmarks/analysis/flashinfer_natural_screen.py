#!/usr/bin/env python3
"""Offline preparation/reference metrics, never called by Rust serving."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);p.add_argument('--model',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--passages',type=Path);args=p.parse_args()
directory=args.directory;directory.mkdir(parents=True,exist_ok=True)
if args.mode=='prepare':
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    cases=[]
    for text in json.loads(args.passages.read_text()):
        tokens=tokenizer.encode(text,add_special_tokens=False).ids
        assert len(tokens)>=64
        cases.append({'prompt':tokens[:32],'targets':tokens[32:64]})
    (directory/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    (directory/'input-hashes.json').write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.passages,args.model/'tokenizer.json',directory/'cases.json']},indent=2)+'\n')
else:
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cases=json.loads((directory/'cases.json').read_text());assert len(cases)==8
    logits={}
    for name in ['baseline','candidate']:
        raw=bytearray((directory/(name+'.bf16')).read_bytes());assert len(raw)==8*32*49152*2
        logits[name]=torch.frombuffer(raw,dtype=torch.bfloat16).float().reshape(8,32,49152).cuda()
        assert bool(logits[name].isfinite().all())
    model=AutoModelForCausalLM.from_pretrained(str(args.model),dtype=torch.float32,attn_implementation='eager',local_files_only=True,trust_remote_code=False).eval().cuda()
    rows=[]
    with torch.inference_mode():
        for index,case in enumerate(cases):
            tokens=case['prompt']+case['targets'];assert len(tokens)==64
            ref=model(torch.tensor([tokens],device='cuda'),use_cache=False).logits[0,31:63].float()
            assert bool(ref.isfinite().all());rlp=ref.log_softmax(-1);prob=rlp.exp();targets=torch.tensor(case['targets'],device='cuda')
            row={'index':index}
            for name in logits:
                lp=logits[name][index].log_softmax(-1)
                row[name]={'nll':float(-lp.gather(1,targets[:,None]).mean()),'kl_from_fp32':float((prob*(rlp-lp)).sum(-1).mean()),'argmax_fp32_matches':int((ref.argmax(-1)==logits[name][index].argmax(-1)).sum())}
            rows.append(row)
    aggregate={name:{key:sum(row[name][key] for row in rows)/(1 if key=='argmax_fp32_matches' else len(rows)) for key in rows[0][name]} for name in logits}
    passed=all(aggregate['candidate'][key]<=aggregate['baseline'][key] for key in ['nll','kl_from_fp32'])
    report={'passages':rows,'aggregate':aggregate,'relative_screen_passed':passed,'general_quality_accepted':False,'serving_measured':False,'targets':256,'logit_hashes':{name:hashlib.sha256((directory/(name+'.bf16')).read_bytes()).hexdigest() for name in logits}}
    (directory/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
