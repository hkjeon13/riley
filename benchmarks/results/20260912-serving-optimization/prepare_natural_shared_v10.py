import pathlib,urllib.request,json,hashlib
import pyarrow.parquet as pq
from tokenizers import Tokenizer
r=pathlib.Path('/tmp/riley-opt-260912/shared-natural-v10');r.mkdir(exist_ok=False)
revision='b08601e04326c79dfdd32d625aee71d232d685c3';url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{revision}/wikitext-2-raw-v1/validation-00000-of-00001.parquet';data=urllib.request.urlopen(url,timeout=60).read();(r/'validation.parquet').write_bytes(data)
tok=Tokenizer.from_file('/data/riley-benchmark/20260827T051948Z-d7ad713a/model/tokenizer.json');eligible=[]
for i,text in enumerate(pq.read_table(r/'validation.parquet').column('text').to_pylist()):
 ids=tok.encode(text,add_special_tokens=False).ids
 if len(ids)>=160:
  key=hashlib.sha256(('riley-natural-v10-v1:'+str(i)).encode()).hexdigest();start=int(key[:16],16)%(len(ids)-159);eligible.append((key,{'row':i,'offset':start,'tokens':ids[start:start+160]}))
items=[v for _,v in sorted(eligible)[:64]];assert len(items)==64
(r/'corpus.json').write_text(json.dumps(items,indent=2)+'\n')
policy={'purpose':'predeclared finite-corpus numerical noninferiority screen; not final serving or general quality qualification','documents':64,'prompt_tokens':128,'scored_next_tokens_per_document':32,'forced_history':'dataset ground-truth next tokens, identical for original/shared/FP32','mean_nll_delta_upper_95_ci_nats':0.01,'max_document_mean_nll_delta_nats':0.05,'mean_kl_from_fp32_additive_slack':0.0001,'p99_kl_from_fp32_additive_slack':0.001,'bootstrap_unit':'document','bootstrap_seed':20260912,'bootstrap_repetitions':10000,'nonfinite_allowed':0,'top1_agreement':'diagnostic, not substituted for ground-truth loss','source_url':url,'source_revision':revision,'source_sha256':hashlib.sha256(data).hexdigest(),'corpus_sha256':hashlib.sha256((r/'corpus.json').read_bytes()).hexdigest(),'selection':'lowest64SHA256keys over validation rows with at least160tokens; one deterministic contiguous window per row; fixed before candidate evaluation'}
(r/'policy.json').write_text(json.dumps(policy,indent=2)+'\n');print(json.dumps({'eligible':len(eligible),'selected':len(items),'corpus_sha256':policy['corpus_sha256']}))
