import pathlib,json,hashlib,collections
import pyarrow.parquet as pq
from tokenizers import Tokenizer
r=pathlib.Path('/tmp/riley-opt-260912');out=r/'variable-corpus-v11';out.mkdir(exist_ok=False)
t=Tokenizer.from_file('/data/riley-benchmark/20260827T051948Z-d7ad713a/model/tokenizer.json')
src=r/'shared-natural-v10/validation.parquet';items=[]
for row,text in enumerate(pq.read_table(src).column('text').to_pylist()):
 ids=t.encode(text,add_special_tokens=False).ids
 if 16<=len(ids)<=1024:
  key=hashlib.sha256(f'riley-variable-v11:{row}'.encode()).hexdigest()
  items.append((key,{'source_row':row,'prompt':text,'token_ids':ids,'prompt_tokens':len(ids),'output_tokens':[32,64,128][int(key[-8:],16)%3]}))
chosen=[v for _,v in sorted(items)[:256]];assert len(chosen)==256
(out/'requests.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+'\n')
lengths=sorted(x['prompt_tokens'] for x in chosen)
meta={'scope':'natural prose completion, not a representative chat distribution or general quality benchmark','source_revision':'b08601e04326c79dfdd32d625aee71d232d685c3','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'selection':'lowest256 SHA256 keys from full validation rows containing16..1024tokens, no truncation, output length selected independently from key','eligible':len(items),'requests':256,'prompt_min':min(lengths),'prompt_max':max(lengths),'prompt_median':(lengths[127]+lengths[128])/2,'prompt_buckets':dict(collections.Counter(next(b for b in [32,64,128,256,512,1024] if x<=b) for x in lengths)),'output_counts':dict(collections.Counter(x['output_tokens'] for x in chosen)),'corpus_sha256':hashlib.sha256((out/'requests.json').read_bytes()).hexdigest(),'performance_qualified':False}
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
