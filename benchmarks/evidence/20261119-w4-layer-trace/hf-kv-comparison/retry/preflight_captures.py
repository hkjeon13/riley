import json
from pathlib import Path
import sys
from hashlib import sha256
from hf_kv_comparison import VOCAB
from replay_attention_mask import capture

base=Path(sys.argv[1])
reference=base/'w4-layer-trace-r3-20261119.qpoSLydn'
native=base/'w4-kv-mask-20261119.DEXef1dj'
fixtures=json.loads((reference/'fixtures/legacy.json').read_bytes())['baseline']
for case in (3,9):
    prompt=fixtures[case]
    _, first, last, pages, keys, values, data_sha256 = capture(native/f'native{case}',len(prompt['prompt_tokens'])-1)
    expected=json.loads((native/f'case{case}-replay.json').read_bytes())
    assert data_sha256 == expected['page_data_sha256'] and expected['prompt_sha256']==prompt['prompt_sha256']
    directory=reference/f'hf{case}/hf-rows'
    logits=(directory/'rows.bf16').read_bytes()[-VOCAB*2:]
    info=json.loads((directory/'metadata.json').read_bytes())
    assert len(logits)==VOCAB*2 and info['prompt_sha256']==expected['prompt_sha256']
    print(json.dumps({'case':case,'native_positions':[first,last],'kv_pages':len(pages),'kv_shape':keys.shape,'native_sha256':data_sha256,'hf_first_id':info['selected_id'],'hf_logits_sha256':sha256(logits).hexdigest()}))
