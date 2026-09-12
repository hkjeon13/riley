from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912');names=['serving-round45-analysis.json','serving-screen-round45.log','serving-screen-round44.log','round44-restoration-public.json','round45-restoration-public.json','blender-keep-stopped-public.json','fill-v35-analysis.json','fill-v35-run.log','fill-diagnostic-v34/build.json','variable-candidate-v33/build.json']
for directory in ['variable-serving-screen-round44','variable-serving-screen-round45','natural-fill-v35']:
 names.extend(str(p.relative_to(r)) for p in (r/directory).iterdir() if p.is_file() and p.name not in ['session-stop.log','session-restore.log'])
x={'source_commit':'214c8ed7309d00ea908c7e29e3c2ed962f8f945d','configuration_experiment':True,'accepted_natural_setting':{'batch_token_budget':512,'prefill_chunk_tokens':512},'fixed_chunk':128,'vllm_budget':512,'blender_restore_authorization_revoked':True,'files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'private_restoration_journals_exported':False}
(r/'serving-round45-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'serving-round45-export.tar.gz','w:gz') as t:
 for n in names+['serving-round45-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
