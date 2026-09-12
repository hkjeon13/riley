from pathlib import Path
import json,hashlib,tarfile
r=Path('/tmp/riley-opt-260912')
names=['v3-loaded-owner-v11.patch','v3-loaded-owner-build-v11.log','v3-loaded-gpu-v11.log','v3-loaded-rope-diagnostic-v11.log','v3-loaded-rope-reference-v11.log','v3-loaded-gpu-matched-rope-v11.log','v3-loaded-memcheck-v11.log','v3-loaded-fp32-v11.log','loaded-rope-diagnostic-v11.json']
names += [str(p.relative_to(r)) for p in (r/'loaded-rope-fixture-v11').iterdir() if p.is_file() and not p.is_symlink()]
manifest={'commit':'5272d2aab20966a9bb4efafa3d0c908acb042ae9','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'external':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'prefill-full-model-v11/weights.bin',r/'prefill-full-model-v11/requests.bin',r/'prefill-shapes-target-v11/release/deps/v3_recorder_decode_gpu-8f239812f3612ece']}}
(r/'v3-loaded-owner-v11-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(r/'v3-loaded-owner-v11.tar.gz','w:gz') as tar:
 for n in names+['v3-loaded-owner-v11-manifest.json']:tar.add(r/n,arcname=n)
print('exported',len(names),'files')
