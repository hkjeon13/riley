from pathlib import Path
import hashlib,json,tarfile,subprocess
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';names=['native16-v38-receipt.json']
for pattern in ['native16-v38-*.log','primitive16-v38-*.log']:names.extend(p.name for p in r.glob(pattern))
(r/'native16-v38.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','4ee3d8667c304ea019322f45e9291a80d74c4ac1'],cwd=src));names.append('native16-v38.patch')
for n in ['decode_shared16.cuh','decode_shared16_attention.cuh','decode_shared16_model.cuh','decode_shared16_result.cuh']:(r/n).write_bytes((src/'kernels/src'/n).read_bytes());names.append(n)
for n in ['shared16_model_probe.cu','shared16_primitive_probe.cu','README_shared16.md']:(r/n).write_bytes((src/'kernels/tests'/n).read_bytes());names.append(n)
x={'commit':'4ee3d8667c304ea019322f45e9291a80d74c4ac1','files':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'serving_integrated':False}
(r/'native16-v38-manifest.json').write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(r/'native16-v38-export.tar.gz','w:gz') as t:
 for n in names+['native16-v38-manifest.json']:t.add(r/n,arcname=n)
print('exported',len(names),'files')
