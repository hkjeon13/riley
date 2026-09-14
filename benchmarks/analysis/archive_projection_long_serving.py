"""Partition a completed long comparison into bounded, lossless per-lane archives."""
import argparse,hashlib,json,pathlib,tarfile
ap=argparse.ArgumentParser();ap.add_argument('source',type=pathlib.Path);ap.add_argument('output',type=pathlib.Path);args=ap.parse_args()
assert json.loads((args.source/'complete.json').read_text())=={'lanes':16,'all_complete':True}
preparation=json.loads((args.source/'preparation.json').read_text());assert preparation['retained']==8192 and preparation['warmup']==256
snapshot=args.source/'client-snapshot.py'
assert snapshot.is_file() and hashlib.sha256(snapshot.read_bytes()).hexdigest()==preparation['client_sha256']
lanes=[r['name'] for r in json.loads((args.source/'progress.json').read_text())];assert len(set(lanes))==16
assert all(json.loads((args.source/(name+'-exit.json')).read_text())['exit_code']==0 for name in lanes)
groups={'common':[]};groups.update({name:[] for name in lanes})
for path in sorted(args.source.iterdir()):
    assert path.is_file() and not path.is_symlink()
    assert path.suffix in ('.json','.log','.py'), 'unexpected artifact type'
    matching=[name for name in lanes if path.name.startswith(name+'-') or path.name==name+'.log'];assert len(matching)<=1
    groups[matching[0] if matching else 'common'].append(path)
args.output.mkdir()
manifest={}
for group,files in groups.items():
    target=args.output/(group+'.tar.gz')
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
        for path in files:tar.add(path,arcname=args.source.name+'/'+path.name,recursive=False)
    assert target.stat().st_size<64*1024**2,'archive exceeds publication size bound'
    manifest[target.name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'members':len(files)}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Archived all evidence in 17 bounded parts')
