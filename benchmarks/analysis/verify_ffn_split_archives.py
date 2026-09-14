"""Verify all materialized response archive bytes before the semantic exporter."""
import pathlib,json,hashlib,tarfile,re,sys
r=pathlib.Path(sys.argv[1]);concurrency=8 if r.name.endswith('-c8') else 64 if r.name.endswith('-c64') else 32;base=f'ffn-split-serving-c{concurrency}-v1/'
manifest=json.loads((r/'evidence/manifest.json').read_text());assert len(manifest)==17
with tarfile.open(r/'evidence/common.tar.gz') as tar:
 material=json.load(tar.extractfile(base+'materialization.json'));assert material['all_files_equal']
 assert json.load(tar.extractfile(base+'execution.json'))==[{'name':'serving','exit_code':0}]
 assert json.load(tar.extractfile(base+'blender-restored.json'))['restored']
 prep=json.load(tar.extractfile(base+'preparation.json'))
 controller=tar.extractfile(base+'controller-snapshot.py').read()
 assert hashlib.sha256(controller).hexdigest()==prep['controller_sha256']
 assert controller==pathlib.Path(__file__).with_name('ffn_split_serving_screen.py').read_bytes()
seen=set();total=0
for name,meta in manifest.items():
 p=r/'evidence'/name
 assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 with tarfile.open(p) as tar:
  members=tar.getmembers();assert len(members)==meta['members']
  for m in members:
   assert m.isfile() and m.name.startswith(base) and m.name not in seen
   name=m.name[len(base):];assert '/' not in name and pathlib.Path(name).suffix in ('.json','.log','.py')
   seen.add(m.name);h=hashlib.sha256();count=0;tail=b''
   with tar.extractfile(m) as f:
    while chunk:=f.read(1024*1024):
     count+=len(chunk);h.update(chunk);data=tail+chunk
     assert not re.search(rb'(?:Bearer\s+[A-Za-z0-9._-]{20,}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|--token\s+[A-Za-z0-9._-]{20,}|-----BEGIN .*PRIVATE KEY)',data),name
     tail=data[-200:]
   assert count==m.size
   if name in material['files']:assert material['files'][name]=={'bytes':count,'sha256':h.hexdigest()}
   else:assert name in ('materialization.json','execution.json','blender-restored.json')
   total+=count
assert {base+name for name in material['files']}<=seen
(r/'archive-verification.json').write_text(json.dumps({'archives':17,'files':len(seen),'uncompressed_bytes':total,'all_archive_hashes_verified':True,'all_original_file_hashes_match_materialization':True,'controller_snapshot_verified':True,'lifecycle_and_restoration_passed':True,'credential_pattern_scan_passed':True},indent=2)+'\n')
print('archive integrity verified:',len(seen),'files',total,'bytes')
