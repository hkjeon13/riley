from pathlib import Path
import hashlib,json,os,shutil
names=['ffn-split-serving-c32-v1']
root=Path('/data/riley-serving-260913-recovery/tmpfs-preserved-20260914');root.mkdir(exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def manifest(p):
 result={}
 for f in sorted(p.rglob('*')):
  assert not f.is_symlink()
  if f.is_file():result[str(f.relative_to(p))]={'bytes':f.stat().st_size,'sha256':digest(f)}
 return result
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 assert not any(a.endswith(b'/projection_cta_serving_screen.py') or a.endswith(b'/ffn_split_serving_screen.py') or a.endswith(b'/ffn_adaptive_serving_screen.py') or a.endswith(b'/dense_wire_serving_screen.py') or a.endswith(b'/owner_index_serving_screen.py') or a.endswith(b'/kv_tile_census_screen.py') or Path(a.decode(errors='replace')).name.startswith('dense_wire_matrix') for a in argv),'measurement still live'
for name in names:
 src=Path('/dev/shm')/name;dst=root/name
 assert src.is_dir() and not src.is_symlink() and not dst.exists(),name
 before=manifest(src);assert before
 shutil.copytree(src,dst)
 assert manifest(dst)==before and manifest(src)==before,name
 for f in dst.rglob('*'):
  if f.is_file():
   with f.open('rb') as stream:os.fsync(stream.fileno())
 receipt={'source':str(src),'destination':str(dst),'files':before,'verified':True}
 rp=root/(name+'-receipt.json')
 with rp.open('x') as stream:json.dump(receipt,stream);stream.flush();os.fsync(stream.fileno())
 shutil.rmtree(src)
 print(name,len(before),sum(x['bytes'] for x in before.values()),flush=True)
print(Path('/proc/meminfo').read_text().splitlines()[:4],flush=True)
