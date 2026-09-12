from pathlib import Path
import hashlib,json,subprocess
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
(r/'wire16-v39.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=src))
files=[p for p in r.glob('wire16-v39-*.log')]+[r/'wire16-v39.patch',r/'blender-keep-stopped-public.json']+list((r/'wire16-v39-fixtures').glob('*.bin'))
for path in ['crates/riley-runtime/src/llama/multi_descriptor/variable_wire16.rs','kernels/tests/README_variable_wire16.md','kernels/tests/variable_wire16_packet_test.cpp']:
 p=src/path;out=r/('wire16-v39-'+p.name);out.write_bytes(p.read_bytes());files.append(out)
x={'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip(),'wire_only':True,'serving_integrated':False,'production_binary':'V37','rust_tests':11,'native_accepted':11,'native_rejected':156588,'native_sanitizers':['address','undefined'],'gpu_validation_new_wire':False,'benchmark_new_wire':False,'blender_keep_stopped':True}
p=r/'wire16-v39-receipt.json';p.write_text(json.dumps(x,indent=2)+'\n');files.append(p)
manifest={'files':[{'path':str(p.relative_to(r)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(files)]}
(r/'wire16-v39-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(len(files))
