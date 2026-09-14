import os,json,subprocess,hashlib,shutil
from pathlib import Path
root=Path('/data/riley-serving-260913-recovery');out=root/'projection-cta-model-build-v1';out.mkdir()
env=os.environ.copy();env.update(PATH=str(root/'toolchain130/nvidia/cu13/bin')+':/data/cmake-3.31.12/bin:/home/psyche/.cargo/bin:/usr/bin:/bin',LD_LIBRARY_PATH=str(root/'toolchain130/nvidia/cu13/lib'),CARGO_TARGET_DIR=str(root/'target'),RILEY_FA3_SOURCE=str(root/'deps/flash-attention'),CARGO_BUILD_JOBS='4')
argv=['cargo','test','-p','riley-scheduler','--features','cuda','--test','decode_window_gpu','--release','--offline','--no-run','--message-format=json']
with (out/'build.jsonl').open('w') as log, (out/'build.log').open('w') as err:
 p=subprocess.run(argv,cwd=root/'source',env=env,stdout=log,stderr=err)
(out/'build-exit.json').write_text(json.dumps({'exit_code':p.returncode,'argv':argv})+'\n');print('build',p.returncode,flush=True)
assert p.returncode==0
artifacts=[json.loads(line) for line in (out/'build.jsonl').read_text().splitlines() if line.startswith('{')]
artifacts=[r for r in artifacts if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name')=='decode_window_gpu' and r.get('executable')]
assert len(artifacts)==1
binary=Path(artifacts[0]['executable']);dst=root/'projection-cta-model-test-v1';assert not dst.exists();shutil.copy2(binary,dst)
(out/'test-binary.json').write_text(json.dumps({'executable':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})+'\n')
print(str(dst),flush=True)
