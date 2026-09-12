import pathlib,subprocess,hashlib,json,shutil
r=pathlib.Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11';head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip();assert head=='00c9da9c9c50d214c1f2f111c2ce8ab0d2e04d1e'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
files=['crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs','crates/riley-scheduler/src/execution.rs','crates/riley-server/src/engine.rs','crates/riley-runtime/src/llama/variable_session.rs'];saved={f:(src/f).read_bytes() for f in files};receipts={}
try:
 for phase,commit in [('before','0526d4f3d341a3aa815f806c6f2e6cbab8d86d07'),('after',head)]:
  for f in files:(src/f).write_bytes(subprocess.check_output(['git','show',commit+':'+f],cwd=src))
  subprocess.run(['python3',str(r/'profile_host_v25.py')],check=True)
  patch=subprocess.check_output(['git','diff'],cwd=src);(r/f'host-v30-{phase}-diagnostic.patch').write_bytes(patch)
  subprocess.run(['python3',str(r/'build_host_v30.py')],check=True)
  shutil.copy2(r/'host-v30-diagnostic-release.log',r/f'host-v30-{phase}-release.log')
  binary=r/'prefill-shapes-target-v11/release/riley';receipts[phase]={'base_commit':commit,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'diagnostic_patch_sha256':hashlib.sha256(patch).hexdigest()}
  subprocess.run(['python3',str(r/f'run_host_v30_{phase}.py')],check=True)
finally:
 for f,b in saved.items():(src/f).write_bytes(b)
 shutil.copy2(r/'variable-candidate-v30/riley',r/'prefill-shapes-target-v11/release/riley')
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=src)
 assert hashlib.sha256((r/'prefill-shapes-target-v11/release/riley').read_bytes()).hexdigest()=='6f7d248187d9b614fd0d27a85a6863d38b70464c2b69270551dd6c249bd43582'
 (r/'host-v30-diagnostic-builds.json').write_text(json.dumps({'phases':receipts,'production_restored':True},indent=2)+'\n')
subprocess.run(['python3',str(r/'analyze_host_v30.py')],check=True)
