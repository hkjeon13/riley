import hashlib,json,pathlib,subprocess
root=pathlib.Path('/tmp/riley-opt-260912'); source=root/'multisequence-shared-source-v10'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
assert not git('status','--porcelain','--untracked-files=all')
assert 'Finished `release`' in (root/'shared-v10-optin-r2-server-build.log').read_text()
binary=root/'multisequence-shared-target-v10/release/riley'
receipt=json.loads((root/'mixed-p128-nsys-v10/receipt.json').read_text())
assert sha(binary)==receipt['binary_sha256']
out=root/'multisequence-candidate-v10';out.mkdir()
frozen=out/'riley'
with frozen.open('xb') as f:f.write(binary.read_bytes())
frozen.chmod(0o755)
files={p:sha(source/p) for p in git('ls-files').splitlines() if (source/p).is_file()}
x={'source_root':str(source),'source_commit':git('rev-parse','HEAD'),'source_files':files,'binaries':{str(frozen):sha(frozen)},'build_log':{'path':str(root/'shared-v10-optin-r2-server-build.log'),'sha256':sha(root/'shared-v10-optin-r2-server-build.log')},'correctness_receipts':{str(root/p):sha(root/p) for p in ['mixed-p128-nsys-v10/receipt.json','shared-v10-output-modes.log','shared-v10-native-contract.log','shared-natural-v10/policy.json','shared-natural-v10/summary.json']},'performance_qualified':False}
(out/'build.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'source_commit':x['source_commit'],'binary_sha256':sha(frozen),'build_sha256':sha(out/'build.json')}))
