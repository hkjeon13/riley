from pathlib import Path
import subprocess,json
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
def git(*args):return subprocess.check_output(['git',*args],cwd=s,text=True).strip()
assert git('rev-parse','HEAD')=='a33cdc8488122eebefad090cc493dec370193e49'
assert not git('status','--porcelain')
subprocess.run(['git','revert','--no-commit','a33cdc8488122eebefad090cc493dec370193e49','636376e67fdcc508552371628808fc21d2378c44'],cwd=s,check=True)
subprocess.run(['git','diff','--exit-code','06302d8d8396b8f2f4996fec8595bbfa1dcd7450','--'],cwd=s,check=True)
subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@local','commit','-m','Restore token-major values after packed serving trials showed no net gain'],cwd=s,check=True)
assert git('rev-parse','HEAD^{tree}')==git('rev-parse','06302d8^{tree}')
receipt={'source_commit':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}'),'matches_v52_tree_exactly':True,'reason':'V54 regressed serving; V55 recovered most loss but no material improvement over V52; keep experimental binaries and evidence, restore baseline before FFN batch'}
(r/'token-major-restore-v56.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
(r/'token-major-restore-v56.patch').write_bytes(subprocess.check_output(['git','show','--format=fuller','HEAD'],cwd=s))
