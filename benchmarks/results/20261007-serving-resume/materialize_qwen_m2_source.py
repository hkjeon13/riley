"""Materialize exact Cargo/CUDA sources under a real filtered Git commit."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def materialize(archive,pack,receipt_path,destination):
    destination=destination.resolve()
    receipt=json.loads(receipt_path.read_bytes());files=receipt['files'];destination.mkdir()
    assert sha(archive)==receipt['archive_sha256'] and sha(pack)==receipt['pack_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len(members)==len(files)
        assert {m.name for m in members}==set(files)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        tar.extractall(destination)
    assert all(sha(destination/name)==digest for name,digest in files.items())
    def git(*args,**kwargs):
        return subprocess.check_output(['git','-C',str(destination),*args],**kwargs)
    git('init','--quiet')
    with pack.open('rb') as inp:
        subprocess.run(['git','-C',str(destination),'index-pack','--stdin','--promisor'],stdin=inp,stdout=subprocess.DEVNULL,check=True)
    git('update-ref','HEAD',receipt['source_commit'])
    git('hash-object','-w','--stdin-paths',input=('\n'.join(files)+'\n').encode())
    git('read-tree','HEAD')
    tracked=git('ls-files','-z').decode().rstrip('\0').split('\0')
    excluded=[name for name in tracked if name not in files]
    if excluded:git('update-index','--skip-worktree','-z','--stdin',input=('\0'.join(excluded)+'\0').encode())
    assert not git('status','--porcelain','--untracked-files=no'),'materialized source differs from commit'
    for name in files:
        expected=git('rev-parse','HEAD:'+name).decode().strip()
        actual=git('hash-object',str(destination/name)).decode().strip()
        assert actual==expected,'source blob differs from actual commit: '+name
    return {'source_commit':receipt['source_commit'],'verified_source_files':len(files),
        'code_worktree_clean':True,'excluded_nonbuild_files_skip_worktree':len(excluded),
        'scope':'Cargo workspace and all tracked CUDA files verified against true Git blobs; excluded docs/results not materialized'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('pack',type=Path);p.add_argument('receipt',type=Path);p.add_argument('destination',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=materialize(a.archive,a.pack,a.receipt,a.destination);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
