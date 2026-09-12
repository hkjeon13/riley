"""Build/run exact-M1 strided admission diagnostic against pinned native objects."""
from pathlib import Path
import hashlib,json,os,subprocess
ROOT=Path('/tmp/riley-opt-260912')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def main():
 out=ROOT/'multisequence-strided-graph-v1';out.mkdir(exist_ok=False)
 prior_path=ROOT/'multisequence-projection-candidates-v1/build/compile.json';prior=json.loads(prior_path.read_text())
 source=ROOT/'batch8-source';gemm=source/'kernels/src/gemm.cu'
 assert sha(gemm)=='0ab4d7af6a7591d000212e450a9823ac0eb75fad5dd3df761ae95fe13deb9d67'
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='8329c1aeec6e013f581128888c536e15f8bf7300'
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True).strip()
 pins={}
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
    assert sha(x['path'])==x['sha256'];pins[x['path']]=x['sha256']
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(prior['link']);walk(prior['cxx']);walk(prior['nvcc'])
 native=ROOT/'multisequence_strided_graph_v1.cpp';binary=out/'probe'
 libs=prior['link']['libraries'];argv=[prior['cxx']['path'],'-std=c++17','-O2','-x','c++','-I'+str(source/'kernels/include'),'-I'+str(source/'kernels/src'),'-I/data/riley-g04-cuda13/include',str(native),'-x','none',prior['link']['archive']['path']]
 argv += [libs[k]['path'] for k in ('cublaslt','cudart','driver')]
 if 'nvml' in libs:argv.append(libs['nvml']['path'])
 argv += ['-pthread','-ldl','-o',str(binary)]
 pins[str(gemm)]=sha(gemm);pins[str(native)]=sha(native);pins[str(Path(__file__))]=sha(__file__)
 cases_path=ROOT/'multisequence-projection-candidates-v1/cases.tsv'
 fixture_paths={cases_path,ROOT/'multisequence-projection-candidates-v1/fixtures.json'}
 for line in cases_path.read_text().splitlines():
  fields=line.split('\t');assert len(fields)==11;fixture_paths.update(Path(v) for v in fields[-2:])
 for p in fixture_paths:pins[str(p)]=sha(p)
 with (out/'build.log').open('x') as f:subprocess.run(argv,check=True,stdout=f,stderr=f)
 env=dict(os.environ);assert not env.get('LD_PRELOAD')
 env['LD_LIBRARY_PATH']=str(ROOT/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib';env['CUDA_VISIBLE_DEVICES']='0'
 with (out/'native.jsonl').open('x') as f,(out/'stderr.log').open('x') as e:subprocess.run([str(binary),"--cases",str(cases_path),"--device","0"],env=env,stdout=f,stderr=e,check=True)
 rows=[json.loads(line) for line in (out/'native.jsonl').read_text().splitlines()]
 assert rows[0]['uuid']=='9087e4256acab722b8c9cc0423b39fb0'
 cases=[r for r in rows if r['kind']=='case'];assert len(cases)==2178
 assert [r['case_id'] for r in cases]==list(range(2178))
 assert all(r['executed'] and r['guards_intact'] and r['inputs_unchanged'] and r['weights_unchanged'] and r['case_allocations_freed'] and r['plan_metadata_unchanged'] for r in cases)
 assert all(r['retained_graph_replays']==3 and r['original_zero_original_checked'] for r in cases)
 assert rows[-1]['completed'] and rows[-1]['all_resources_closed'] and not rows[-1]['performance_measured']
 assert all(sha(p)==h for p,h in pins.items())
 result=dict(completed=True,source_commit='8329c1aeec6e013f581128888c536e15f8bf7300',source_files=pins,build_argv=argv,binary=ref(binary),raw=ref(out/'native.jsonl'),build_log=ref(out/'build.log'),stderr=ref(out/'stderr.log'),prior_compile=ref(prior_path),cases=cases,summary=rows[-1],gemm_executed=True,performance_measured=False)
 with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'completed':True,'cases':2178,'exact':rows[-1]['all_outputs_exact'],'mismatches':sum(r['mismatches'] for r in cases),'receipt':ref(out/'receipt.json')}))
if __name__=='__main__':main()
