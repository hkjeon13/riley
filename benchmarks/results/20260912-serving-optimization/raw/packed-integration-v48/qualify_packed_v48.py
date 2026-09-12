from pathlib import Path
import os,subprocess,json
r=Path('/tmp/riley-opt-260912');s=r/'prefill-shapes-source-v11'
p=s/'crates/riley-runtime/src/llama/executor/config.rs';t=p.read_text().replace('        self.variable_graph = false;', '        self.variable_graph = false;\n        self.packed_prefill = false;');p.write_text(t)
p=s/'crates/riley-server/src/main.rs';t=p.read_text();a=t.index('    #[test]',t.index('fn variable',t.index('mod tests')) if 'fn variable' in t[t.index('mod tests'):] else t.index('mod tests')) if False else 0
# Add an explicit parser qualification next to the existing V5 test.
a=t.index('    fn ',t.index('"variable-smol-v5"',t.index('mod tests'))) if False else 0
marker='    #[test]\n';pos=t.index(marker,t.index('mod tests'))
t=t[:pos]+'''    #[test]
    fn packed_v6_profile_preserves_aggregate_budget(){
        let result=super::parse_arguments(["serve","--model","/tmp/model","--graph-numerics","variable-smol-v6","--execution-graph-policy","require","--max-active-sequences","32","--batch-token-budget","1024","--prefill-chunk-tokens","512"].map(std::ffi::OsString::from)).unwrap();
        let super::CliCommand::Serve(options)=result else{panic!("serve")};assert!(options.variable_graph&&options.variable_graph32&&options.packed_prefill&&!options.variable_graph16);assert_eq!(options.prefill_chunk_tokens,512);
    }
'''+t[pos:];p.write_text(t)
for old,new in [('run_v5_http_v47.py','run_v6_http_v48.py'),('run_v5_fallback_v47.py','run_v6_fallback_v48.py')]:
 t=(r/old).read_text().replace('v5-','v6-').replace('v47','v48').replace('variable-smol-v5','variable-smol-v6').replace("'--batch-token-budget','512'","'--batch-token-budget','1024'");(r/new).write_text(t)
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'),CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH=str(r/'driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu')+':/data/riley-g04-cuda13/lib',RILEY_REAL_CHECKPOINT='/data/riley-benchmark/20260827T051948Z-d7ad713a/model',RILEY_V3_MODEL_FIXTURE=str(r/'loaded-rope-fixture-v11'))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],env=env,text=True).strip(),'GPU occupied'
jobs=[('final-build',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('cli',['cargo','test','--release','-p','riley-server','--features','cuda,server','--bin','riley','packed_v6_profile','--','--nocapture']),('owned',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--test','v3_shared_owned_gpu','--','--ignored','--test-threads=1','--nocapture']),('memcheck',['/data/cuda-12.8.1/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(r/'prefill-shapes-target-v11/release/deps/v3_shared_owned_gpu-bb1a782b521f0de1'),'loaded_v6','--ignored','--test-threads=1','--nocapture']),('http',['python3',str(r/'run_v6_http_v48.py')])]
for name,args in jobs:
 with (r/f'packed-v48-{name}.log').open('w') as log:subprocess.run(args,cwd=s,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
for backend in ['cpu','gpu-greedy']:
 env['V46_BACKEND']=backend
 with (r/f'packed-v48-fallback-{backend}.log').open('w') as log:subprocess.run(['python3',str(r/'run_v6_fallback_v48.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS fallback',backend,flush=True)
a=json.loads((r/'v6-fallback-ordered-v48-cpu/responses.json').read_text());b=json.loads((r/'v6-fallback-ordered-v48-gpu-greedy/responses.json').read_text());assert a==b
(r/'packed-v48-fallback-comparison.json').write_text(json.dumps({'exact_match':True,'responses_per_backend':len(a),'request_ids_exact':True},indent=2))
print('PASS fallback parity',flush=True)
