from pathlib import Path
import os,subprocess
r=Path('/tmp/riley-opt-260912');src=r/'prefill-shapes-source-v11'
p=src/'crates/riley-scheduler/src/execution.rs';s=p.read_text();a='    let mut local_tokens=Vec::new();';assert a in s;s=s.replace(a,'    let mut local_tokens=if compact_greedy && workspace.is_none(){reserve_vec(prepared.output_count,"V4 compact tokens").map_err(|e|IterationExecutionFailure::new(id,Some(ExecutionAbort::NotDispatched),e))?}else{Vec::new()};',1);p.write_text(s)
p=src/'crates/riley-scheduler/tests/v3_shared_owned_gpu.rs';s=p.read_text();s=s.replace('let mut peak_active=0;','let mut peak_active=0;let mut workspace=Vec::with_capacity(ROWS);let workspace_pointer=workspace.as_ptr();',1);s=s.replace('let downloaded=riley_scheduler::execution::execute_llama_iteration_variable_graph_mode(&authority,&mut session,greedy).unwrap();','let mut downloaded=if greedy{riley_scheduler::execution::execute_llama_iteration_variable_graph_greedy_workspace(&authority,&mut session,&mut workspace)}else{riley_scheduler::execution::execute_llama_iteration_variable_graph(&authority,&mut session)}.unwrap();if greedy{assert_eq!(workspace.capacity(),0);}',1);s=s.replace(' drop(authority);let result=downloaded.into_result',' if greedy{downloaded.restore_greedy_token_workspace(&mut workspace).unwrap();assert_eq!(workspace.as_ptr(),workspace_pointer);}\n drop(authority);let result=downloaded.into_result',1);p.write_text(s)
# Retain the integration probe as a repository-owned reproducible test artifact.
import shutil
shutil.copy2(r/'compact_result_probe_v46.cu',src/'kernels/tests/compact_shared_result_probe.cu')
env=os.environ.copy();env.update(PATH='/home/psyche/.cargo/bin:/data/riley-g04-cuda13/bin:'+env['PATH'],CUDA_HOME='/data/riley-g04-cuda13',CUDAToolkit_ROOT='/data/riley-g04-cuda13',CMAKE='/data/cmake-3.31.12/bin/cmake',CMAKE_BUILD_PARALLEL_LEVEL='4',CARGO_BUILD_JOBS='4',CARGO_TARGET_DIR=str(r/'prefill-shapes-target-v11'))
jobs=[('release-final',['cargo','build','--release','-p','riley-server','--features','cuda,server','--bin','riley']),('eligibility',['cargo','test','--release','-p','riley-server','--features','cuda,server','--lib','gpu_greedy_ineligibility']),('workspace-tests',['cargo','test','--release','-p','riley-scheduler','--features','cuda','--lib','workspace']),('owned-final',['python3',str(r/'run_compact_owned_v46.py')])]
for name,args in jobs:
 with (r/f'compact-v46-{name}.log').open('w') as log:subprocess.run(args,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
