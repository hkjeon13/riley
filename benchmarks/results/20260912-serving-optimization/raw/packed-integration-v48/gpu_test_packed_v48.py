from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-scheduler/tests/v3_shared_owned_gpu.rs';s=p.read_text();a=s.index('fn run_shared_profile_mode<const ROWS:usize>');b=s.index('\n\n#[test]',a);t=s[a:b]
t=t.replace('fn run_shared_profile_mode<const ROWS:usize>', 'fn run_packed_profile').replace('let root=PathBuf','const ROWS:usize=32;\nlet root=PathBuf',1)
a=t.index('let mut session=if compact');b=t.index(';assert_eq!',a);t=t[:a]+'let mut session=executor.into_owned_variable_packed_session(&context,capacity,compact)?'+t[b:]
t=t.replace('iteration_token_budget:chunk', 'iteration_token_budget:capacity as usize')
a=t.index('if ROWS==32{riley_scheduler::ExecutionShapePolicy::VariablePrefillDecode32}');b=t.index(')?;',a);t=t[:a]+'riley_scheduler::ExecutionShapePolicy::PackedPrefillDecode32'+t[b:]
t=t.replace('let mut peak_active=0;', 'let mut prefill_width=0;let mut peak_active=0;')
t=t.replace('widest=widest.max(plan.decode_items().len());', 'prefill_width=prefill_width.max(plan.prefill_items().len());assert!(plan.prefill_items().len()<=4);assert!(plan.total_tokens()<=capacity as usize);widest=widest.max(plan.decode_items().len());')
t=t.replace('assert_eq!(peak_active,active.min(request_count));', 'assert!(prefill_width>=2,"packed execution was not exercised");assert_eq!(peak_active,active.min(request_count));')
t=t.replace('LOADED_SHARED rows_max={}', 'LOADED_PACKED rows_max={}')
s+='\n'+t+'''
#[test]
#[ignore="requires GPU; V6 packed full logits and32 decode owners"]
fn loaded_v6_full32()->Result<(),Box<dyn std::error::Error>>{run_packed_profile(32,32,2048,1024,512,false)}
#[test]
#[ignore="requires GPU; V6 packed alternating full and compact"]
fn loaded_v6_compact32()->Result<(),Box<dyn std::error::Error>>{run_packed_profile(32,32,2048,1024,512,true)}
#[test]
#[ignore="requires GPU; V6 mixed published and partial prefill"]
fn loaded_v6_partial()->Result<(),Box<dyn std::error::Error>>{run_packed_profile(4,3,128,1024,256,true)}
''';p.write_text(s)
print('Added V6 real-model session tests for full/compact32 and mixed partial owners')
