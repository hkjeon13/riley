from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-scheduler/tests/v3_shared_owned_gpu.rs';s=p.read_text();a=s.index('fn run_packed_profile(');b=s.index('\n#[test]',a);t=s[a:b].replace('fn run_packed_profile(', 'fn run_mixed_profile(').replace('into_owned_variable_packed_session','into_owned_variable_mixed_session').replace('ExecutionShapePolicy::PackedPrefillDecode32','ExecutionShapePolicy::MixedPrefillDecode32').replace('let mut prefill_width=0;', 'let mut mixed_iterations=0;let mut prefill_width=0;').replace('prefill_width=prefill_width.max', 'mixed_iterations+=usize::from(!plan.prefill_items().is_empty()&&!plan.decode_items().is_empty());prefill_width=prefill_width.max').replace('assert!(prefill_width>=2,','assert!(mixed_iterations>0,"mixed iteration not exercised");assert!(prefill_width>=2,').replace('LOADED_PACKED rows_max={}', 'LOADED_MIXED rows_max={}')
s+='\n'+t+'''
#[test]
#[ignore="requires GPU; V7 mixed full-logit execution"]
fn loaded_v7_full32()->Result<(),Box<dyn std::error::Error>>{run_mixed_profile(32,32,2048,1024,512,false)}
#[test]
#[ignore="requires GPU; V7 mixed alternating compact and full"]
fn loaded_v7_compact32()->Result<(),Box<dyn std::error::Error>>{run_mixed_profile(32,32,2048,1024,512,true)}
#[test]
#[ignore="requires GPU; V7 mixed partial output and active decode"]
fn loaded_v7_partial()->Result<(),Box<dyn std::error::Error>>{run_mixed_profile(4,3,128,1024,256,true)}
''';p.write_text(s)
print('Added three real-model tests requiring actual mixed prefill/decode iterations')
