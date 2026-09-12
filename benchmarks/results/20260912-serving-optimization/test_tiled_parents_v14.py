from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-cuda/tests/v3_recorder_gpu.rs');s=p.read_text();a='fn v3_recorder_head_and_partial_completion()->Result<(),Box<dyn std::error::Error>>{'
b=a+'verify_recorder(false)}\n#[test]\n#[ignore="requires CUDA; tiled parent extent/alias and lifecycle"]\nfn v3_tiled_recorder_parent_lifecycle()->Result<(),Box<dyn std::error::Error>>{verify_recorder(true)}\nfn verify_recorder(tiled:bool)->Result<(),Box<dyn std::error::Error>>{'
assert a in s;s=s.replace(a,b).replace('sizes[7]=sizes[7].max(9*4096*4);','sizes[7]=sizes[7].max(9*4096*4);\nif tiled {sizes.extend([1769472;90]);}')
s=s.replace('let mut head=context.prepare_gemm', 'if tiled {weights.extend(27..117);}\nlet mut head=context.prepare_gemm')
a='owner.record_v3_prefill(&devices,None,&weights,0,0,32,4)?;'
b='''if tiled {
 let mut alias=weights.clone();alias[273]=26;assert!(owner.record_v3_prefill(&devices,None,&alias,0,0,32,4).is_err());
 let mut extent=weights.clone();extent[273]=24;assert!(owner.record_v3_prefill(&devices,None,&extent,0,0,32,4).is_err());
 let mut duplicate=weights.clone();duplicate[274]=duplicate[273];assert!(owner.record_v3_prefill(&devices,None,&duplicate,0,0,32,4).is_err());
 assert!(owner.record_v3_prefill(&devices,None,&weights[..362],0,0,32,4).is_err());
}
'''+a
assert a in s;s=s.replace(a,b).replace('eprintln!("V3_RECORDER replays=3 cold_rejections=2','eprintln!("V3_RECORDER tiled={} extra_parent_rejections={} replays=3 cold_rejections=2').replace('allocation_zero=true");Ok(())','allocation_zero=true",tiled,if tiled {4}else{0});Ok(())');p.write_text(s)
