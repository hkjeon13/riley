from pathlib import Path
b=Path('/tmp/riley-shared-integration-v10');s=(b/'crates/riley-scheduler/tests/multi_graph_authority_gpu.rs').read_text();a=s.index('#[test]');z=s.index('fn run_mode(',a)
s=s[:a]+'''#[test]
#[ignore = "requires qualified CUDA runtime and real SmolLM2 checkpoint"]
fn shared_four_rows_modes_cancel_and_reuse() -> Result { run_mode(true,4,true) }
#[test]
#[ignore = "requires qualified CUDA runtime and real SmolLM2 checkpoint"]
fn shared_eight_rows_modes_cancel_and_reuse() -> Result { run_mode(true,8,true) }
fn token(bytes: &[u8]) -> u32 {
    let mut best=f32::NEG_INFINITY;let mut id=0;
    for (i,b) in bytes.chunks_exact(2).enumerate() {
        let v=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16);
        assert!(v.is_finite());if v>best { best=v;id=i as u32; }
    } id
}
'''+s[z:]
s=s.replace('into_owned_multi_decode_graph(&context)', 'into_owned_shared_multi_decode_graph(&context)').replace('into_owned_decode_graph(&context)', 'into_owned_shared_multi_decode_graph(&context)')
s=s.replace('        let mut samples =', '''        let expected_download = execute_llama_iteration_multi_graph(&authority, &mut oracle, None)
            .map_err(|e| format!("full shared oracle execution failed: {e:?}"))?;
        let mut samples =''')
a=s.index('            let row = LlamaBatchRow::new(');z=s.index('            if !greedy {',a)
s=s[:a]+'''            let slot = work.output_slot().unwrap().get() as usize;
            let expected = &expected_download.logits_bf16_native()[slot*98304..(slot+1)*98304];
            let reference_token = token(expected);
'''+s[z:]
s=s.replace('oracle.greedy_token()?','reference_token').replace('        owner.confirm_scheduler_commit(id)?;', '        owner.confirm_scheduler_commit(id)?;\n        oracle.confirm_scheduler_commit(id)?;')
s=s.replace('    stream.close()?;', '    stream.close()?;\n    assert!(context.allocation_stats()?.is_zero());')
s=s.replace('live scheduler owner parity:', 'shared full/greedy owner parity:').replace('{rows} full-logit rows','{rows} compared output rows')
(b/'crates/riley-scheduler/tests/shared_output_modes_gpu.rs').write_text(s)
s=Path('/tmp/run_shared_diagnostic_v10.py').read_text().replace('shared_model_diagnostic_gpu','shared_output_modes_gpu').replace('model-diagnostic','output-modes').replace("'shared-v10-r2-'","'shared-v10-'");Path('/tmp/run_shared_modes_v10.py').write_text(s)
