from pathlib import Path
b=Path('/tmp/riley-shared-integration-v10');p=b/'crates/riley-scheduler/tests/multi_graph_authority_gpu.rs';s=p.read_text();prefix=s[:s.index('#[test]')];body=s[s.index('fn run(cancel_and_replace: bool, capacity: usize)'):];s=prefix+'''#[test]
#[ignore = "requires qualified CUDA runtime, checkpoint and fresh RILEY_SHARED_DUMP directory"]
fn shared_rows_common_prefix_model_diagnostic() -> Result { run(false,8) }
'''+body
s=s.replace('?.into_owned_multi_decode_graph(&context)?','?.into_owned_shared_multi_decode_graph(&context)?',1)
s=s.replace('    let mut cancelled_blocks', '''    let dump = std::path::PathBuf::from(std::env::var_os("RILEY_SHARED_DUMP").ok_or("dump path missing")?);
    std::fs::create_dir(&dump)?;
    use std::io::Write;
    let mut records = std::fs::OpenOptions::new().create_new(true).write(true).open(dump.join("rows.jsonl"))?;
    let mut cancelled_blocks''')
a=s.index('            if !greedy { assert_eq!(');z=s.index('            samples[slot] =',a)
s=s[:a]+'''            assert!(!greedy);
            let actual = &downloaded.logits_bf16_native()[slot*98304..(slot+1)*98304];
            let name = format!("r{}-i{}", work.request_id().get(), now);
            for (suffix,bytes) in [("shared",actual),("original",expected)] {
                let mut f = std::fs::OpenOptions::new().create_new(true).write(true).open(dump.join(format!("{name}-{suffix}.bin")))?;
                f.write_all(bytes)?;
            }
            let mut se=0.0f64;let mut norm=0.0f64;let mut max=0.0f64;let mut changed=0usize;
            for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let af=f32::from_bits(u32::from(u16::from_le_bytes([a[0],a[1]]))<<16) as f64;
                let bf=f32::from_bits(u32::from(u16::from_le_bytes([b[0],b[1]]))<<16) as f64;
                assert!(af.is_finite()&&bf.is_finite());let d=(af-bf).abs();se+=d*d;norm+=bf*bf;max=max.max(d);changed+=a!=b;
            }
            writeln!(records,"{{\\"name\\":\\"{name}\\",\\"request\\":{},\\"input_tokens\\":{:?},\\"target_length\\":{},\\"changed_logits\\":{changed},\\"relative_l2_to_original\\":{},\\"max_abs_to_original\\":{max},\\"original_token\\":{}}}",work.request_id().get(),work.input_tokens(),work.target_logical_length(),(se/norm.max(1e-30)).sqrt(),oracle.greedy_token()?)?;
'''+s[z:]
s=s.replace('live scheduler owner parity:', 'shared model diagnostic (not numerical acceptance):')
p=b/'crates/riley-scheduler/tests/shared_model_diagnostic_gpu.rs';p.write_text(s)
