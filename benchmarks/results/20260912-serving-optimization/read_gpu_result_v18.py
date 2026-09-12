from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs');s=p.read_text();a='            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);';b='''            if let Some(dir)=std::env::var_os("RILEY_V18_GPU_RESULTS") {
                let gpu=std::fs::read(std::path::PathBuf::from(dir).join(format!("gpu-result-{active}.bin"))).unwrap();
                assert_eq!(gpu,bytes);assert_eq!(validate_batch_result(&gpu,&e).unwrap().len(),active as usize);
            }
'''+a;assert a in s;p.write_text(s.replace(a,b))
