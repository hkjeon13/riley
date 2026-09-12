from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs');s=p.read_text();a='            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);';b='''            if let Some(dir)=std::env::var_os("RILEY_V18_FIXTURES") {
                let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();let mut packet=vec![0;REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
                std::fs::write(dir.join(format!("request-{active}.bin")),packet).unwrap();std::fs::write(dir.join(format!("result-{active}.bin")),&bytes).unwrap();
            }
'''+a;assert a in s;p.write_text(s.replace(a,b))
