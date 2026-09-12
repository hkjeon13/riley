from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/graph_decode_full.rs';s=p.read_text();needle='            include_bytes!("multi_descriptor/variable_wire.rs").as_slice()]';assert needle in s;s=s.replace(needle,'            include_bytes!("multi_descriptor/result_scan.rs").as_slice(),\n'+needle);p.write_text(s)
p=r/'crates/riley-runtime/src/llama/multi_descriptor/result_scan.rs';s=p.read_text();a=s.index('    #[test] fn every_bf16')
s=s[:a]+'''    #[test]
    #[ignore="CPU scan timing diagnostic, not serving qualification"]
    fn scan_timing_diagnostic(){
        use std::{hint::black_box,time::Instant};
        let bytes:Vec<u8>=(0..49152).flat_map(|i|(((i*139+73)%0x7f80)as u16).to_le_bytes()).collect();
        for run in 0..4 {
            for vector in if run%2==0 {[false,true]}else{[true,false]} {
                let start=Instant::now();
                for _ in 0..4096 {let result=if vector {finite_argmax(black_box(&bytes))}else{super::super::variable_wire::scalar_finite_argmax(black_box(&bytes))};black_box(result.unwrap());}
                eprintln!("SCAN vector={} iterations=4096 ns_per_row={}",vector,start.elapsed().as_nanos()/4096);
            }
        }
    }
'''+s[a:];p.write_text(s)
