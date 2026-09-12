from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-cuda/tests/v3_shared_recorder_gpu.rs';s=p.read_text().replace('record_v3_shared','record_v4_shared').replace('shared_recorder_owns_resources','sixteen_recorder_owns_resources')
for old,new in [('17536','30848'),('13440','26752'),('0x33444d52','0x34444d52'),('(4,3)','(4,4)'),('(24,8)','(24,16)'),('8*9*4096*4','16*9*4096*4'),('30*8*16*384','30*16*16*384'),('8*1152','16*1152'),('8*98304','16*98304'),('787456','1574912'),('allocate_pinned_host_buffer(1574912)','allocate_pinned_host_buffer(3149824)'),('new(8,49152','new(16,49152'),(',0,32,8)',',0,32,16)'),('p32(&mut p,28,8)','p32(&mut p,28,16)'),('[2usize,4,8]','[1usize,2,4,8,9,15,16]'),('((i+1)%8)','((i+1)%16)')]:s=s.replace(old,new)
s=s.replace('let mut wrong=p.clone();p32(&mut wrong,128+', '''let mut wrong=p.clone();p32(&mut wrong,0,0x33444d52);assert!(owner.replay_transfer(&wrong).is_err());assert!(owner.read_transfer(&mut out).is_err());
let mut wrong=p.clone();p32(&mut wrong,128+''')
s=s.replace('assert_eq!(&out[b+4..b+8],&1u32.to_le_bytes());', 'assert_eq!(&out[b+4..b+8],&1u32.to_le_bytes());assert_eq!(&out[b+124..b+128],&0x34524d52u32.to_le_bytes());')
s=s.replace('SHARED_NATIVE','SHARED16_NATIVE').replace('decode_rows=2,4,8','decode_rows=1,2,4,8,9,15,16')
(r/'crates/riley-cuda/tests/v4_shared_recorder_gpu.rs').write_text(s)
p=r/'crates/riley-server/src/main.rs';s=p.read_text();pos=s.index('    #[test]',s.index('mod tests'))
s=s[:pos]+'''    #[test]
    fn variable_v4_cli_selects_sixteen_row_profile_explicitly(){
        for capacity in ["1","4","8","16","32"] {
            let result=super::parse_arguments(["serve","--model","/tmp/model","--graph-numerics","variable-smol-v4","--execution-graph-policy","require","--max-active-sequences",capacity,"--batch-token-budget","512","--prefill-chunk-tokens","512"].map(std::ffi::OsString::from)).unwrap();
            let super::CliCommand::Serve(options)=result else{panic!("serve")};assert!(options.variable_graph&&options.variable_graph16);
        }
    }
'''+s[pos:];p.write_text(s)
