from pathlib import Path
root=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=root/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();pos=s.index('    #[test] fn export_versioned')
s=s[:pos]+'''    fn maximum_prefill16()->Expectation<16> {
        let mut e=fixture_rows::<16>(InputStage::Prefill,1);
        e.rows[0].progress=Progress{prompt_tokens:2048,output_limit:128,context_tokens:4096,committed_tokens:0,input_tokens:1024,generated_index:0,stage:InputStage::Prefill};
        e.rows[0].input_tokens=(0..1024).map(|i|i*31%49152).collect();
        e.rows[0].physical_ids=(0..64).collect();e.rows[0].valid_tokens=vec![16;64];
        e.block_ownership=(0..64).map(|physical_id|BlockOwnership{physical_id,sequence_tag:1}).collect();e
    }
    #[test] fn maximum_prefill16_tokens_do_not_overlap_unused_descriptors() {
        let e=maximum_prefill16();let mut packet=vec![0xa5;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        assert!(packet[HEADER_BYTES+ROW_BYTES..Layout::<16>::TOKENS_OFFSET].iter().all(|&x|x==0));
        for(i,&token)in e.rows[0].input_tokens.iter().enumerate(){let at=Layout::<16>::TOKENS_OFFSET+i*4;assert_eq!(&packet[at..at+4],&token.to_le_bytes());}
        let mut result=vec![0;Layout::<16>::BATCH_RESULT_BYTES];let mut header=[0;128];result_identity_into(&mut header,&e,0).unwrap();result[..128].copy_from_slice(&header);
        assert_eq!(validate_batch_result(&result,&e).unwrap()[0].token,None);
        result[128]=1;assert!(validate_batch_result(&result,&e).is_err());
    }
'''+s[pos:]
s=s.replace('        for(stage,name)in[', '''        let e=maximum_prefill16();let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];encode_into(&mut packet,&e).unwrap();
        std::fs::write(dir.join("v4-prefill-max.bin"),packet).unwrap();
        for(stage,name)in[''')
p.write_text(s)
p=root/'kernels/tests/variable_wire16_packet_test.cpp';s=p.read_text().replace('assert(argc==11)','assert(argc==12)');p.write_text(s)
