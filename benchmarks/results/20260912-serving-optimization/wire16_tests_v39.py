from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs')
s=p.read_text().replace('    fn fixture(stage:InputStage,active:u32)->Expectation {','    fn fixture(stage:InputStage,active:u32)->Expectation {fixture_rows::<8>(stage,active)}\n    fn fixture_rows<const ROWS:usize>(stage:InputStage,active:u32)->Expectation<ROWS> {').replace('physical_block_count:4096,max_active_rows:8,stage','physical_block_count:4096,max_active_rows:ROWS as u32,stage')
pos=s.index('    #[test] fn variable_shapes')
s=s[:pos]+'''    #[test] fn sixteen_row_contract_rejects_old_versions_and_binds_all_bytes() {
        assert_eq!(Layout::<8>::REQUEST_BYTES,REQUEST_BYTES);
        assert_eq!(Layout::<8>::BATCH_RESULT_BYTES,BATCH_RESULT_BYTES);
        assert_eq!(Layout::<16>::TOKENS_OFFSET,26752);
        assert_eq!(Layout::<16>::REQUEST_BYTES,30848);
        assert_eq!(Layout::<16>::BATCH_RESULT_BYTES,1574912);
        assert!(Layout::<16>::REQUEST_BYTES<Layout::<16>::OUTPUT_OFFSET);
        assert_eq!(Layout::<16>::STAGING_BYTES,3149824);
        for stage in [InputStage::Prefill,InputStage::Decode] {
            let e=fixture_rows::<16>(stage,if stage==InputStage::Prefill{1}else{16});
            let mut packet=vec![0;Layout::<16>::REQUEST_BYTES];let mut scratch=packet.clone();
            encode_into(&mut packet,&e).unwrap();
            for offset in 0..packet.len() {packet[offset]^=1;assert!(validate_packet(&packet,&mut scratch,&e).is_err(),"offset{offset}");packet[offset]^=1;}
            validate_packet(&packet,&mut scratch,&e).unwrap();
            if stage==InputStage::Prefill {
                assert!(packet[HEADER_BYTES+ROW_BYTES..Layout::<16>::TOKENS_OFFSET].iter().all(|&x|x==0));
                assert_eq!(&packet[Layout::<16>::TOKENS_OFFSET..Layout::<16>::TOKENS_OFFSET+4],&33u32.to_le_bytes());
            }
            let old=fixture(stage,if stage==InputStage::Prefill{1}else{8});
            assert!(validate_packet(&packet,&mut vec![0;REQUEST_BYTES],&old).is_err());
            let mut old_packet=vec![0;REQUEST_BYTES];encode_into(&mut old_packet,&old).unwrap();
            assert!(validate_packet(&old_packet,&mut scratch,&e).is_err());
            // Matching extents cannot disguise old magic/version, even for one row.
            u32_at(&mut packet,0,MAGIC);u32_at(&mut packet,4,3);
            assert!(validate_packet(&packet,&mut scratch,&e).is_err());
        }
    }
    #[test] fn sixteen_row_results_bind_upper_rows_and_reject_old_single_result_magic() {
        for active in [1,2,4,8,9,15,16] {
            let e=fixture_rows::<16>(InputStage::Decode,active);
            let mut bytes=vec![0;Layout::<16>::BATCH_RESULT_BYTES];
            for i in 0..active as usize {
                let mut h=[0;128];result_row_identity_into(&mut h,&e,i,0).unwrap();
                bytes[i*RESULT_BYTES..i*RESULT_BYTES+128].copy_from_slice(&h);
            }
            let rows=validate_batch_result(&bytes,&e).unwrap();assert_eq!(rows.len(),active as usize);
            for(i,row)in rows.iter().enumerate(){assert_eq!(row.output_slot,active-1-i as u32);assert_eq!(row.token,Some(0));}
            for row in 0..active as usize {for offset in 0..128 {
                let at=row*RESULT_BYTES+offset;bytes[at]^=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]^=1;
            }}
            if active<16 {for at in [active as usize*RESULT_BYTES,bytes.len()-1] {bytes[at]=1;assert!(validate_batch_result(&bytes,&e).is_err());bytes[at]=0;}}
            let at=(active as usize-1)*RESULT_BYTES+128;
            bytes[at..at+2].copy_from_slice(&0x7f80u16.to_le_bytes());assert!(validate_batch_result(&bytes,&e).is_err());
        }
        let e=fixture_rows::<16>(InputStage::Decode,1);let old=fixture(InputStage::Decode,1);
        let mut bytes=vec![0;RESULT_BYTES];let mut h=[0;128];result_identity_into(&mut h,&e,0).unwrap();bytes[..128].copy_from_slice(&h);
        validate_result(&bytes,&e).unwrap();assert!(validate_result(&bytes,&old).is_err());
        result_identity_into(&mut h,&old,0).unwrap();bytes[..128].copy_from_slice(&h);
        validate_result(&bytes,&old).unwrap();assert!(validate_result(&bytes,&e).is_err());
    }
    #[test] fn sixteen_row_invalid_authority_and_capacity_never_mutate_packet() {
        let e=fixture_rows::<16>(InputStage::Decode,16);let mut cases=vec![];
        let mut x=e.clone();x.max_active_rows=8;cases.push(x);
        let mut x=e.clone();x.max_active_rows=32;cases.push(x);
        let mut x=e.clone();x.rows[15].physical_ids[0]=x.rows[0].physical_ids[0];cases.push(x);
        let mut x=e.clone();x.rows[15].cookie=x.rows[0].cookie;cases.push(x);
        let mut x=e.clone();x.rows[15].output_slot=x.rows[0].output_slot;cases.push(x);
        let mut x=e.clone();x.replay_id=4;cases.push(x);
        let mut x=e.clone();x.rows.push(x.rows[0].clone());cases.push(x);
        for x in cases {let mut packet=vec![0xa5;Layout::<16>::REQUEST_BYTES];assert!(encode_into(&mut packet,&x).is_err());assert!(packet.iter().all(|&x|x==0xa5));}
        let unsupported=fixture_rows::<32>(InputStage::Decode,1);let mut packet=vec![0xa5;Layout::<32>::REQUEST_BYTES];
        assert!(encode_into(&mut packet,&unsupported).is_err());assert!(packet.iter().all(|&x|x==0xa5));
        let mut old=fixture(InputStage::Decode,8);old.max_active_rows=16;
        assert!(validate(&old).is_err());
    }
'''+s[pos:];p.write_text(s)
