from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();pos=s.index('    fn maximum_prefill16()')
s=s[:pos]+'''
    fn mixed_fixture(prefills:u32,decodes:u32,chunk:u32)->Expectation<32>{
        let mut e=packed_fixture(prefills,chunk);e.mixed_execution=true;e.stage=if prefills>0{InputStage::Prefill}else{InputStage::Decode};
        let decode=fixture_rows::<32>(InputStage::Decode,decodes);
        for mut row in decode.rows{row.sequence_tag+=u64::from(prefills);row.cookie+=u64::from(prefills);for p in &mut row.physical_ids{*p+=prefills*128;}e.rows.push(row);}
        e.block_ownership=e.rows.iter().flat_map(|r|r.physical_ids.iter().map(move |&physical_id|BlockOwnership{physical_id,sequence_tag:r.sequence_tag})).collect();
        let mut published=0;let mut partial=e.rows.iter().filter(|r|r.progress.validate().unwrap().logits_input_row.is_some()).count() as u32;
        for row in &mut e.rows{row.output_slot=if row.progress.validate().unwrap().logits_input_row.is_some(){let n=published;published+=1;n}else{let n=partial;partial+=1;n};}e
    }
    #[test] fn mixed_v7_stage_tiles_and_result_identity(){
        for (prefills,decodes,chunk) in [(0,32,1),(1,31,128),(4,28,249),(4,0,256),(2,3,31)] {
            let e=mixed_fixture(prefills,decodes,chunk);let mut p=vec![0;MIXED_REQUEST_BYTES];encode_into(&mut p,&e).unwrap();
            let get=|at|u32::from_le_bytes(p[at..at+4].try_into().unwrap());assert_eq!(get(0),0x37444d52);assert_eq!(get(4),7);assert_eq!(get(16),if prefills==0{1}else if decodes==0{0}else{2});
            let mut tile=0;for (owner,row) in e.rows.iter().enumerate(){let b=128+owner*ROW_BYTES;assert_eq!(get(b+72),stage_number(row.progress.stage));if prefills>0{assert_eq!(get(b+68),tile);let n=row.progress.input_tokens;for local in 0..if n<32{n}else{n.div_ceil(8)}{assert_eq!(get(Layout::<32>::REQUEST_BYTES+tile as usize*4),((owner as u32)<<16)|local);tile+=1;}}}
            assert_eq!(get(96),tile);let mut scratch=vec![0;p.len()];for at in 0..p.len(){p[at]^=0x80;assert!(validate_packet(&p,&mut scratch,&e).is_err(),"byte{at}");p[at]^=0x80;}
            let bytes=compact_fixture(&e);assert_eq!(validate_compact_result(&bytes,&e).unwrap().len(),(prefills+decodes)as usize);
            let mut wrong=e.clone();wrong.mixed_execution=false;assert!(validate_compact_result(&bytes,&wrong).is_err());
            if let Some(dir)=std::env::var_os("RILEY_V49_WIRE_FIXTURES"){let dir=std::path::PathBuf::from(dir);std::fs::create_dir_all(&dir).unwrap();std::fs::write(dir.join(format!("v7-{prefills}-{decodes}-{chunk}.bin")),p).unwrap();}
        }
    }
    #[test] fn mixed_v7_over_budget_and_stage_fail_before_write(){
        let e=mixed_fixture(4,28,250);let mut p=vec![0xa5;MIXED_REQUEST_BYTES];assert!(encode_into(&mut p,&e).is_err());assert!(p.iter().all(|&x|x==0xa5));
        let mut e=mixed_fixture(1,3,128);e.stage=InputStage::Decode;assert!(validate(&e).is_err());e.stage=InputStage::Prefill;e.packed_prefill=false;assert!(validate(&e).is_err());
        let e=mixed_fixture(1,3,128);assert!(encode_into(&mut vec![0;Layout::<32>::REQUEST_BYTES],&e).is_err());
    }
'''+s[pos:];p.write_text(s)
p=r/'crates/riley-scheduler/src/authority.rs';s=p.read_text();a=s.index('    #[test]\n    fn packed_policy_batches_prefill');b=s.index('    #[test]',a+15);t=s[a:b].replace('packed_policy_batches_prefill_and_preserves_retry_partial_cancel','mixed_policy_budget_stage_partial_retry_and_cancel').replace('ExecutionShapePolicy::PackedPrefillDecode32','ExecutionShapePolicy::MixedPrefillDecode32').replace('mixed_execution:false','mixed_execution:true').replace('let mut prefill_width=0;', 'let mut prefill_width=0;let mut saw_mixed=false;').replace('assert!(plan.prefill_items().is_empty() || plan.decode_items().is_empty());','saw_mixed|=!plan.prefill_items().is_empty()&&!plan.decode_items().is_empty();').replace('variable_wire::Layout::<32>::REQUEST_BYTES','variable_wire::MIXED_REQUEST_BYTES').replace('assert!(partial && retried && widest>=3 && prefill_width>=3);','assert!(partial && retried && widest>=3 && prefill_width>=3 && saw_mixed);');s=s[:b]+t+s[b:];p.write_text(s)
print('Added V7 wire and mixed scheduler regression tests')
