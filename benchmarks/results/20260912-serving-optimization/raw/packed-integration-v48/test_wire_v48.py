from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11')
p=r/'crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs';s=p.read_text();s=s.replace('    let (mut tags,mut cookies,mut slots,mut used)=', '    let published=e.rows.iter().filter(|r|r.progress.committed_tokens.checked_add(r.progress.input_tokens)==Some(r.progress.prompt_tokens) || e.stage==InputStage::Decode).count();\n    let (mut tags,mut cookies,mut slots,mut used)=')
s=s.replace('        let v=row.progress.validate()?;', '''        let v=row.progress.validate()?;
        if e.packed_prefill {check((row.output_slot as usize<published)==v.logits_input_row.is_some(),"slot","published slots must precede internal partial slots")?;}''',1)
# Rust cast parser needs parentheses before comparison.
s=s.replace('row.output_slot as usize<published','(row.output_slot as usize)<published')
t='''
    fn packed_fixture(active:u32,chunk:u32)->Expectation<32>{
        let mut e=fixture_rows::<32>(InputStage::Prefill,active);e.packed_prefill=true;
        e.block_ownership.clear();
        let published=(active+1)/2;let(mut output,mut partial)=(0,published);
        for (i,row) in e.rows.iter_mut().enumerate(){
            let committed=13*i as u32;let target=committed+chunk;
            row.progress=Progress{prompt_tokens:target+if i%2==0{0}else{7},output_limit:32,context_tokens:4096,committed_tokens:committed,input_tokens:chunk,generated_index:0,stage:InputStage::Prefill};
            row.input_tokens=(0..chunk).map(|j|(i as u32*97+j)%49152).collect();
            row.output_slot=if i%2==0{let n=output;output+=1;n}else{let n=partial;partial+=1;n};
            let v=row.progress.validate().unwrap();row.physical_ids=(0..v.live_pages).map(|j|i as u32*128+j).collect();row.valid_tokens=vec![16;v.live_pages as usize];*row.valid_tokens.last_mut().unwrap()=v.last_page_tokens;
            e.block_ownership.extend(row.physical_ids.iter().map(|&physical_id|BlockOwnership{physical_id,sequence_tag:row.sequence_tag}));
        }e
    }
    #[test] fn packed_v6_offsets_publication_and_corruption(){
        let export=std::env::var_os("RILEY_V48_WIRE_FIXTURES").map(std::path::PathBuf::from);
        if let Some(d)=&export{std::fs::create_dir_all(d).unwrap();}
        for active in 1..=4 {for chunk in [1,73,128,256] {
            let e=packed_fixture(active,chunk);let mut p=vec![0;Layout::<32>::REQUEST_BYTES];encode_into(&mut p,&e).unwrap();
            assert_eq!(u32::from_le_bytes(p[36..40].try_into().unwrap()),active*chunk);
            let mut offset=0;
            for (i,row) in e.rows.iter().enumerate(){let b=HEADER_BYTES+i*ROW_BYTES;assert_eq!(&p[b+64..b+68],&(offset as u32).to_le_bytes());for &token in &row.input_tokens{let at=Layout::<32>::TOKENS_OFFSET+offset*4;assert_eq!(&p[at..at+4],&token.to_le_bytes());offset+=1;}}
            let mut scratch=vec![0;p.len()];validate_packet(&p,&mut scratch,&e).unwrap();
            for at in 0..p.len(){p[at]^=0x80;assert!(validate_packet(&p,&mut scratch,&e).is_err(),"byte{at}");p[at]^=0x80;}
            let compact=compact_fixture(&e);let rows=validate_compact_result(&compact,&e).unwrap();assert_eq!(rows.iter().filter(|r|r.token.is_some()).count(),((active+1)/2)as usize);
            let mut wrong=e.clone();wrong.packed_prefill=false;assert!(validate_compact_result(&compact,&wrong).is_err());
            if active>1{let mut wrong=e.clone();wrong.rows.swap(0,1);let slot=wrong.rows[0].output_slot;wrong.rows[0].output_slot=wrong.rows[1].output_slot;wrong.rows[1].output_slot=slot;assert!(validate(&wrong).is_err());}
            if let Some(d)=&export{std::fs::write(d.join(format!("v6-prefill-{active}-{chunk}.bin")),&p).unwrap();}
        }}
        for active in [1,16,32]{let mut e=fixture_rows::<32>(InputStage::Decode,active);e.packed_prefill=true;let mut p=vec![0;Layout::<32>::REQUEST_BYTES];encode_into(&mut p,&e).unwrap();assert_eq!(&p[36..40],&[0;4]);if let Some(d)=&export{std::fs::write(d.join(format!("v6-decode-{active}.bin")),p).unwrap();}}
    }
    #[test] fn packed_v6_rejects_capacity_before_write(){
        for e in [packed_fixture(4,257),packed_fixture(5,128)]{let mut p=vec![0xa5;Layout::<32>::REQUEST_BYTES];assert!(encode_into(&mut p,&e).is_err());assert!(p.iter().all(|&x|x==0xa5));}
        let mut e=fixture_rows::<16>(InputStage::Prefill,1);e.packed_prefill=true;assert!(validate(&e).is_err());
        let mut e=packed_fixture(4,256);e.rows[1].physical_ids[0]=e.rows[0].physical_ids[0];assert!(validate(&e).is_err());
    }
'''
s=s.replace('    fn maximum_prefill16()',t+'    fn maximum_prefill16()');p.write_text(s)
print('Added packed wire fixtures and authority corruption coverage')
