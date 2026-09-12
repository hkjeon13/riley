from pathlib import Path
r=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11');base=(r/'crates/riley-cuda/tests/v3_recorder_gpu.rs').read_text();s=base.split('#[test]')[0]
s+='''#[test]
#[ignore="requires GPU; native shared capture/transfer/ledger"]
fn shared_recorder_owns_resources_and_transfers_all_records()->Result<(),Box<dyn std::error::Error>>{
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut upload=context.allocate_pinned_host_buffer(1<<20)?;
let mut sizes=vec![1152*32;5];sizes.extend([384*32,384*32,8*9*4096*4,3072*32,3072*32,2304*32,3072*32]);sizes.extend([30*4*16*384,30*4*16*384,64*128,64*128,17536,1152,128,4,98304,8,56623104,1152,663552,221184,1769472,8*1152,8*98304,787456]);
let mut buffers=vec![];for size in sizes{let mut b=context.allocate_device_buffer(size)?;b.upload_from_slice(0,&vec![0;size as usize],&mut upload,&mut stream)?;buffers.push(b);}
let mut weights=vec![22,23,22];for _ in 0..30{weights.extend([23,24,25,25,24,23,26,26,26]);}
let mut head=context.prepare_gemm(CudaGemmConfig::new(1,49152,576,0)?)?;let mut shared=context.prepare_gemm(CudaGemmConfig::new(8,49152,576,0)?)?;let mut staging=context.allocate_pinned_host_buffer(1574912)?;
let mut owner=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream:&mut stream,devices:buffers.iter_mut().collect(),pinned:vec![&mut staging],plans:vec![&mut head,&mut shared]})?;
let devices=std::array::from_fn(|i|if i<22{i}else{i+5});
let mut bad=devices;bad[24]=bad[23];assert!(owner.record_v3_shared(&bad,None,&weights,0,1,0,32,4).is_err());
assert!(owner.record_v3_shared(&devices,None,&weights,0,0,0,32,4).is_err());
owner.record_v3_shared(&devices,None,&weights,0,1,0,32,4)?;
for (i,(start,count,prompt)) in [(0,17,33),(17,16,33),(0,1,1)].into_iter().enumerate(){let p=packet(start,count,prompt,i as u64+1);let mut out=vec![255;787456];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;assert_eq!(&out[..4],&0u32.to_le_bytes());assert_eq!(&out[4..8],&u32::from(start+count==prompt).to_le_bytes());assert!(out[128..].iter().all(|&x|x==0));}
let mut p=packet(1,1,1,4);p32(&mut p,16,1);p32(&mut p,20,2);p32(&mut p,152,1);p32(&mut p,172,0);
let row=p[128..1792].to_vec();p[1792..3456].copy_from_slice(&row);p32(&mut p,1792+40,1);p64(&mut p,1792+48,2);p64(&mut p,1792+56,5);p32(&mut p,1792+128,2);
let mut out=vec![255;787456];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;
for i in 0..2 {let b=i*98432;assert_eq!(&out[b..b+4],&0u32.to_le_bytes());assert_eq!(&out[b+4..b+8],&1u32.to_le_bytes());assert_eq!(&out[b+40..b+48],&((i+1)as u64).to_le_bytes());assert_eq!(&out[b+116..b+120],&(i as u32).to_le_bytes());assert!(out[b+128..b+98432].iter().all(|&x|x==0));}
assert!(out[2*98432..].iter().all(|&x|x==0));
let mut wrong=p.clone();p32(&mut wrong,1792+36,128);assert!(owner.replay_transfer(&wrong).is_err());assert!(owner.read_transfer(&mut out).is_err());
owner.close()?;head.close()?;shared.close()?;drop(buffers);drop(staging);drop(upload);stream.close()?;assert!(context.allocation_stats()?.is_zero());
eprintln!("SHARED_NATIVE prefill_replays=3 decode_rows=2 inactive_zero=true cold_rejections=2 foreign_context_rejected=true stale_output_hidden=true allocation_zero=true");Ok(())}
'''
(r/'crates/riley-cuda/tests/v3_shared_recorder_gpu.rs').write_text(s)
