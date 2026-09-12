#![cfg(feature="cuda")]
use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};
fn p32(p:&mut[u8],i:usize,v:u32){p[i..i+4].copy_from_slice(&v.to_le_bytes());}
fn p64(p:&mut[u8],i:usize,v:u64){p[i..i+8].copy_from_slice(&v.to_le_bytes());}
fn packet(start:u32,count:u32,prompt:u32,replay:u64)->Vec<u8>{let mut p=vec![0;30848];let target=start+count;let live=target.div_ceil(16);
for(i,v)in[(0,0x34444d52),(4,4),(8,30848),(12,1664),(20,1),(24,16),(28,4),(128,33),(132,target-1),(136,count),(140,live),(144,start),(148,target),(156,32),(160,prompt),(164,64),(172,if target==prompt{count-1}else{u32::MAX})]{p32(&mut p,i,v);}
for(i,v)in[(40,1),(48,replay),(56,replay),(176,1),(184,replay)]{p64(&mut p,i,v);}p[64]=1;
for i in 0..live{p32(&mut p,256+i as usize*4,(i*3+1)%4);let valid=if i+1<live{16u16}else{((target-1)%16+1)as u16};p[1280+i as usize*2..1282+i as usize*2].copy_from_slice(&valid.to_le_bytes());}
for i in 0..count{p32(&mut p,26752+i as usize*4,33);}p}
#[test]
#[ignore="requires GPU; native shared capture/transfer/ledger"]
fn sixteen_recorder_owns_resources_and_transfers_all_records()->Result<(),Box<dyn std::error::Error>>{
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut upload=context.allocate_pinned_host_buffer(1<<20)?;
let mut sizes=vec![1152*32;5];sizes.extend([384*32,384*32,16*9*4096*4,3072*32,3072*32,2304*32,3072*32]);sizes.extend([30*16*16*384,30*16*16*384,64*128,64*128,30848,1152,128,4,98304,8,56623104,1152,663552,221184,1769472,16*1152,16*98304,1574912]);
let mut buffers=vec![];for size in sizes{let mut b=context.allocate_device_buffer(size)?;b.upload_from_slice(0,&vec![0;size as usize],&mut upload,&mut stream)?;buffers.push(b);}
let mut weights=vec![22,23,22];for _ in 0..30{weights.extend([23,24,25,25,24,23,26,26,26]);}
let mut head=context.prepare_gemm(CudaGemmConfig::new(1,49152,576,0)?)?;let mut shared=context.prepare_gemm(CudaGemmConfig::new(16,49152,576,0)?)?;let mut staging=context.allocate_pinned_host_buffer(3149824)?;
let mut owner=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream:&mut stream,devices:buffers.iter_mut().collect(),pinned:vec![&mut staging],plans:vec![&mut head,&mut shared]})?;
let devices=std::array::from_fn(|i|if i<22{i}else{i+5});
let mut bad=devices;bad[24]=bad[23];assert!(owner.record_v4_shared(&bad,None,&weights,0,1,0,32,16).is_err());
assert!(owner.record_v4_shared(&devices,None,&weights,0,0,0,32,16).is_err());
owner.record_v4_shared(&devices,None,&weights,0,1,0,32,16)?;
for (i,(start,count,prompt)) in [(0,17,33),(17,16,33),(0,1,1)].into_iter().enumerate(){let mut p=packet(start,count,prompt,i as u64+1);p32(&mut p,28,16);let mut out=vec![255;1574912];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;assert_eq!(&out[..4],&0u32.to_le_bytes());assert_eq!(&out[4..8],&u32::from(start+count==prompt).to_le_bytes());assert!(out[128..].iter().all(|&x|x==0));}
for active in [1usize,2,4,8,9,15,16] {
let mut p=packet(1,1,1,4);p[26752..].fill(0);p32(&mut p,28,16);p32(&mut p,16,1);p32(&mut p,20,active as u32);p32(&mut p,152,1);p32(&mut p,172,0);
let row=p[128..1792].to_vec();for i in 1..active {let b=128+i*1664;p[b..b+1664].copy_from_slice(&row);p32(&mut p,b+40,i as u32);p64(&mut p,b+48,i as u64+1);p64(&mut p,b+56,i as u64+4);p32(&mut p,b+128,((i+1)%16)as u32);}
let mut out=vec![255;1574912];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;
for i in 0..active {let b=i*98432;assert_eq!(&out[b..b+4],&0u32.to_le_bytes());assert_eq!(&out[b+4..b+8],&1u32.to_le_bytes());assert_eq!(&out[b+124..b+128],&0x34524d52u32.to_le_bytes());assert_eq!(&out[b+40..b+48],&((i+1)as u64).to_le_bytes());assert_eq!(&out[b+116..b+120],&(i as u32).to_le_bytes());assert!(out[b+128..b+98432].iter().all(|&x|x==0));}
assert!(out[active*98432..].iter().all(|&x|x==0));
let mut wrong=p.clone();p32(&mut wrong,0,0x33444d52);assert!(owner.replay_transfer(&wrong).is_err());assert!(owner.read_transfer(&mut out).is_err());
let mut wrong=p.clone();p32(&mut wrong,128+(active-1)*1664+36,128);assert!(owner.replay_transfer(&wrong).is_err());assert!(owner.read_transfer(&mut out).is_err());
}
owner.close()?;head.close()?;shared.close()?;drop(buffers);drop(staging);drop(upload);stream.close()?;assert!(context.allocation_stats()?.is_zero());
eprintln!("SHARED16_NATIVE prefill_replays=3 decode_rows=1,2,4,8,9,15,16 inactive_zero=true cold_rejections=2 foreign_context_rejected=true stale_output_hidden=true allocation_zero=true");Ok(())}
