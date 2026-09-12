#![cfg(feature="cuda")]
use riley_cuda::{BorrowedGraphResourceParents,BorrowedGraphResourceReservation,CudaRuntime,CudaGemmConfig};
fn p32(p:&mut[u8],i:usize,v:u32){p[i..i+4].copy_from_slice(&v.to_le_bytes());}
fn p64(p:&mut[u8],i:usize,v:u64){p[i..i+8].copy_from_slice(&v.to_le_bytes());}
fn packet(start:u32,count:u32,prompt:u32,replay:u64)->Vec<u8>{let mut p=vec![0;17536];let target=start+count;let live=target.div_ceil(16);
for(i,v)in[(0,0x33444d52),(4,3),(8,17536),(12,1664),(20,1),(24,8),(28,4),(128,33),(132,target-1),(136,count),(140,live),(144,start),(148,target),(156,32),(160,prompt),(164,64),(172,if target==prompt{count-1}else{u32::MAX})]{p32(&mut p,i,v);}
for(i,v)in[(40,1),(48,replay),(56,replay),(176,1),(184,replay)]{p64(&mut p,i,v);}p[64]=1;
for i in 0..live{p32(&mut p,256+i as usize*4,(i*3+1)%4);let valid=if i+1<live{16u16}else{((target-1)%16+1)as u16};p[1280+i as usize*2..1282+i as usize*2].copy_from_slice(&valid.to_le_bytes());}
for i in 0..count{p32(&mut p,13440+i as usize*4,33);}p}
#[test]
#[ignore="requires CUDA; zero-weight recorder/head/ledger lifecycle, not model quality"]
fn v3_recorder_head_and_partial_completion()->Result<(),Box<dyn std::error::Error>>{
let context=CudaRuntime::initialize()?.device(0)?.create_context()?;let mut stream=context.create_stream()?;let mut upload=context.allocate_pinned_host_buffer(1<<20)?;
let mut sizes=vec![1152*32;5];sizes.extend([384*32,384*32,384*32,3072*32,3072*32,2304*32,3072*32]);sizes.extend([30*4*16*384,30*4*16*384,64*128,64*128,17536,1152,128,4,98304,8,56623104,1152,663552,221184,1769472]);
let mut buffers=vec![];for size in sizes{let mut b=context.allocate_device_buffer(size)?;b.upload_from_slice(0,&vec![0;size as usize],&mut upload,&mut stream)?;buffers.push(b);}
let mut weights=vec![22,23,22];for _ in 0..30{weights.extend([23,24,25,25,24,23,26,26,26]);}
let mut head=context.prepare_gemm(CudaGemmConfig::new(1,49152,576,0)?)?;let mut staging=context.allocate_pinned_host_buffer(196864)?;
let mut owner=BorrowedGraphResourceReservation::reserve(BorrowedGraphResourceParents{stream:&mut stream,devices:buffers.iter_mut().collect(),pinned:vec![&mut staging],plans:vec![&mut head]})?;
let devices=std::array::from_fn(|i|i);let mut bad=devices;bad[1]=0;assert!(owner.record_v3_prefill(&bad,None,&weights,0,0,32,4).is_err());
let mut wrong=weights.clone();wrong[3]=24;assert!(owner.record_v3_prefill(&devices,None,&wrong,0,0,32,4).is_err());
owner.record_v3_prefill(&devices,None,&weights,0,0,32,4)?;
for (i,(start,count,prompt)) in [(0,17,33),(17,16,33),(0,1,1)].into_iter().enumerate(){let p=packet(start,count,prompt,i as u64+1);let mut out=vec![255;98432];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;assert_eq!(&out[..4],&0u32.to_le_bytes());assert_eq!(&out[4..8],&u32::from(start+count==prompt).to_le_bytes());assert!(out[8..16].iter().all(|&x|x==0));assert!(out[128..].iter().all(|&x|x==0));assert_eq!(&out[24..32],&((i as u64)+1).to_le_bytes());assert_eq!(&out[40..48],&1u64.to_le_bytes());assert_eq!(&out[124..128],&0x33524d52u32.to_le_bytes());let mut bad=p.clone();p32(&mut bad,256,4);assert!(owner.replay_transfer(&bad).is_err());assert!(owner.read_transfer(&mut out).is_err());}
owner.close()?;head.close()?;drop(buffers);drop(staging);drop(upload);stream.close()?;assert!(context.allocation_stats()?.is_zero());eprintln!("V3_RECORDER replays=3 cold_rejections=2 malformed_rejections=3 partial_publish_suppressed=true head_zero_logits=true allocation_zero=true");Ok(())}
