from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-cuda/tests/v3_shared_recorder_gpu.rs');s=p.read_text().replace('30*4*16*384','30*8*16*384').replace('0,32,4)','0,32,8)').replace('let p=packet(start,count,prompt,i as u64+1);','let mut p=packet(start,count,prompt,i as u64+1);p32(&mut p,28,8);')
a=s.index('let mut p=packet(1,1,1,4);');b=s.index('owner.close()',a)
s=s[:a]+'''for active in [2usize,4,8] {
let mut p=packet(1,1,1,4);p[13440..].fill(0);p32(&mut p,28,8);p32(&mut p,16,1);p32(&mut p,20,active as u32);p32(&mut p,152,1);p32(&mut p,172,0);
let row=p[128..1792].to_vec();for i in 1..active {let b=128+i*1664;p[b..b+1664].copy_from_slice(&row);p32(&mut p,b+40,i as u32);p64(&mut p,b+48,i as u64+1);p64(&mut p,b+56,i as u64+4);p32(&mut p,b+128,((i+1)%8)as u32);}
let mut out=vec![255;787456];owner.replay_transfer(&p)?;owner.read_transfer(&mut out)?;
for i in 0..active {let b=i*98432;assert_eq!(&out[b..b+4],&0u32.to_le_bytes());assert_eq!(&out[b+4..b+8],&1u32.to_le_bytes());assert_eq!(&out[b+40..b+48],&((i+1)as u64).to_le_bytes());assert_eq!(&out[b+116..b+120],&(i as u32).to_le_bytes());assert!(out[b+128..b+98432].iter().all(|&x|x==0));}
assert!(out[active*98432..].iter().all(|&x|x==0));
let mut wrong=p.clone();p32(&mut wrong,128+(active-1)*1664+36,128);assert!(owner.replay_transfer(&wrong).is_err());assert!(owner.read_transfer(&mut out).is_err());
}
'''+s[b:];s=s.replace('decode_rows=2','decode_rows=2,4,8');p.write_text(s)
