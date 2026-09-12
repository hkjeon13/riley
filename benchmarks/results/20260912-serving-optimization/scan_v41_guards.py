from pathlib import Path
p=Path('/tmp/riley-opt-260912/prefill-shapes-source-v11/crates/riley-runtime/src/llama/multi_descriptor/result_scan.rs');s=p.read_text().replace('let mut b=vec![0;34];b[16..18].copy_from_slice(&word.to_le_bytes());\n            assert_eq!(finite_argmax(&b),super::super::variable_wire::scalar_finite_argmax(&b));','''for at in [word as usize%32,32+word as usize%32,64] {
                let mut b=vec![0;130];b[at*2..at*2+2].copy_from_slice(&word.to_le_bytes());
                assert_eq!(finite_argmax(&b),super::super::variable_wire::scalar_finite_argmax(&b));
            }''')
a=s.index('    #[test] fn zero_scan')
s=s[:a]+'''    #[cfg(all(target_os="linux",target_arch="x86_64"))]
    #[test] fn vector_reads_stop_at_an_inaccessible_guard_page(){
        use std::ffi::c_void;
        unsafe extern "C" {
            fn getpagesize()->i32;
            fn mmap(addr:*mut c_void,len:usize,prot:i32,flags:i32,fd:i32,offset:isize)->*mut c_void;
            fn mprotect(addr:*mut c_void,len:usize,prot:i32)->i32;
            fn munmap(addr:*mut c_void,len:usize)->i32;
        }
        // SAFETY: test owns this Linux anonymous mapping; no pointers escape.
        unsafe {
            let page=getpagesize() as usize;assert!(page>=4096);
            let ptr=mmap(std::ptr::null_mut(),page*2,3,0x22,-1,0);assert_ne!(ptr as isize,-1);
            assert_eq!(mprotect(ptr.cast::<u8>().add(page).cast(),page,0),0);
            std::slice::from_raw_parts_mut(ptr.cast::<u8>(),page).fill(0);
            for len in (0..=129).chain([257,4095,4096]) {
                let bytes=std::slice::from_raw_parts_mut(ptr.cast::<u8>().add(page-len),len);
                assert!(all_zero(bytes));
                if len%2==0 {assert_eq!(finite_argmax(bytes).unwrap(),0);}
                if len>0 {bytes[len-1]=1;assert!(!all_zero(bytes));bytes[len-1]=0;}
            }
            assert_eq!(munmap(ptr,page*2),0);
        }
    }
'''+s[a:];p.write_text(s)
