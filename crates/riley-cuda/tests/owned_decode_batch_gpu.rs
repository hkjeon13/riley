//! Real activation controls for exact singleton batching and unchanged prefill groups.
#![cfg(feature = "cuda-cublas-gemm-probe")]
use riley_cuda::{CudaRuntime,CudaGemmConfig,CublasGemmProbeParams,CudaBufferSpan,CudaBufferSpanMut,CudaDType,begin_step_capture,end_step_capture,launch_step_graph};
use serde_json::Value;
use sha2::{Digest,Sha256};
use std::{fs,path::PathBuf};
#[test]
#[ignore = "Requires one admitted SM80 GPU and pinned real-activation fixture"]
fn grouped_decode_matches_independent_rows_and_hf_with_replayed_inputs() -> Result<(),Box<dyn std::error::Error>> {
    let root=PathBuf::from(std::env::var("RILEY_COMPONENT_ORACLE_ROOT")?);
    let fixture:Value=serde_json::from_slice(&fs::read(root.join("fixture.json"))?)?;
    let read=|case:&Value,key:&str| -> Result<Vec<u8>,Box<dyn std::error::Error>> {
        let name=case[key].as_str().ok_or("missing filename")?;
        assert!(name.bytes().all(|b|b.is_ascii_alphanumeric()||b"-_.".contains(&b)) && !name.contains(".."));
        let bytes=fs::read(root.join(name))?;
        assert_eq!(Some(bytes.len() as u64),fixture["files"][name]["bytes"].as_u64());
        let hash=Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>();
        assert_eq!(Some(hash.as_str()),fixture["files"][name]["sha256"].as_str());Ok(bytes)
    };
    let mode=if std::env::var("RILEY_TEST_MLP_BATCH").as_deref()==Ok("1") {2} else {1};
    let rt=CudaRuntime::initialize()?;assert_eq!(rt.device_count(),1);
    let c=rt.device(0)?.create_context()?;let mut stream=c.create_stream()?;let mut other=c.create_stream()?;
    let mut staging=c.allocate_pinned_host_buffer(64<<20)?;let mut cases=0;
    for case in fixture["cases"].as_array().ok_or("cases missing")? {
        if case["m"].as_u64()!=Some(1) || case["kind"]!="linear" || case.get("bias").is_some() {continue;}
        let seed=read(case,"input")?;let weights=read(case,"weight")?;let hf=read(case,"output")?;
        let n=case["n"].as_u64().ok_or("n")?;let k=case["k"].as_u64().ok_or("k")?;
        let mut input=c.allocate_device_buffer(64*k*2)?;let mut weight=c.allocate_device_buffer(weights.len() as u64)?;
        let mut output=c.allocate_device_buffer(64*n*2)?;weight.upload_from_slice(0,&weights,&mut staging,&mut stream)?;
        let mut plan=c.prepare_owned_hf_gemm(CudaGemmConfig::new(64,n,k,0)?)?;
        let mut patterns=(1usize..=32).map(|count|(0..count as u32).collect::<Vec<_>>()).collect::<Vec<_>>();
        for groups in [vec![2,3,1,1,1,2,1,1,1],vec![1,1,2,1,1,3,1,1,1],vec![32]] {
            let mut slots=Vec::new();let mut id=0;
            for size in groups {slots.extend(std::iter::repeat(id).take(size));id+=1;}
            while slots.len()<32 {slots.push(id);id+=1;}
            patterns.push(slots);
        }
        for slots in patterns {
            let count=slots.len();
            let mut data=Vec::new();for row in 0..count {let mut v=seed.clone();v.rotate_left((row*2)%seed.len());data.extend(v);}
            input.upload_from_slice(0,&data,&mut staging,&mut stream)?;
            let mut expected=vec![0;count*n as usize*2];
            {
                let mut params=CublasGemmProbeParams {input:CudaBufferSpan::new(&input,CudaDType::BF16,0,64*k*2)?,weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len())?,output:CudaBufferSpanMut::new(&mut output,CudaDType::BF16,0,64*n*2)?};
                let mut batch=stream.begin_command_batch()?;
                plan.execute_row_groups(&mut params,&slots,None,0,&mut batch.commands())?;batch.finish()?;
            }
            output.download_to_slice(0,&mut expected,&mut staging,&mut stream)?;
            if count==1 || slots[0]!=slots[1] {assert_eq!(&expected[..n as usize*2],hf.as_slice(),"HF singleton mismatch {}",case["id"]);}
            let graph={
                let mut params=CublasGemmProbeParams {input:CudaBufferSpan::new(&input,CudaDType::BF16,0,64*k*2)?,weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len())?,output:CudaBufferSpanMut::new(&mut output,CudaDType::BF16,0,64*n*2)?};
                let mut batch=stream.begin_command_batch()?;
                plan.execute_row_groups(&mut params,&slots,None,mode,&mut batch.commands())?;
                drop(params);
                let mut probe=[0;2];
                assert!(input.download_to_slice(0,&mut probe,&mut staging,&mut other).is_err());
                assert!(weight.download_to_slice(0,&mut probe,&mut staging,&mut other).is_err());
                batch.finish()?;
                let mut batch=stream.begin_command_batch()?;
                let mut params=CublasGemmProbeParams {input:CudaBufferSpan::new(&input,CudaDType::BF16,0,64*k*2)?,weight:CudaBufferSpan::new(&weight,CudaDType::BF16,0,weight.byte_len())?,output:CudaBufferSpanMut::new(&mut output,CudaDType::BF16,0,64*n*2)?};
                let graph={let mut commands=batch.commands();begin_step_capture(&mut commands)?;plan.execute_row_groups(&mut params,&slots,None,mode,&mut commands)?;end_step_capture(&mut commands)?};
                launch_step_graph(&mut batch.commands(),&graph)?;batch.finish()?;graph
            };
            let stable=c.allocation_stats()?;let mut actual=vec![0;expected.len()];output.download_to_slice(0,&mut actual,&mut staging,&mut stream)?;
            assert!(actual==expected,"grouped changed {} C{}; first byte {:?}",case["id"],count,actual.iter().zip(&expected).position(|(a,b)|a!=b));
            // New values at the same addresses must be consumed on replay.
            data.fill(0);input.upload_from_slice(0,&data,&mut staging,&mut stream)?;
            {let mut batch=stream.begin_command_batch()?;launch_step_graph(&mut batch.commands(),&graph)?;batch.finish()?;}
            output.download_to_slice(0,&mut actual,&mut staging,&mut stream)?;assert!(actual.iter().all(|&v|v==0));
            assert_eq!(c.allocation_stats()?,stable);graph.close()?;
            cases+=1;
        }
        plan.close()?;output.close()?;weight.close()?;input.close()?;
    }
    assert!(cases>=16,"fixture selected no linear decode cases");
    println!("OWNED_BATCH_CONDITIONS {cases}");staging.close()?;other.close()?;stream.close()?;assert!(c.allocation_stats()?.is_zero());c.close()?;Ok(())
}
