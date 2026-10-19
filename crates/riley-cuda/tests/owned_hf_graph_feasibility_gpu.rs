//! Diagnostic only: keeps operands and plan live until the recorded graph is
//! closed. This does not admit serving graphs or prove graph resource ownership.
#![cfg(feature = "cuda-cublas-gemm-probe")]
use riley_cuda::{
    CublasGemmProbeParams, CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaGemmConfig,
    CudaRuntime, begin_step_capture, end_step_capture, launch_step_graph,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};

#[test]
#[ignore = "GPU2 only, pinned HF fixture; feasibility and timing diagnostic, not qualification"]
fn owned_classic_gemm_step_graph_preserves_frozen_hf_output()
-> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("RILEY_COMPONENT_ORACLE_ROOT")?);
    let output = PathBuf::from(std::env::var("RILEY_GRAPH_FEASIBILITY_OUTPUT")?);
    fs::create_dir(&output)?;
    let manifest_bytes = fs::read(root.join("fixture.json"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    let case = manifest["cases"]
        .as_array()
        .ok_or("cases")?
        .iter()
        .find(|v| v["id"] == "smollm2-135m-decode-q-proj")
        .ok_or("case")?;
    let read = |key: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let name = case[key].as_str().ok_or("name")?;
        assert!(
            name.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                && !name.contains("..")
        );
        let bytes = fs::read(root.join(name))?;
        assert_eq!(
            Some(bytes.len() as u64),
            manifest["files"][name]["bytes"].as_u64()
        );
        assert_eq!(
            Sha256::digest(&bytes).iter().map(|v|format!("{v:02x}")).collect::<String>(),
            manifest["files"][name]["sha256"].as_str().ok_or("hash")?
        );
        Ok(bytes)
    };
    let input_bytes = read("input")?;
    let weight_bytes = read("weight")?;
    let expected = read("output")?;
    let (m, n, k) = (
        case["m"].as_u64().ok_or("m")?,
        case["n"].as_u64().ok_or("n")?,
        case["k"].as_u64().ok_or("k")?,
    );
    assert_eq!((m, n, k), (1, 576, 576));
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut staging = c.allocate_pinned_host_buffer(1 << 20)?;
    let mut input = c.allocate_device_buffer(input_bytes.len() as u64)?;
    let mut weight = c.allocate_device_buffer(weight_bytes.len() as u64)?;
    let mut result = c.allocate_device_buffer(expected.len() as u64)?;
    input.upload_from_slice(0, &input_bytes, &mut staging, &mut stream)?;
    weight.upload_from_slice(0, &weight_bytes, &mut staging, &mut stream)?;
    let mut plan = c.prepare_owned_hf_gemm(CudaGemmConfig::new(m, n, k, 0)?)?;
    let mut params = CublasGemmProbeParams {
        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
        weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight.byte_len())?,
        output: CudaBufferSpanMut::new(&mut result, CudaDType::BF16, 0, expected.len() as u64)?,
    };
    // Initialize all lazy cuBLAS state before entering thread-local capture.
    plan.execute(&mut params, &mut stream)?;
    let operations = 180;
    let graph = {
        let mut batch = stream.begin_command_batch()?;
        let graph = {
            let mut commands = batch.commands();
            begin_step_capture(&mut commands)?;
            for _ in 0..operations {
                plan.execute(&mut params, &mut commands)?;
            }
            end_step_capture(&mut commands)?
        };
        batch.finish()?;
        graph
    };
    let stable = c.allocation_stats()?;
    let mut timings = Vec::new();
    // Both modes submit identical arithmetic and synchronize once per sample.
    // Buffers, plan and stream stay alive for every launch and graph teardown.
    for repeat in 0..6 {
        for use_graph in if repeat % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            let mut batch = stream.begin_command_batch()?;
            {
                let mut commands = batch.commands();
                if use_graph {
                    launch_step_graph(&mut commands, &graph)?;
                } else {
                    for _ in 0..operations {
                        plan.execute(&mut params, &mut commands)?;
                    }
                }
            }
            batch.finish()?;
            timings.push(json!({"repeat":repeat,"mode":if use_graph {"graph"} else {"eager"},"elapsed_ns":start.elapsed().as_nanos()}));
        }
    }
    drop(params);
    let mut actual = vec![0; expected.len()];
    result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
    assert_eq!(
        actual, expected,
        "frozen HF output changed after graph replay"
    );
    assert_eq!(
        c.allocation_stats()?,
        stable,
        "graph replay must reuse cold device allocations"
    );
    // This explicit teardown ordering is part of the diagnostic's precondition.
    graph.close()?;
    plan.close()?;
    result.close()?;
    weight.close()?;
    input.close()?;
    staging.close()?;
    stream.close()?;
    assert!(c.allocation_stats()?.is_zero());
    c.close()?;
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"graph feasibility only; one frozen HF projection repeated, no full model/performance or lifetime admission", "operations_per_sample":operations,"fixture_sha256":Sha256::digest(&manifest_bytes).iter().map(|v|format!("{v:02x}")).collect::<String>(),"exact_output":true,"timings":timings}),
        )?,
    )?;
    Ok(())
}
