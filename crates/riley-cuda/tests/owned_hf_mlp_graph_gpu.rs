//! Fixed-address MLP graph feasibility only. All operands and both plans stay
//! alive until graph close; production graph ownership remains unqualified.
#![cfg(feature = "cuda-cublas-gemm-probe")]
use riley_cuda::{
    CublasGemmProbeParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType,
    CudaDeviceBuffer, CudaExecutionStream, CudaGemmConfig, CudaPinnedHostBuffer,
    CudaPreparedCublasGemmProbe, CudaRuntime, CudaStream, GatedMultiplyParams, begin_step_capture,
    end_step_capture, launch_step_graph, silu_gated_multiply,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, time::Instant};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
fn sha(v: &[u8]) -> String {
    Sha256::digest(v)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn upload(
    c: &CudaContext,
    s: &mut CudaStream,
    h: &mut CudaPinnedHostBuffer,
    v: &[u8],
) -> Result<CudaDeviceBuffer> {
    let mut b = c.allocate_device_buffer(v.len() as u64)?;
    b.upload_from_slice(0, v, h, s)?;
    Ok(b)
}
#[allow(clippy::too_many_arguments)]
fn mlp<S: CudaExecutionStream + ?Sized>(
    input: &CudaDeviceBuffer,
    wg: &CudaDeviceBuffer,
    wu: &CudaDeviceBuffer,
    wd: &CudaDeviceBuffer,
    gate_plan: &mut CudaPreparedCublasGemmProbe,
    down_plan: &mut CudaPreparedCublasGemmProbe,
    gate: &mut CudaDeviceBuffer,
    up: &mut CudaDeviceBuffer,
    product: &mut CudaDeviceBuffer,
    result: &mut CudaDeviceBuffer,
    slots: &[u32],
    stream: &mut S,
) -> Result {
    for (weight, out) in [(wg, &mut *gate), (wu, &mut *up)] {
        let bytes = 32 * 1536 * 2;
        let mut p = CublasGemmProbeParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, 32 * 576 * 2)?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, weight.byte_len())?,
            output: CudaBufferSpanMut::new(out, CudaDType::BF16, 0, bytes)?,
        };
        gate_plan.execute_row_groups(&mut p, slots, None, stream)?;
    }
    let elements = slots.len() as u64 * 1536;
    let bytes = product.byte_len();
    silu_gated_multiply(
        &mut GatedMultiplyParams {
            activated_gate: CudaBufferSpan::new(gate, CudaDType::BF16, 0, gate.byte_len())?,
            up: CudaBufferSpan::new(up, CudaDType::BF16, 0, up.byte_len())?,
            output: CudaBufferSpanMut::new(product, CudaDType::BF16, 0, bytes)?,
            element_count: elements,
        },
        stream,
    )?;
    let bytes = 32 * 576 * 2;
    let mut p = CublasGemmProbeParams {
        input: CudaBufferSpan::new(product, CudaDType::BF16, 0, 32 * 1536 * 2)?,
        weight: CudaBufferSpan::new(wd, CudaDType::BF16, 0, wd.byte_len())?,
        output: CudaBufferSpanMut::new(result, CudaDType::BF16, 0, bytes)?,
    };
    down_plan.execute_row_groups(&mut p, slots, None, stream)?;
    Ok(())
}
#[test]
#[ignore = "GPU2 only; pinned HF MLP oracle required; graph feasibility not serving admission"]
fn request_grouped_mlp_graph_matches_independent_hf() -> Result {
    let root = PathBuf::from(std::env::var("RILEY_MLP_ORACLE_ROOT")?);
    let out = PathBuf::from(std::env::var("RILEY_MLP_GRAPH_OUTPUT")?);
    fs::create_dir(&out)?;
    let fixture = fs::read(root.join("fixture.json"))?;
    let m: Value = serde_json::from_slice(&fixture)?;
    let read = |name: &str| -> Result<Vec<u8>> {
        assert!(
            !name.contains("..")
                && name
                    .bytes()
                    .all(|v| v.is_ascii_alphanumeric() || b"-_.".contains(&v))
        );
        let v = fs::read(root.join(name))?;
        assert_eq!(Some(v.len() as u64), m["files"][name]["bytes"].as_u64());
        assert_eq!(sha(&v), m["files"][name]["sha256"].as_str().ok_or("hash")?);
        Ok(v)
    };
    assert_eq!(m["hidden"], 576);
    assert_eq!(m["intermediate"], 1536);
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut staging = c.allocate_pinned_host_buffer(4 << 20)?;
    let wg = upload(
        &c,
        &mut stream,
        &mut staging,
        &read(m["weights"]["gate"].as_str().ok_or("gate")?)?,
    )?;
    let wu = upload(
        &c,
        &mut stream,
        &mut staging,
        &read(m["weights"]["up"].as_str().ok_or("up")?)?,
    )?;
    let wd = upload(
        &c,
        &mut stream,
        &mut staging,
        &read(m["weights"]["down"].as_str().ok_or("down")?)?,
    )?;
    let mut input = c.allocate_device_buffer(64 * 576 * 2)?;
    let mut gate = c.allocate_device_buffer(64 * 1536 * 2)?;
    let mut up = c.allocate_device_buffer(64 * 1536 * 2)?;
    let mut product = c.allocate_device_buffer(64 * 1536 * 2)?;
    let mut result = c.allocate_device_buffer(64 * 576 * 2)?;
    let mut gate_plan = c.prepare_owned_hf_gemm(CudaGemmConfig::new(32, 1536, 576, 0)?)?;
    let mut down_plan = c.prepare_owned_hf_gemm(CudaGemmConfig::new(32, 576, 1536, 0)?)?;
    let mut managed = riley_cuda::CudaOwnedHfMlpGraph::prepare(&c, 32, 576, 1536, 2)?;
    let mut other = c.create_stream()?;
    let mut reports = Vec::new();
    for case in m["cases"].as_array().ok_or("cases")? {
        let b = case["batch"].as_u64().ok_or("batch")?;
        assert!([1, 2, 8, 32].contains(&b));
        let slots: Vec<u32> = (0..b as u32).collect();
        let x = read(case["input"].as_str().ok_or("input")?)?;
        let expected = read(case["output"].as_str().ok_or("output")?)?;
        input.upload_from_slice(0, &x, &mut staging, &mut stream)?;
        mlp(
            &input,
            &wg,
            &wu,
            &wd,
            &mut gate_plan,
            &mut down_plan,
            &mut gate,
            &mut up,
            &mut product,
            &mut result,
            &slots,
            &mut stream,
        )?;
        let mut actual = vec![0; expected.len()];
        result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        assert_eq!(actual, expected, "eager HF MLP mismatch at batch {b}");
        let graph = {
            let mut batch = stream.begin_command_batch()?;
            let graph = {
                let mut commands = batch.commands();
                begin_step_capture(&mut commands)?;
                mlp(
                    &input,
                    &wg,
                    &wu,
                    &wd,
                    &mut gate_plan,
                    &mut down_plan,
                    &mut gate,
                    &mut up,
                    &mut product,
                    &mut result,
                    &slots,
                    &mut commands,
                )?;
                end_step_capture(&mut commands)?
            };
            batch.finish()?;
            graph
        };
        let stable = c.allocation_stats()?;
        let mut times = Vec::new();
        for repeat in 0..6 {
            for graph_mode in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let mut batch = stream.begin_command_batch()?;
                {
                    let mut commands = batch.commands();
                    for _ in 0..30 {
                        if graph_mode {
                            launch_step_graph(&mut commands, &graph)?;
                        } else {
                            mlp(
                                &input,
                                &wg,
                                &wu,
                                &wd,
                                &mut gate_plan,
                                &mut down_plan,
                                &mut gate,
                                &mut up,
                                &mut product,
                                &mut result,
                                &slots,
                                &mut commands,
                            )?;
                        }
                    }
                }
                batch.finish()?;
                times.push(json!({"repeat":repeat,"mode":if graph_mode{"graph"}else{"eager"},"elapsed_ns":start.elapsed().as_nanos()}));
                result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                assert_eq!(actual, expected, "MLP replay changed at batch {b}");
            }
        }
        assert_eq!(c.allocation_stats()?, stable);
        for _ in 0..2 {
            let mut batch = stream.begin_command_batch()?;
            {
                let mut commands = batch.commands();
                managed.execute(
                    &mut riley_cuda::CudaOwnedHfMlpParams {
                        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                        gate_weight: CudaBufferSpan::new(&wg, CudaDType::BF16, 0, wg.byte_len())?,
                        up_weight: CudaBufferSpan::new(&wu, CudaDType::BF16, 0, wu.byte_len())?,
                        down_weight: CudaBufferSpan::new(&wd, CudaDType::BF16, 0, wd.byte_len())?,
                        gate: &mut gate,
                        up: &mut up,
                        product: &mut product,
                        output: &mut result,
                        active_rows: b as u32,
                    },
                    &mut commands,
                )?;
            }
            assert!(
                result
                    .download_to_slice(0, &mut actual, &mut staging, &mut other)
                    .is_err(),
                "replay must retain its output lease through batch completion"
            );
            batch.finish()?;
            result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
            assert_eq!(actual, expected, "managed graph mismatch at batch {b}");
        }
        graph.close()?;
        reports.push(json!({"batch":b,"exact":true,"output_sha256":sha(&actual),"timings":times}));
    }
    assert_eq!(managed.stats().captures, 2);
    assert_eq!(managed.stats().hits, 2);
    assert_eq!(managed.stats().eager_fallbacks, 4);
    // Free/reallocate the output while cached graph metadata still exists.
    // The new allocation tag must prevent a stale C1 hit even if CUDA reuses
    // the same address. A full cache takes the exact eager path instead.
    result.close()?;
    result = c.allocate_device_buffer(64 * 576 * 2)?;
    let first = &m["cases"][0];
    let x = read(first["input"].as_str().ok_or("input")?)?;
    let expected = read(first["output"].as_str().ok_or("output")?)?;
    input.upload_from_slice(0, &x, &mut staging, &mut stream)?;
    let mut batch = stream.begin_command_batch()?;
    {
        let mut commands = batch.commands();
        managed.execute(
            &mut riley_cuda::CudaOwnedHfMlpParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                gate_weight: CudaBufferSpan::new(&wg, CudaDType::BF16, 0, wg.byte_len())?,
                up_weight: CudaBufferSpan::new(&wu, CudaDType::BF16, 0, wu.byte_len())?,
                down_weight: CudaBufferSpan::new(&wd, CudaDType::BF16, 0, wd.byte_len())?,
                gate: &mut gate,
                up: &mut up,
                product: &mut product,
                output: &mut result,
                active_rows: 1,
            },
            &mut commands,
        )?;
    }
    batch.finish()?;
    let mut actual = vec![0; expected.len()];
    result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
    assert_eq!(actual, expected);
    assert_eq!(managed.stats().hits, 2);
    assert_eq!(managed.stats().eager_fallbacks, 5);
    // Changing streams retires prior graphs before cuBLAS workspace rebinding.
    stream.close()?;
    stream = c.create_stream()?;
    let mut batch = stream.begin_command_batch()?;
    {
        let mut commands = batch.commands();
        managed.execute(
            &mut riley_cuda::CudaOwnedHfMlpParams {
                input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                gate_weight: CudaBufferSpan::new(&wg, CudaDType::BF16, 0, wg.byte_len())?,
                up_weight: CudaBufferSpan::new(&wu, CudaDType::BF16, 0, wu.byte_len())?,
                down_weight: CudaBufferSpan::new(&wd, CudaDType::BF16, 0, wd.byte_len())?,
                gate: &mut gate,
                up: &mut up,
                product: &mut product,
                output: &mut result,
                active_rows: 1,
            },
            &mut commands,
        )?;
    }
    batch.finish()?;
    result.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
    assert_eq!(actual, expected);
    assert_eq!(managed.stats().stream_resets, 1);
    assert_eq!(managed.stats().captures, 3);
    let managed_stats = managed.stats();
    managed.close()?;
    other.close()?;
    gate_plan.close()?;
    down_plan.close()?;
    result.close()?;
    product.close()?;
    up.close()?;
    gate.close()?;
    input.close()?;
    wd.close()?;
    wu.close()?;
    wg.close()?;
    staging.close()?;
    stream.close()?;
    assert!(c.allocation_stats()?.is_zero());
    c.close()?;
    fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"fixed-address MLP segment diagnostic; operands retained by test; not production graph lifetime or full-model performance qualification","fixture_sha256":sha(&fixture),"mlp_repetitions_per_sample":30,"managed_stats":{"hits":managed_stats.hits,"captures":managed_stats.captures,"capacity_fallbacks":managed_stats.eager_fallbacks},"cases":reports}),
        )?,
    )?;
    Ok(())
}
