//! Iteration completion must retain a reused bias plan and all of its operands.
#![cfg(feature = "cuda-cublas-gemm-probe")]

use riley_cuda::{
    BiasGemmParams, CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaGemmConfig,
    CudaGemmReductionPolicy, CudaRuntime,
};

#[test]
#[ignore = "Requires one admitted CUDA GPU; BF16 bias GEMM and cross-stream lifetime contract"]
fn bias_plan_rebinds_operands_inside_a_batch_without_releasing_leases()
-> Result<(), Box<dyn std::error::Error>> {
    let runtime = CudaRuntime::initialize()?;
    let context = runtime.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut other = context.create_stream()?;
    let mut host = context.allocate_pinned_host_buffer(1 << 20)?;
    let (m, n, k) = (8, 256, 128);
    let mut input = context.allocate_device_buffer(m * k * 2)?;
    let mut weight = context.allocate_device_buffer(n * k * 2)?;
    let mut bias_a = context.allocate_device_buffer(n * 2)?;
    let mut bias_b = context.allocate_device_buffer(n * 2)?;
    input.upload_from_slice(
        0,
        &0x3f80u16.to_le_bytes().repeat((m * k) as usize),
        &mut host,
        &mut stream,
    )?;
    weight.upload_from_slice(
        0,
        &0x3f80u16.to_le_bytes().repeat((n * k) as usize),
        &mut host,
        &mut stream,
    )?;
    bias_a.upload_from_slice(
        0,
        &0x3f80u16.to_le_bytes().repeat(n as usize),
        &mut host,
        &mut stream,
    )?;
    bias_b.upload_from_slice(
        0,
        &0xc000u16.to_le_bytes().repeat(n as usize),
        &mut host,
        &mut stream,
    )?;
    let config = CudaGemmConfig::new(m, n, k, 1 << 20)?
        .with_reduction_policy(CudaGemmReductionPolicy::HfCompatibleBiasSplitKV1);
    let mut plan = context.prepare_hf_compatible_bias_epilogue_probe(
        config,
        CudaBufferSpan::new(&bias_a, CudaDType::BF16, 0, n * 2)?,
    )?;
    let need = plan.algorithm_metadata().workspace_bytes();
    let mut workspace = if need == 0 {
        None
    } else {
        Some(context.allocate_device_buffer(need)?)
    };
    let mut out_a = context.allocate_device_buffer(m * n * 2)?;
    let mut out_b = context.allocate_device_buffer(m * n * 2)?;
    let stable = context.allocation_stats()?;
    for _ in 0..3 {
        let mut batch = stream.begin_command_batch()?;
        {
            let mut commands = batch.commands();
            // Reusing one prepared descriptor with different bias pointers
            // must not make the first launch read the second launch's bias.
            for (bias, output) in [(&bias_a, &mut out_a), (&bias_b, &mut out_b)] {
                plan.execute(
                    &mut BiasGemmParams {
                        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, m * k * 2)?,
                        weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, n * k * 2)?,
                        bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, n * 2)?,
                        output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, m * n * 2)?,
                        workspace: workspace
                            .as_mut()
                            .map(|b| CudaBufferSpanMut::new(b, CudaDType::U8, 0, need))
                            .transpose()?,
                    },
                    &mut commands,
                )?;
            }
        }
        for buffer in [
            &mut input,
            &mut weight,
            &mut bias_a,
            &mut bias_b,
            &mut out_a,
            &mut out_b,
        ]
        .into_iter()
        .chain(workspace.as_mut())
        {
            let mut bytes = vec![0; buffer.byte_len() as usize];
            assert!(
                buffer
                    .download_to_slice(0, &mut bytes, &mut host, &mut other)
                    .is_err(),
                "unfinished bias batch must exclude other streams"
            );
        }
        batch.finish()?;
        for (output, expected) in [(&mut out_a, 129.0f32), (&mut out_b, 126.0f32)] {
            let mut bytes = vec![0; (m * n * 2) as usize];
            output.download_to_slice(0, &mut bytes, &mut host, &mut stream)?;
            assert_eq!(
                bytes,
                ((expected.to_bits() >> 16) as u16)
                    .to_le_bytes()
                    .repeat((m * n) as usize)
            );
        }
        assert_eq!(context.allocation_stats()?, stable);
    }
    plan.close()?;
    if let Some(buffer) = workspace {
        buffer.close()?;
    }
    out_b.close()?;
    out_a.close()?;
    bias_b.close()?;
    bias_a.close()?;
    weight.close()?;
    input.close()?;
    host.close()?;
    assert!(context.allocation_stats()?.is_zero());
    other.close()?;
    stream.close()?;
    context.close()?;
    Ok(())
}
