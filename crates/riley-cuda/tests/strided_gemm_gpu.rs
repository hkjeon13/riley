#![cfg(feature = "cuda")]
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, CudaStridedGemmConfig, GemmParams,
};

#[test]
#[ignore = "requires qualified RTX 4090 CUDA runtime"]
fn strided_rust_owner_executes_rows_and_preserves_padding() -> Result<(), Box<dyn std::error::Error>>
{
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let mut checked = 0;
    for (n, k) in [
        (960, 576),
        (3072, 576),
        (576, 576),
        (576, 1536),
        (49152, 576),
    ] {
        for batch in [2, 4] {
            for padded in [false, true] {
                let config = CudaStridedGemmConfig::new(n, k, batch, padded)?;
                let mut plan = context.prepare_strided_gemm(config)?;
                assert_eq!(plan.algorithm_metadata().dimensions(), (1, n, k));
                assert_eq!(plan.algorithm_metadata().workspace_bytes(), 0);
                let mut input = vec![0x5a; config.input_bytes() as usize];
                for row in 0..batch as usize {
                    let base = row * config.input_stride() as usize * 2;
                    input[base..base + k as usize * 2].fill(0);
                    let word = (((row + 1) as f32).to_bits() >> 16) as u16;
                    input[base..base + 2].copy_from_slice(&word.to_le_bytes());
                }
                let mut weight = vec![0; config.weight_bytes() as usize];
                for row in 0..n as usize {
                    weight[row * k as usize * 2..row * k as usize * 2 + 2]
                        .copy_from_slice(&0x3f00_u16.to_le_bytes());
                }
                let mut x = context.allocate_device_buffer(config.input_bytes())?;
                let mut w = context.allocate_device_buffer(config.weight_bytes())?;
                let mut y = context.allocate_device_buffer(config.output_bytes())?;
                x.upload_from_slice(0, &input, &mut staging, &mut stream)?;
                w.upload_from_slice(0, &weight, &mut staging, &mut stream)?;
                y.upload_from_slice(
                    0,
                    &vec![0x5a; config.output_bytes() as usize],
                    &mut staging,
                    &mut stream,
                )?;
                {
                    let mut invalid = GemmParams {
                        input: CudaBufferSpan::new(
                            &x,
                            CudaDType::BF16,
                            0,
                            config.input_bytes() - 2,
                        )?,
                        weight: CudaBufferSpan::new(&w, CudaDType::BF16, 0, config.weight_bytes())?,
                        output: CudaBufferSpanMut::new(
                            &mut y,
                            CudaDType::BF16,
                            0,
                            config.output_bytes(),
                        )?,
                        workspace: None,
                    };
                    assert!(plan.execute(&mut invalid, &mut stream).is_err());
                    assert!(!plan.is_poisoned());
                }
                {
                    let mut params = GemmParams {
                        input: CudaBufferSpan::new(&x, CudaDType::BF16, 0, config.input_bytes())?,
                        weight: CudaBufferSpan::new(&w, CudaDType::BF16, 0, config.weight_bytes())?,
                        output: CudaBufferSpanMut::new(
                            &mut y,
                            CudaDType::BF16,
                            0,
                            config.output_bytes(),
                        )?,
                        workspace: None,
                    };
                    plan.execute(&mut params, &mut stream)?;
                }
                let mut output = vec![0; config.output_bytes() as usize];
                y.download_to_slice(0, &mut output, &mut staging, &mut stream)?;
                for row in 0..batch as usize {
                    let expected = ((((row + 1) as f32) * 0.5).to_bits() >> 16) as u16;
                    let base = row * config.output_stride() as usize * 2;
                    for col in 0..n as usize {
                        assert_eq!(
                            u16::from_le_bytes([
                                output[base + col * 2],
                                output[base + col * 2 + 1]
                            ]),
                            expected
                        );
                    }
                    assert!(
                        output[base + n as usize * 2..base + config.output_stride() as usize * 2]
                            .iter()
                            .all(|v| *v == 0x5a)
                    );
                }
                let mut readback = vec![0; input.len()];
                x.download_to_slice(0, &mut readback, &mut staging, &mut stream)?;
                assert_eq!(readback, input);
                plan.close()?;
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 20);
    println!(
        "STRIDED_RUST_OWNER cases=20 exact_impulse_outputs=true padding_preserved=true invalid_span_reuse=true"
    );
    Ok(())
}
