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
        for batch in [2, 4, 8] {
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
                let reservation =
                    riley_cuda::BorrowedGraphResourceReservation::reserve_with_strided(
                        riley_cuda::BorrowedGraphResourceParents {
                            stream: &mut stream,
                            devices: vec![&mut x, &mut w, &mut y],
                            pinned: vec![&mut staging],
                            plans: vec![],
                        },
                        vec![&mut plan],
                    )?;
                if padded {
                    reservation.close()?;
                } else {
                    drop(reservation);
                }
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
                y.upload_from_slice(
                    0,
                    &vec![0x5a; config.output_bytes() as usize],
                    &mut staging,
                    &mut stream,
                )?;
                let mut graph = riley_cuda::BorrowedStridedGemmGraph::prepare(
                    riley_cuda::BorrowedStridedGemmResources {
                        stream: &mut stream,
                        plan: &mut plan,
                        input: &mut x,
                        weight: &mut w,
                        output: &mut y,
                        workspace: None,
                    },
                )?;
                for _ in 0..3 {
                    graph.replay()?;
                }
                if padded {
                    graph.close()?;
                } else {
                    drop(graph);
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
    assert_eq!(checked, 30);
    println!(
        "STRIDED_RUST_OWNER cases=30 exact_impulse_outputs=true padding_preserved=true invalid_span_reuse=true strided_graph_replays=90 explicit_and_drop_close=true"
    );
    Ok(())
}

#[test]
#[ignore = "requires qualified RTX 4090 CUDA runtime"]
fn dense_n8_graph_rows_match_independent_m1() -> Result<(), Box<dyn std::error::Error>> {
    use riley_cuda::CudaGemmConfig;
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(1 << 20)?;
    let mut state = 0x1a29_78b3_u32;
    let mut random_bytes = |count: usize| {
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let bits =
                (((state >> 31) << 15) | ((125 + state % 5) << 7) | ((state >> 16) & 127)) as u16;
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    };
    let mut rows_checked = 0;
    for (n, k) in [
        (960, 576),
        (3072, 576),
        (576, 576),
        (576, 1536),
        (49152, 576),
    ] {
        for _pattern in 0..2 {
            let config = CudaStridedGemmConfig::new(n, k, 8, false)?;
            let input = random_bytes(8 * k as usize);
            let weight = random_bytes(n as usize * k as usize);
            let mut w = context.allocate_device_buffer(config.weight_bytes())?;
            let mut x1 = context.allocate_device_buffer(k * 2)?;
            let mut y1 = context.allocate_device_buffer(n * 2)?;
            w.upload_from_slice(0, &weight, &mut staging, &mut stream)?;
            let mut one = context.prepare_gemm(CudaGemmConfig::new(1, n, k, 16 * 1024 * 1024)?)?;
            assert_eq!(one.algorithm_metadata().workspace_bytes(), 0);
            let mut expected = Vec::with_capacity(config.output_bytes() as usize);
            for row in 0..8 {
                x1.upload_from_slice(
                    0,
                    &input[row * k as usize * 2..(row + 1) * k as usize * 2],
                    &mut staging,
                    &mut stream,
                )?;
                {
                    let mut params = GemmParams {
                        input: CudaBufferSpan::new(&x1, CudaDType::BF16, 0, k * 2)?,
                        weight: CudaBufferSpan::new(&w, CudaDType::BF16, 0, config.weight_bytes())?,
                        output: CudaBufferSpanMut::new(&mut y1, CudaDType::BF16, 0, n * 2)?,
                        workspace: None,
                    };
                    one.execute(&mut params, &mut stream)?;
                }
                let mut bytes = vec![0; n as usize * 2];
                y1.download_to_slice(0, &mut bytes, &mut staging, &mut stream)?;
                expected.extend(bytes);
            }
            one.close()?;
            for batch in [4, 8] {
                let config = CudaStridedGemmConfig::new(n, k, batch, false)?;
                let input = &input[..config.input_bytes() as usize];
                let expected = &expected[..config.output_bytes() as usize];
                let mut x = context.allocate_device_buffer(config.input_bytes())?;
                let mut y = context.allocate_device_buffer(config.output_bytes())?;
                x.upload_from_slice(0, input, &mut staging, &mut stream)?;
                let mut plan = context.prepare_strided_gemm(config)?;
                y.upload_from_slice(
                    0,
                    &vec![0xa5; config.output_bytes() as usize],
                    &mut staging,
                    &mut stream,
                )?;
                {
                    let mut graph = riley_cuda::BorrowedStridedGemmGraph::prepare(
                        riley_cuda::BorrowedStridedGemmResources {
                            stream: &mut stream,
                            plan: &mut plan,
                            input: &mut x,
                            weight: &mut w,
                            output: &mut y,
                            workspace: None,
                        },
                    )?;
                    for _ in 0..3 {
                        graph.replay()?;
                    }
                    graph.close()?;
                }
                let mut actual = vec![0; config.output_bytes() as usize];
                y.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
                let first = actual
                    .chunks_exact(2)
                    .zip(expected.chunks_exact(2))
                    .position(|(a, b)| a != b);
                assert!(
                    first.is_none(),
                    "dense batch={batch} differs from M1: N={n} K={k} first={first:?}"
                );
                let mut unchanged = vec![0; input.len()];
                x.download_to_slice(0, &mut unchanged, &mut staging, &mut stream)?;
                assert_eq!(unchanged, input);
                plan.close()?;
                rows_checked += batch;
            }
        }
    }
    assert_eq!(rows_checked, 120);
    println!(
        "DENSE_N4_N8_M1 rows=120 graph_replays=60 exact_bf16_outputs=true input_unchanged=true performance_claim=false"
    );
    Ok(())
}
