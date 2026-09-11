use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BF16_ARGMAX_STATUS_SUCCESS,
    BorrowedArgmaxGraph, BorrowedArgmaxResources, CudaRuntime,
};

#[test]
#[ignore = "requires CUDA GPU"]
fn borrowed_argmax_ties_nonfinite_completion_and_recovery() -> Result<(), Box<dyn std::error::Error>>
{
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let mut logits = context.allocate_device_buffer(3 * 7 * 2)?;
    let mut results = context.allocate_device_buffer(32)?;
    let mut short = context.allocate_device_buffer(7)?;
    let stats = context.allocation_stats()?;
    assert!(
        BorrowedArgmaxGraph::prepare(
            BorrowedArgmaxResources {
                stream: &mut stream,
                input: &mut logits,
                output: &mut short
            },
            3,
            7
        )
        .is_err()
    );
    for (index, special) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 9.0]
        .into_iter()
        .enumerate()
    {
        let mut values = vec![1.0_f32, 2.0, 2.0, -1.0, 0.0, 1.0, 0.0];
        values.extend([-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0]);
        values.extend([0.0, 1.0, special, 2.0, 2.0, -1.0, 0.0]);
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_ne_bytes())
            .collect();
        logits.upload_from_slice(0, &bytes, &mut staging, &mut stream)?;
        results.upload_from_slice(0, &[0xa5; 32], &mut staging, &mut stream)?;
        let mut graph = BorrowedArgmaxGraph::prepare(
            BorrowedArgmaxResources {
                stream: &mut stream,
                input: &mut logits,
                output: &mut results,
            },
            3,
            7,
        )?;
        for _ in 0..16 {
            graph.replay()?;
        }
        if index % 2 == 0 {
            graph.close()?;
        } else {
            drop(graph);
        }
        let mut actual = [0; 32];
        results.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        for (row, (token, status)) in [
            (1, BF16_ARGMAX_STATUS_SUCCESS),
            (0, BF16_ARGMAX_STATUS_SUCCESS),
            if special.is_finite() {
                (2, BF16_ARGMAX_STATUS_SUCCESS)
            } else {
                (BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE)
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                u32::from_ne_bytes(actual[row * 8..row * 8 + 4].try_into()?),
                token
            );
            assert_eq!(
                u32::from_ne_bytes(actual[row * 8 + 4..row * 8 + 8].try_into()?),
                status
            );
        }
        assert_eq!(&actual[24..], &[0xa5; 8]);
        let mut input_after = vec![0; bytes.len()];
        logits.download_to_slice(0, &mut input_after, &mut staging, &mut stream)?;
        assert_eq!(input_after, bytes);
        assert_eq!(stats, context.allocation_stats()?);
    }
    short.close()?;
    results.close()?;
    logits.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
