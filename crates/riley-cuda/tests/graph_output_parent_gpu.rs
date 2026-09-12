use riley_cuda::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_STATUS_NON_FINITE, BorrowedOutputGraph,
    BorrowedOutputResources, CudaRuntime,
};
#[test]
#[ignore = "requires CUDA GPU"]
fn output_parent_offset_d2h_status_tail_and_recovery() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let mut pinned = context.allocate_pinned_host_buffer(64)?;
    let mut input = context.allocate_device_buffer(16)?;
    let mut indices = context.allocate_device_buffer(32)?;
    let mut gathered = context.allocate_device_buffer(16)?;
    let mut output = context.allocate_device_buffer(16)?;
    let logits: Vec<u8> = [1.0_f32, 4.0, 4.0, 0.0, 7.0, 2.0, 1.0, 0.0]
        .into_iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_ne_bytes())
        .collect();
    input.upload_from_slice(0, &logits, &mut staging, &mut stream)?;
    let stats = context.allocation_stats()?;
    assert!(
        BorrowedOutputGraph::prepare(
            BorrowedOutputResources {
                stream: &mut stream,
                input: &mut input,
                indices: &mut indices,
                gathered: &mut gathered,
                output: &mut output,
                pinned: &mut pinned
            },
            2,
            &[1, 0],
            4,
            1
        )
        .is_err()
    );
    for (case, map) in [[1_u32, 0], [1, 99], [0, 1]].into_iter().enumerate() {
        let mut bytes = vec![0xa5; 32];
        for (i, v) in map.into_iter().enumerate() {
            bytes[8 + i * 4..12 + i * 4].copy_from_slice(&v.to_ne_bytes());
        }
        indices.upload_from_slice(0, &bytes, &mut staging, &mut stream)?;
        pinned.write(0, &[0x5a; 64])?;
        let host_map = if case == 2 { [0, 1] } else { [1, 0] };
        let mut graph = BorrowedOutputGraph::prepare(
            BorrowedOutputResources {
                stream: &mut stream,
                input: &mut input,
                indices: &mut indices,
                gathered: &mut gathered,
                output: &mut output,
                pinned: &mut pinned,
            },
            2,
            &host_map,
            4,
            8,
        )?;
        for _ in 0..16 {
            let records = graph.replay()?;
            let words: Vec<u32> = records
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            let expected = match case {
                0 => vec![0, 0, 1, 0],
                1 => vec![
                    0,
                    0,
                    BF16_ARGMAX_INVALID_TOKEN_ID,
                    BF16_ARGMAX_STATUS_NON_FINITE,
                ],
                _ => vec![1, 0, 0, 0],
            };
            assert_eq!(words, expected);
        }
        if case == 1 {
            drop(graph);
        } else {
            graph.close()?;
        }
        let mut tail = [0; 48];
        pinned.read(16, &mut tail)?;
        assert_eq!(tail, [0x5a; 48]);
        let mut after = vec![0; 32];
        indices.download_to_slice(0, &mut after, &mut staging, &mut stream)?;
        assert_eq!(after, bytes);
        let mut after = vec![0; 16];
        input.download_to_slice(0, &mut after, &mut staging, &mut stream)?;
        assert_eq!(after, logits);
        assert_eq!(stats, context.allocation_stats()?);
    }
    output.close()?;
    gathered.close()?;
    indices.close()?;
    input.close()?;
    pinned.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
