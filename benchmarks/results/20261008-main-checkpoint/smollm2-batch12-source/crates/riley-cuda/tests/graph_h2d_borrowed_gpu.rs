use riley_cuda::{BorrowedH2DGraph, BorrowedH2DResources, CudaRuntime};
#[test]
#[ignore = "requires CUDA GPU"]
fn borrowed_h2d_exact_slabs_reject_refresh_and_drop() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaRuntime::initialize()?.device(0)?.create_context()?;
    let mut stream = context.create_stream()?;
    let mut source = context.allocate_pinned_host_buffer(257)?;
    let mut device = context.allocate_device_buffer(257)?;
    let mut short = context.allocate_device_buffer(256)?;
    let mut staging = context.allocate_pinned_host_buffer(4096)?;
    let stats = context.allocation_stats()?;
    assert!(
        BorrowedH2DGraph::prepare(BorrowedH2DResources {
            stream: &mut stream,
            input: &mut source,
            output: &mut short
        })
        .is_err()
    );
    for seed in 0..3_u8 {
        let mut bytes: Vec<u8> = (0..257)
            .map(|i| (i as u8).wrapping_mul(17).wrapping_add(seed))
            .collect();
        source.write(0, &bytes)?;
        device.upload_from_slice(0, &vec![0xff; 257], &mut staging, &mut stream)?;
        let mut graph = BorrowedH2DGraph::prepare(BorrowedH2DResources {
            stream: &mut stream,
            input: &mut source,
            output: &mut device,
        })?;
        assert!(graph.replay(&bytes[..256]).is_err());
        for _ in 0..16 {
            bytes[0] = bytes[0].wrapping_add(1);
            graph.replay(&bytes)?;
        }
        if seed == 1 {
            drop(graph);
        } else {
            graph.close()?;
        }
        let mut actual = vec![0; 257];
        device.download_to_slice(0, &mut actual, &mut staging, &mut stream)?;
        assert_eq!(actual, bytes);
        source.read(0, &mut actual)?;
        assert_eq!(actual, bytes);
        assert_eq!(stats, context.allocation_stats()?);
    }
    short.close()?;
    device.close()?;
    source.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
