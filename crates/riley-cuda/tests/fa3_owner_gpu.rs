#![cfg(all(feature = "cuda", riley_fa3))]
use riley_cuda::{CudaErrorKind, CudaRuntime, FA3_COMPILED, Fa3Batch, PreparedFa3Attention};

#[test]
fn unsupported_device_releases_native_leases() -> Result<(), Box<dyn std::error::Error>> {
    assert!(FA3_COMPILED);
    let runtime = CudaRuntime::initialize()?;
    let device = runtime.device(0)?;
    if device.properties().compute_capability() == (9, 0) {
        println!(
            "skip: this rejection test needs a non-Hopper device; no attention numerical test executed"
        );
        return Ok(());
    }
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    let mut q = context.allocate_device_buffer(9 * 64 * 2)?;
    let mut k = context.allocate_device_buffer(3 * 16 * 64 * 2)?;
    let mut v = context.allocate_device_buffer(3 * 16 * 64 * 2)?;
    let mut o = context.allocate_device_buffer(9 * 64 * 2)?;
    let batch = Fa3Batch::new(true, 1, vec![0, 1], vec![1], vec![0, 1], vec![0])?;
    for _ in 0..3 {
        let result = PreparedFa3Attention::new(&batch, &mut stream, &mut q, &mut k, &mut v, &mut o);
        match result {
            Err(e) => assert_eq!(e.kind(), CudaErrorKind::NotSupported),
            Ok(_) => panic!("non-Hopper accepted"),
        }
    }
    q.close()?;
    k.close()?;
    v.close()?;
    o.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    println!(
        "FA3 Rust/native rejection: three creates, all buffers and stream closed; Hopper runtime skipped"
    );
    Ok(())
}
