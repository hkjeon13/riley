use super::*;
unsafe extern "C" {
    fn riley_cuda_graph_resources_test_replay_fault(
        owner: *mut RawGraphResources,
        fault: u32,
        error: *mut ErrorInfo,
    ) -> i32;
}
#[test]
#[ignore = "requires CUDA test-fault-injection; ambiguous completion is process-isolated"]
fn aggregate_replay_failure_retains_or_releases_by_completion() -> CudaResult<()> {
    const CHILD: &str = "RILEY_AGGREGATE_REPLAY_FAULT_CHILD";
    let Ok(value) = std::env::var(CHILD) else {
        for fault in [1, 2] {
            let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args(["--exact", "ffi::aggregate_replay_fault_tests::aggregate_replay_failure_retains_or_releases_by_completion", "--ignored", "--nocapture", "--test-threads=1"])
                .env(CHILD, fault.to_string()).output().expect("isolated fault process");
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            println!("G03_REPLAY_FAULT fault={fault} isolated=true passed=true");
        }
        return Ok(());
    };
    let fault = value.parse::<u32>().expect("test fault id");
    let _runtime = crate::CudaRuntime::initialize()?;
    let mut context = ContextHandle::create(0)?;
    let mut stream = StreamHandle::create(&context)?;
    let mut first = DeviceBufferHandle::create(&context, 32)?;
    let mut second = DeviceBufferHandle::create(&context, 32)?;
    let mut input = PinnedHostBufferHandle::create(&context, 32)?;
    let mut output = PinnedHostBufferHandle::create(&context, 32)?;
    let mut graph =
        GraphResourcesHandle::reserve(&stream, &[&first, &second], &[&input, &output], &[])?;
    graph.record_transfer(&input, &first, &second, &output)?;
    graph.replay_transfer(&[3; 32])?;
    let mut bytes = [0; 32];
    graph.read_transfer(&mut bytes)?;
    assert_eq!(bytes, [3; 32]);
    let mut error = ErrorInfo::new();
    // SAFETY: test-only hook receives this live, thread-confined retained owner.
    let status = unsafe {
        riley_cuda_graph_resources_test_replay_fault(
            graph.pointer.map_or(ptr::null_mut(), NonNull::as_ptr),
            fault,
            &mut error,
        )
    };
    status_result(status, "arm aggregate test replay fault", &error)?;
    assert!(graph.replay_transfer(&[7; 32]).is_err());
    assert!(graph.read_transfer(&mut bytes).is_err());
    assert!(graph.replay_transfer(&[9; 32]).is_err());
    if fault == 2 {
        assert!(graph.close().is_err());
        assert!(first.close().is_err());
        assert!(second.close().is_err());
        assert!(input.close().is_err());
        assert!(output.close().is_err());
        assert!(stream.close().is_err());
        assert!(context.close().is_err());
        // Intentionally retained native allocations live until this child exits.
    } else {
        graph.close()?;
        graph.close()?;
        first.close()?;
        second.close()?;
        input.close()?;
        output.close()?;
        stream.close()?;
        context.close()?;
    }
    Ok(())
}
