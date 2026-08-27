#![cfg(feature = "cuda-test-fault-injection")]

use std::error::Error;
use std::process::Command;

use riley_cuda::{CudaErrorKind, CudaErrorStage, CudaMemoryFault, CudaRuntime};

const CHILD_ENV: &str = "RILEY_CUDA_MEMORY_FAULT_CHILD";
const CASES: [&str; 4] = [
    "create-rollback-ambiguous",
    "explicit-close-ambiguous",
    "deferred-submission-error",
    "completion-restore-ambiguous",
];

fn context() -> Result<riley_cuda::CudaContext, Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    assert!(
        runtime.device_count() > 0,
        "remote runner has no CUDA device"
    );
    Ok(runtime.device(0)?.create_context()?)
}

fn assert_runtime_error(error: &riley_cuda::CudaError, stage: CudaErrorStage) {
    assert_eq!(error.kind(), CudaErrorKind::Runtime);
    assert_eq!(error.stage(), stage);
    assert_ne!(error.native_code(), 0);
}

fn create_rollback_ambiguous() -> Result<(), Box<dyn Error>> {
    let context = context()?;
    context.reset_memory_fault_injection()?;

    context.arm_memory_fault(CudaMemoryFault::DeviceCreateRollbackAmbiguous)?;
    let error = match context.allocate_device_buffer(4096) {
        Ok(_) => panic!("injected post-allocation create must fail"),
        Err(error) => error,
    };
    assert_runtime_error(&error, CudaErrorStage::Create);

    context.arm_memory_fault(CudaMemoryFault::PinnedCreateRollbackAmbiguous)?;
    let error = match context.allocate_pinned_host_buffer(8192) {
        Ok(_) => panic!("injected post-allocation pinned create must fail"),
        Err(error) => error,
    };
    assert_runtime_error(&error, CudaErrorStage::Create);

    let accounting = context.allocation_stats()?;
    assert_eq!(accounting.device_live_bytes(), 4096);
    assert_eq!(accounting.device_live_allocations(), 1);
    assert_eq!(accounting.pinned_host_live_bytes(), 8192);
    assert_eq!(accounting.pinned_host_live_allocations(), 1);
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 2);
    assert_eq!(injected.device_free_attempts(), 1);
    assert_eq!(injected.pinned_free_attempts(), 1);
    assert_eq!(injected.copy_use_release_attempts(), 0);
    assert_eq!(injected.armed_fault(), None);

    let error = context
        .close()
        .expect_err("unresolved rollback allocations must reject context close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    Ok(())
}

fn explicit_close_ambiguous() -> Result<(), Box<dyn Error>> {
    let context = context()?;
    let device = context.allocate_device_buffer(4096)?;
    let pinned = context.allocate_pinned_host_buffer(8192)?;
    context.reset_memory_fault_injection()?;

    context.arm_memory_fault(CudaMemoryFault::DeviceCloseAmbiguous)?;
    let error = device
        .close()
        .expect_err("ambiguous cudaFree must be reported");
    assert_runtime_error(&error, CudaErrorStage::Close);

    context.arm_memory_fault(CudaMemoryFault::PinnedCloseAmbiguous)?;
    let error = pinned
        .close()
        .expect_err("ambiguous cudaFreeHost must be reported");
    assert_runtime_error(&error, CudaErrorStage::Close);

    let accounting = context.allocation_stats()?;
    assert_eq!(accounting.device_live_allocations(), 1);
    assert_eq!(accounting.pinned_host_live_allocations(), 1);
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 2);
    assert_eq!(injected.device_free_attempts(), 1);
    assert_eq!(injected.pinned_free_attempts(), 1);

    let error = context
        .close()
        .expect_err("ambiguous explicit frees must reject context close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    Ok(())
}

fn deferred_submission_error() -> Result<(), Box<dyn Error>> {
    let context = context()?;
    let mut stream = context.create_stream()?;
    let mut device = context.allocate_device_buffer(32)?;
    let mut pinned = context.allocate_pinned_host_buffer(32)?;
    let expected: Vec<u8> = (0_u8..32).collect();
    pinned.write(0, &expected)?;
    let live_before = context.allocation_stats()?;
    context.reset_memory_fault_injection()?;
    context.arm_memory_fault(CudaMemoryFault::CopyDeferredSubmissionError)?;

    let error = device
        .copy_from_pinned_async(0, &mut pinned, 0, 32, &mut stream)?
        .synchronize()
        .expect_err("deferred submission error must survive confirmed completion");
    assert_runtime_error(&error, CudaErrorStage::Copy);
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 1);
    assert_eq!(injected.copy_use_release_attempts(), 1);
    assert_eq!(context.allocation_stats()?, live_before);

    // Rust and native reservations were released exactly once despite the
    // original error, so every resource is immediately reusable.
    pinned.write(0, &[0; 32])?;
    device
        .copy_to_pinned_async(0, &mut pinned, 0, 32, &mut stream)?
        .synchronize()?;
    assert_eq!(pinned.to_vec()?, expected);
    assert_eq!(context.allocation_stats()?, live_before);
    device.close()?;
    pinned.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}

fn completion_restore_ambiguous() -> Result<(), Box<dyn Error>> {
    let context = context()?;
    let mut stream = context.create_stream()?;
    let mut device = context.allocate_device_buffer(32)?;
    let mut pinned = context.allocate_pinned_host_buffer(32)?;
    pinned.write(0, &[7; 32])?;
    context.reset_memory_fault_injection()?;

    let pending = device.copy_from_pinned_async(0, &mut pinned, 0, 32, &mut stream)?;
    context.arm_memory_fault(CudaMemoryFault::CopyCompletionRestoreAmbiguous)?;
    let error = pending
        .synchronize()
        .expect_err("ambiguous completion/restoration must retain all leases");
    assert_runtime_error(&error, CudaErrorStage::Synchronize);
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 1);
    assert_eq!(injected.copy_use_release_attempts(), 0);

    let error = pinned
        .write(0, &[1])
        .expect_err("forgotten native token must keep Rust pinned state busy");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    let error = device
        .close()
        .expect_err("active native/Rust device lease must reject close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    let error = pinned
        .close()
        .expect_err("active native/Rust pinned lease must reject close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    let error = stream
        .close()
        .expect_err("active native stream lease must reject close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.device_free_attempts(), 0);
    assert_eq!(injected.pinned_free_attempts(), 0);
    assert_eq!(injected.copy_use_release_attempts(), 0);
    let error = context
        .close()
        .expect_err("poisoned context and unresolved children must reject close");
    assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn memory_fault_cases_are_subprocess_isolated() -> Result<(), Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let parent_pid = std::process::id();
    for case in CASES {
        let mut child = Command::new(&executable)
            .args([
                "--ignored",
                "--exact",
                "memory_fault_subprocess",
                "--nocapture",
            ])
            .env(CHILD_ENV, case)
            .spawn()?;
        let child_pid = child.id();
        println!(
            "riley-cuda-memory-fault-case case={case} event=spawn parent_pid={parent_pid} child_pid={child_pid}"
        );
        let status = child.wait()?;
        println!(
            "riley-cuda-memory-fault-case case={case} event=joined parent_pid={parent_pid} child_pid={child_pid} exit_code={}",
            status.code().unwrap_or(-1)
        );
        assert!(status.success(), "fault subprocess failed: {case}");
    }
    Ok(())
}

#[test]
#[ignore = "remote GPU"]
fn memory_fault_subprocess() -> Result<(), Box<dyn Error>> {
    let case = std::env::var(CHILD_ENV)?;
    let child_pid = std::process::id();
    println!("riley-cuda-memory-fault-case case={case} event=start child_pid={child_pid}");
    let result = match case.as_str() {
        "create-rollback-ambiguous" => create_rollback_ambiguous(),
        "explicit-close-ambiguous" => explicit_close_ambiguous(),
        "deferred-submission-error" => deferred_submission_error(),
        "completion-restore-ambiguous" => completion_restore_ambiguous(),
        _ => panic!("unknown memory fault child case: {case}"),
    };
    if result.is_ok() {
        println!("riley-cuda-memory-fault-case case={case} event=passed child_pid={child_pid}");
    }
    result
}
