//! Small real-CUDA COW transport gates; no model/serving performance claim.
use riley_cuda::{CudaContext, CudaRuntime};
use riley_runtime::paged_kv::{CopyOutcome, KvBlockPool, KvIdentity, KvLayout, PrefixDescriptor};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn context() -> Result<CudaContext, Box<dyn Error>> {
    let runtime = CudaRuntime::initialize()?;
    Ok(runtime
        .devices()?
        .into_iter()
        .next()
        .ok_or("CUDA device required")?
        .create_context()?)
}

// Independent dense BF16 reference: 3 layers, 8 pages, 2 heads, 16 tokens, D8.
fn offset(layer: usize, page: u32, head: usize) -> usize {
    (((layer * 8 + page as usize) * 2 + head) * 16 * 8) * 2
}

fn run_case(length: usize, mode: &str) -> TestResult {
    let context = context()?;
    let mut stream = context.create_stream()?;
    let layout = KvLayout::checked(3, 8, 2, 8)?;
    let bytes = layout.bytes_per_kind();
    let mut keys = context.allocate_device_buffer(bytes)?;
    let mut values = context.allocate_device_buffer(bytes)?;
    let mut staging = context.allocate_pinned_host_buffer(bytes)?;
    let original: [Vec<u8>; 2] = [0, 1].map(|kind| {
        (0..bytes as usize)
            .map(|i| ((i * 131 + kind * 17 + i / 257) % 251) as u8)
            .collect()
    });
    for (buffer, pattern) in [(&mut keys, &original[0]), (&mut values, &original[1])] {
        staging.write(0, pattern)?;
        buffer
            .copy_from_pinned_async(0, &mut staging, 0, bytes, &mut stream)?
            .synchronize()?;
    }
    let mut pool = KvBlockPool::new(layout)?;
    let mut source = pool.create_sequence(64)?;
    let reservation = source.reserve_to(&mut pool, length)?;
    source.commit(&mut pool, reservation)?;
    let descriptor = PrefixDescriptor::new(
        KvIdentity {
            model_revision: [1; 32],
            numerical_profile: [2; 32],
            position_encoding: [3; 32],
            partition: [4; 32],
            layout: layout.into(),
        },
        0,
        &(0..length as u32).collect::<Vec<_>>(),
    )?;
    let mut export = pool.export_prefix(&source, descriptor.clone())?;
    let mut target = pool.create_sequence(64)?;
    target.import_prefix(&mut pool, &export, &descriptor)?;
    let ticket = target
        .begin_copy_on_write(&mut pool)?
        .ok_or("COW expected")?;
    let (from, to, valid) = ticket.copy_range().ok_or("copy range expected")?;
    let mut expected = original.clone();
    for layer in 0..3 {
        for head in 0..2 {
            for kind in 0..2 {
                if mode == "partial" && (layer != 0 || head != 0 || kind != 0) {
                    continue;
                }
                let start = offset(layer, from.physical_index(), head);
                let destination = offset(layer, to.physical_index(), head);
                let count = usize::from(valid) * 8 * 2;
                expected[kind][destination..destination + count]
                    .copy_from_slice(&original[kind][start..start + count]);
            }
        }
    }
    #[cfg(feature = "cuda-test-fault-injection")]
    if mode == "partial" {
        context.reset_memory_fault_injection()?;
        context.arm_memory_fault(riley_cuda::CudaMemoryFault::CopyDeferredSubmissionError)?;
    }
    let mut pending = ticket
        .submit_cuda(&mut keys, &mut values, &mut stream)
        .map_err(|(_, error)| error)?;
    assert!(
        target.block_table().is_err(),
        "copy destination cannot be decoded before event resolution"
    );
    source.close(&mut pool)?;
    pool.release_prefix(&mut export)?;
    let old_tail = target.block_id((length - 1) / 16).unwrap();
    if mode == "orphan" {
        let reclaim = target.abandon_for_reclaim();
        pool.reclaim_sequence(&reclaim)?;
        assert!(pool.stats().allocated_block_count() >= 2);
        assert_eq!(
            pending.synchronize_orphan(&mut pool)?,
            Some(CopyOutcome::Discarded)
        );
    } else if mode == "cancel" {
        pending.cancel();
        target.reset(&mut pool)?;
        let next = target.reserve_to(&mut pool, 1)?;
        assert_eq!(
            pending.synchronize(&mut target, &mut pool)?,
            Some(CopyOutcome::Discarded)
        );
        assert!(
            target.block_table().is_err(),
            "new reservation must survive late completion"
        );
        target.commit(&mut pool, next)?;
        target.close(&mut pool)?;
    } else if mode == "partial" {
        assert!(pending.synchronize(&mut target, &mut pool).is_err());
        assert_eq!(target.block_id((length - 1) / 16), Some(old_tail));
        assert_eq!(target.block_table()?.logical_length() as usize, length);
        assert!(
            pending.synchronize(&mut target, &mut pool).is_err(),
            "failure remains sticky"
        );
        target.close(&mut pool)?;
        #[cfg(feature = "cuda-test-fault-injection")]
        assert_eq!(context.memory_fault_stats()?.faults_fired(), 1);
    } else {
        let result = pending.poll(&mut target, &mut pool)?;
        let result = if result.is_none() {
            pending.synchronize(&mut target, &mut pool)?
        } else {
            result
        };
        assert_eq!(result, Some(CopyOutcome::Published));
        assert_eq!(target.block_id((length - 1) / 16), Some(to));
        assert_eq!(
            pending.poll(&mut target, &mut pool)?,
            Some(CopyOutcome::AlreadyCompleted)
        );
        let append = target.reserve_to(&mut pool, length + 1)?;
        target.commit(&mut pool, append)?;
        target.close(&mut pool)?;
    }
    drop(pending);
    assert_eq!(pool.stats().allocated_block_count(), 0);
    for (kind, buffer) in [&mut keys, &mut values].into_iter().enumerate() {
        buffer
            .copy_to_pinned_async(0, &mut staging, 0, bytes, &mut stream)?
            .synchronize()?;
        assert_eq!(
            staging.to_vec()?,
            expected[kind],
            "whole K/V buffer and unused-tail guards, mode={mode}"
        );
    }
    keys.close()?;
    values.close()?;
    staging.close()?;
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    println!(
        "kv-copy mode={mode} tokens={length} bytes_verified={} host_pages=0 cuda_allocations=0",
        2 * bytes
    );
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU; explicit remote execution"]
fn kv_page_event_copy_and_all_tail_guards() -> TestResult {
    for length in [1, 15, 17, 31, 33] {
        run_case(length, "publish")?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU; explicit remote execution"]
fn kv_page_cancel_reset_and_orphan_drain() -> TestResult {
    run_case(17, "cancel")?;
    run_case(17, "orphan")
}

#[cfg(feature = "cuda-test-fault-injection")]
#[test]
#[ignore = "requires test-only CUDA fault injection; explicit remote execution"]
fn kv_page_partial_submission_never_publishes() -> TestResult {
    run_case(17, "partial")
}

#[cfg(feature = "cuda-test-fault-injection")]
#[test]
#[ignore = "requires isolated test-only CUDA fault injection; explicit remote execution"]
fn kv_page_ambiguous_completion_retains_all_holds() -> TestResult {
    const CHILD: &str = "RILEY_KV_COPY_AMBIGUOUS_CHILD";
    if std::env::var(CHILD).as_deref() != Ok("1") {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--ignored",
                "--exact",
                "kv_page_ambiguous_completion_retains_all_holds",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()?;
        assert!(
            status.success(),
            "isolated ambiguous-completion gate failed"
        );
        return Ok(());
    }
    let context = context()?;
    let mut stream = context.create_stream()?;
    let layout = KvLayout::checked(1, 4, 1, 8)?;
    let bytes = layout.bytes_per_kind();
    let mut keys = context.allocate_device_buffer(bytes)?;
    let mut values = context.allocate_device_buffer(bytes)?;
    let mut staging = context.allocate_pinned_host_buffer(bytes)?;
    staging.write(0, &vec![7; bytes as usize])?;
    for buffer in [&mut keys, &mut values] {
        buffer
            .copy_from_pinned_async(0, &mut staging, 0, bytes, &mut stream)?
            .synchronize()?;
    }
    staging.close()?;
    let mut pool = KvBlockPool::new(layout)?;
    let mut target = pool.create_sequence(16)?;
    let reservation = target.reserve_to(&mut pool, 1)?;
    target.commit(&mut pool, reservation)?;
    let descriptor = PrefixDescriptor::new(
        KvIdentity {
            model_revision: [1; 32],
            numerical_profile: [2; 32],
            position_encoding: [3; 32],
            partition: [4; 32],
            layout: layout.into(),
        },
        0,
        &[1],
    )?;
    let mut export = pool.export_prefix(&target, descriptor)?;
    let ticket = target.begin_copy_on_write(&mut pool)?.unwrap();
    let mut pending = ticket
        .submit_cuda(&mut keys, &mut values, &mut stream)
        .map_err(|(_, error)| error)?;
    context.reset_memory_fault_injection()?;
    context.arm_memory_fault(riley_cuda::CudaMemoryFault::CopyCompletionRestoreAmbiguous)?;
    assert!(pending.synchronize(&mut target, &mut pool).is_err());
    assert!(
        target.block_table().is_err(),
        "unknown completion cannot publish"
    );
    drop(pending);
    let reclaim = target.abandon_for_reclaim();
    pool.reclaim_sequence(&reclaim)?;
    pool.release_prefix(&mut export)?;
    assert_eq!(
        pool.stats().allocated_block_count(),
        2,
        "source and staging retained after ambiguity"
    );
    let injected = context.memory_fault_stats()?;
    assert_eq!(injected.faults_fired(), 1);
    assert_eq!(injected.copy_use_release_attempts(), 0);
    assert_eq!(
        keys.close().unwrap_err().kind(),
        riley_cuda::CudaErrorKind::InvalidState
    );
    assert_eq!(
        values.close().unwrap_err().kind(),
        riley_cuda::CudaErrorKind::InvalidState
    );
    assert_eq!(
        stream.close().unwrap_err().kind(),
        riley_cuda::CudaErrorKind::InvalidState
    );
    assert_eq!(
        context.close().unwrap_err().kind(),
        riley_cuda::CudaErrorKind::InvalidState
    );
    println!("kv-copy ambiguous_completion=true retained_host_pages=2 copy_use_releases=0 cleanup=process_exit");
    Ok(())
}
