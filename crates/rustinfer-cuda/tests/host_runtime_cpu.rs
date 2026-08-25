#![cfg(not(feature = "cuda"))]

use std::error::Error as _;
use std::hint::black_box;

use rustinfer_cuda::{
    CRATE_ROLE, CUDA_ENABLED, CudaAllocationStats, CudaContext, CudaDevice, CudaDeviceBuffer,
    CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaEvent, CudaKernel,
    CudaPendingD2H, CudaPendingFill, CudaPendingH2D, CudaPinnedHostBuffer, CudaRuntime, CudaStream,
    DeviceProperties, EXPECTED_ABI_VERSION,
};

fn assert_send<T: Send>() {}

fn assert_send_sync<T: Send + Sync>() {}

fn assert_standard_error<T: std::error::Error + Send + Sync>() {}

#[test]
fn feature_off_initialize_returns_an_actionable_owned_error() {
    let error = match CudaRuntime::initialize() {
        Ok(_) => panic!("CUDA initialization unexpectedly succeeded without the `cuda` feature"),
        Err(error) => error,
    };

    assert_eq!(error.kind(), CudaErrorKind::Unavailable);
    assert_eq!(error.domain(), CudaErrorDomain::Rust);
    assert_eq!(error.stage(), CudaErrorStage::Initialize);
    assert_eq!(error.native_code(), 0);
    assert_eq!(error.operation(), "CudaRuntime::initialize");
    assert!(error.message().contains("`cuda` feature"));
    assert!(error.message().contains("`--features cuda`"));
    assert!(error.message().contains("CUDA toolkit"));

    let diagnostic = error.to_string();
    assert!(diagnostic.contains("CUDA Unavailable error"));
    assert!(diagnostic.contains("CudaRuntime::initialize"));
    assert!(diagnostic.contains("Rust/Initialize"));
    assert!(diagnostic.contains("native code 0"));
    assert!(diagnostic.contains("`cuda` feature"));
    assert!(diagnostic.contains("`--features cuda`"));

    // The diagnostic is owned and remains available through the standard
    // error trait; it does not borrow a native caller buffer.
    assert!(error.source().is_none());
}

#[test]
fn feature_off_build_exposes_only_compile_time_contract_metadata() {
    assert!(!black_box(CUDA_ENABLED));
    assert_eq!(EXPECTED_ABI_VERSION, 1);
    assert_eq!(CRATE_ROLE, "native CUDA C ABI and host-runtime boundary");
}

#[test]
fn public_ownership_types_keep_their_thread_movement_contracts() {
    assert_send_sync::<CudaRuntime>();
    assert_send_sync::<CudaDevice>();
    assert_send_sync::<DeviceProperties>();
    assert_send_sync::<CudaContext>();
    assert_send_sync::<CudaKernel>();
    assert_standard_error::<CudaError>();

    assert_send::<CudaStream>();
    assert_send::<CudaEvent>();
    assert_send::<CudaPendingFill<'static>>();
    assert_send::<CudaDeviceBuffer>();
    assert_send::<CudaPinnedHostBuffer>();
    assert_send::<CudaPendingH2D<'static>>();
    assert_send::<CudaPendingD2H<'static>>();
    assert_send_sync::<CudaAllocationStats>();
}
