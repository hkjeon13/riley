use rustinfer_cuda::{EXPECTED_ABI_VERSION, abi_version, build_info};

#[test]
fn native_symbols_link_without_device_initialization() -> rustinfer_core::Result<()> {
    assert_eq!(abi_version()?, EXPECTED_ABI_VERSION);

    let info = build_info()?;
    assert!(info.contains("rustinfer-cuda-native"));
    assert!(info.contains("abi=1"));
    assert!(info.contains("nvcc="));
    Ok(())
}
