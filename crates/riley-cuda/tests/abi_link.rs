use riley_cuda::{EXPECTED_ABI_VERSION, abi_version, build_info};

#[test]
fn native_symbols_link_without_device_initialization() -> riley_core::Result<()> {
    assert_eq!(abi_version()?, EXPECTED_ABI_VERSION);

    let info = build_info()?;
    assert!(info.starts_with("riley-cuda-native abi=1 nvcc="));
    Ok(())
}
