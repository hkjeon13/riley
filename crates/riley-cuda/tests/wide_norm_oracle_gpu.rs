//! Byte-exact RMSNorm comparison with separately generated PyTorch CUDA oracles.
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, RmsNormParams,
    hugging_face_qwen_wide_rms_norm_row_groups,
};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};

#[test]
#[ignore = "requires A100 GPU 2 or 3 and RILEY_WIDE_NORM_ORACLE_PATH"]
fn native_wide_norm_matches_torch_for_request_group_reduction_shapes() -> Result<(), Box<dyn Error>>
{
    let root = std::env::var_os("RILEY_WIDE_NORM_ORACLE_PATH")
        .map(PathBuf::from)
        .expect("set pinned oracle directory");
    let cases: serde_json::Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let runtime = CudaRuntime::initialize()?;
    let device = runtime.device(0)?;
    assert_eq!(device.properties().compute_capability(), (8, 0));
    let context = device.create_context()?;
    let mut stream = context.create_stream()?;
    for case in cases["cases"].as_array().ok_or("missing cases")? {
        let id = case["id"].as_str().ok_or("id")?;
        let rows = case["rows"].as_u64().ok_or("rows")?;
        let width = case["width"].as_u64().ok_or("width")?;
        let epsilon = case["epsilon"].as_f64().ok_or("epsilon")? as f32;
        let input = fs::read(root.join(format!("{id}.input.bin")))?;
        let weight = fs::read(root.join(format!("{id}.weight.bin")))?;
        let expected = fs::read(root.join(format!("{id}.output.bin")))?;
        for (suffix, data) in [
            ("input", &input),
            ("weight", &weight),
            ("output", &expected),
        ] {
            let filename = format!("{id}.{suffix}.bin");
            let digest: String = Sha256::digest(data)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            assert_eq!(
                cases["files"][&filename]["sha256"].as_str(),
                Some(digest.as_str()),
                "{filename} digest"
            );
        }
        let slots: Vec<u32> = case["slots"]
            .as_array()
            .ok_or("slots")?
            .iter()
            .map(|x| u32::try_from(x.as_u64().expect("slot")).expect("u32 slot"))
            .collect();
        assert_eq!(input.len(), usize::try_from(rows * width * 2)?);
        assert_eq!(
            expected.len(),
            input.len(),
            "{id}: complete expected output required"
        );
        assert_eq!(weight.len(), usize::try_from(width * 2)?);
        let bytes = u64::try_from(input.len())?;
        let mut staging = context.allocate_pinned_host_buffer(bytes)?;
        let mut x = context.allocate_device_buffer(bytes)?;
        x.upload_from_slice(0, &input, &mut staging, &mut stream)?;
        let mut w = context.allocate_device_buffer(width * 2)?;
        w.upload_from_slice(0, &weight, &mut staging, &mut stream)?;
        let mut y = context.allocate_device_buffer(bytes)?;
        let mut params = RmsNormParams {
            input: CudaBufferSpan::new(&x, CudaDType::BF16, 0, bytes)?,
            weight: CudaBufferSpan::new(&w, CudaDType::BF16, 0, width * 2)?,
            output: CudaBufferSpanMut::new(&mut y, CudaDType::BF16, 0, bytes)?,
            row_count: rows,
            hidden_size: width,
            epsilon,
        };
        hugging_face_qwen_wide_rms_norm_row_groups(&mut params, &slots, &mut stream)?;
        y.copy_to_pinned_async(0, &mut staging, 0, bytes, &mut stream)?
            .synchronize()?;
        let actual = staging.to_vec()?;
        let mismatches = actual
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(mismatches, 0, "{id}: BF16 element mismatches");
        y.close()?;
        w.close()?;
        x.close()?;
        staging.close()?;
        println!("{id}: {} BF16 elements matched", rows * width);
    }
    stream.close()?;
    assert!(context.allocation_stats()?.is_zero());
    context.close()?;
    Ok(())
}
