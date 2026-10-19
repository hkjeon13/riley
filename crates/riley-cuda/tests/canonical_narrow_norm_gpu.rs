//! Compare optimized canonical normalization with a frozen pre-change GPU run.
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaDType, CudaRuntime, ResidualRmsNormParams,
    RmsNormParams, residual_rms_norm, rms_norm,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};
fn sha(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
#[test]
#[ignore = "GPU2 only; pre-change canonical GPU fixture required"]
fn canonical_narrow_norm_preserves_frozen_gpu_bits() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(std::env::var("RILEY_CANONICAL_NORM_ROOT")?);
    let record = std::env::var("RILEY_CANONICAL_NORM_RECORD").as_deref() == Ok("1");
    if record {
        fs::create_dir(&root)?;
    }
    let frozen: Value = if record {
        Value::Null
    } else {
        serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?
    };
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut cases = Vec::new();
    for (width, rows) in [
        (576, 1),
        (576, 2),
        (576, 8),
        (576, 32),
        (576, 128),
        (576, 256),
        (576, 257),
        (512, 1),
        (640, 8),
    ] {
        for epsilon in [1e-5f32, 1e-6f32] {
            let len = width * rows * 2;
            let parent = len + 128;
            let mut h = c.allocate_pinned_host_buffer(parent as u64)?;
            let pattern = |count: usize, salt: usize| -> Vec<u8> {
                (0..count)
                    .flat_map(|i| {
                        let b = if i % 71 == 0 {
                            0u16
                        } else {
                            let magnitude = (((i * 37 + salt * 11) % 1023) + 1) as f32 / 64.;
                            let v = if (i + salt) % 3 == 0 {
                                -magnitude
                            } else {
                                magnitude
                            };
                            let u = v.to_bits();
                            ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
                        };
                        b.to_le_bytes()
                    })
                    .collect()
            };
            let left = pattern(width * rows, 1);
            let right = pattern(width * rows, 2);
            let weight = pattern(width, 3);
            let mut x = c.allocate_device_buffer(len as u64)?;
            x.upload_from_slice(0, &left, &mut h, &mut stream)?;
            let mut z = c.allocate_device_buffer(len as u64)?;
            z.upload_from_slice(0, &right, &mut h, &mut stream)?;
            let mut w = c.allocate_device_buffer((width * 2) as u64)?;
            w.upload_from_slice(0, &weight, &mut h, &mut stream)?;
            let mut y = c.allocate_device_buffer(parent as u64)?;
            let mut r = c.allocate_device_buffer(parent as u64)?;
            for fused in [false, true] {
                let poison = vec![0x5a; parent];
                y.upload_from_slice(0, &poison, &mut h, &mut stream)?;
                r.upload_from_slice(0, &poison, &mut h, &mut stream)?;
                if fused {
                    residual_rms_norm(
                        &mut ResidualRmsNormParams {
                            left: CudaBufferSpan::new(&x, CudaDType::BF16, 0, len as u64)?,
                            right: CudaBufferSpan::new(&z, CudaDType::BF16, 0, len as u64)?,
                            weight: CudaBufferSpan::new(
                                &w,
                                CudaDType::BF16,
                                0,
                                (width * 2) as u64,
                            )?,
                            residual_output: CudaBufferSpanMut::new(
                                &mut r,
                                CudaDType::BF16,
                                0,
                                parent as u64,
                            )?,
                            normalized_output: CudaBufferSpanMut::new(
                                &mut y,
                                CudaDType::BF16,
                                0,
                                parent as u64,
                            )?,
                            row_count: rows as u64,
                            hidden_size: width as u64,
                            epsilon,
                        },
                        &mut stream,
                    )?;
                } else {
                    rms_norm(
                        &mut RmsNormParams {
                            input: CudaBufferSpan::new(&x, CudaDType::BF16, 0, len as u64)?,
                            weight: CudaBufferSpan::new(
                                &w,
                                CudaDType::BF16,
                                0,
                                (width * 2) as u64,
                            )?,
                            output: CudaBufferSpanMut::new(
                                &mut y,
                                CudaDType::BF16,
                                0,
                                parent as u64,
                            )?,
                            row_count: rows as u64,
                            hidden_size: width as u64,
                            epsilon,
                        },
                        &mut stream,
                    )?;
                }
                for (role, buffer) in [("normalized", &mut y), ("residual", &mut r)] {
                    buffer
                        .copy_to_pinned_async(0, &mut h, 0, parent as u64, &mut stream)?
                        .synchronize()?;
                    let mut actual = vec![0; parent];
                    h.read(0, &mut actual)?;
                    assert_eq!(&actual[len..], &poison[len..], "tail overwritten");
                    let name = format!(
                        "w{width}-r{rows}-e{}-f{fused}-{role}.bin",
                        epsilon.to_bits()
                    );
                    if record {
                        fs::write(root.join(&name), &actual)?;
                    } else {
                        let expected = fs::read(root.join(&name))?;
                        assert_eq!(
                            sha(&expected),
                            frozen["files"][&name]["sha256"]
                                .as_str()
                                .ok_or("missing hash")?
                        );
                        assert_eq!(actual, expected, "{name}");
                    }
                    cases.push(json!({"name":name,"sha256":sha(&actual),"bytes":actual.len(),"left_sha256":sha(&left),"right_sha256":sha(&right),"weight_sha256":sha(&weight)}));
                }
            }
            y.close()?;
            r.close()?;
            w.close()?;
            z.close()?;
            x.close()?;
            h.close()?;
        }
    }
    if record {
        let files: serde_json::Map<String, Value> = cases
            .iter()
            .map(|x| (x["name"].as_str().unwrap().to_string(), x.clone()))
            .collect();
        fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(
                &json!({"scope":"frozen canonical GPU baseline; not an independent HF oracle","files":files}),
            )?,
        )?;
    }
    assert!(c.allocation_stats()?.is_zero());
    stream.close()?;
    c.close()?;
    println!("canonical frozen GPU outputs checked {}", cases.len());
    Ok(())
}
