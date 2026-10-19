//! Independent real-activation HF component oracle; preserves every mismatch.
#![cfg(feature = "cuda-cublas-gemm-probe")]
use riley_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaGemmConfig,
    CudaPinnedHostBuffer, CudaRuntime, CudaStream, RmsNormParams, SiluParams,
    hugging_face_qwen_wide_rms_norm_row_groups, hugging_face_smollm2_rms_norm_row_groups, silu,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::PathBuf};
type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
fn sha(v: &[u8]) -> String {
    Sha256::digest(v)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn number(v: &Value, k: &str) -> Result<u64> {
    v[k].as_u64().ok_or_else(|| format!("missing {k}").into())
}
fn upload(
    c: &CudaContext,
    s: &mut CudaStream,
    h: &mut CudaPinnedHostBuffer,
    v: &[u8],
) -> Result<CudaDeviceBuffer> {
    let mut b = c.allocate_device_buffer(v.len() as u64)?;
    b.upload_from_slice(0, v, h, s)?;
    Ok(b)
}
#[test]
#[ignore = "A100 GPU2/3; pinned independent HF component fixture required"]
fn real_hf_components_match_bf16_reference() -> Result {
    let root = PathBuf::from(std::env::var("RILEY_COMPONENT_ORACLE_ROOT")?);
    let output = PathBuf::from(std::env::var("RILEY_COMPONENT_ORACLE_OUTPUT")?);
    fs::create_dir(&output)?;
    let fixture_bytes = fs::read(root.join("fixture.json"))?;
    let manifest: Value = serde_json::from_slice(&fixture_bytes)?;
    let read = |case: &Value, key: &str| -> Result<Vec<u8>> {
        let name = case[key].as_str().ok_or("file missing")?;
        if name.is_empty()
            || name.contains("..")
            || !name
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || b"-_.".contains(&v))
        {
            return Err("unsafe fixture leaf".into());
        }
        let bytes = fs::read(root.join(name))?;
        assert_eq!(
            Some(bytes.len() as u64),
            manifest["files"][name]["bytes"].as_u64()
        );
        assert_eq!(
            sha(&bytes),
            manifest["files"][name]["sha256"]
                .as_str()
                .ok_or("SHA missing")?
        );
        Ok(bytes)
    };
    let rt = CudaRuntime::initialize()?;
    assert_eq!(rt.device_count(), 1);
    let c = rt.device(0)?.create_context()?;
    let mut stream = c.create_stream()?;
    let mut reports = Vec::new();
    for case in manifest["cases"].as_array().ok_or("cases missing")? {
        let mut host = c.allocate_pinned_host_buffer(16 * 1024 * 1024)?;
        let expected = read(case, "output")?;
        let padded_norm = case["kind"] == "norm-padded";
        let extra_bytes = if padded_norm {
            7 * number(case, "width")? * 2
        } else {
            0
        };
        let mut result = c.allocate_device_buffer(expected.len() as u64 + extra_bytes)?;
        if padded_norm {
            let sentinel = [0x10, 0x41].repeat(result.byte_len() as usize / 2);
            result.upload_from_slice(0, &sentinel, &mut host, &mut stream)?;
        }
        let mut resources = Vec::new();
        match case["kind"].as_str().ok_or("kind missing")? {
            "norm" | "norm-padded" => {
                let mut bytes = read(case, "input")?;
                bytes.resize(bytes.len() + extra_bytes as usize, 0);
                let input = upload(&c, &mut stream, &mut host, &bytes)?;
                let weight = upload(&c, &mut stream, &mut host, &read(case, "weight")?)?;
                let rows = number(case, "rows")?;
                let width = number(case, "width")?;
                assert_eq!(input.byte_len(), rows * width * 2 + extra_bytes);
                assert_eq!(weight.byte_len(), width * 2);
                assert_eq!(result.byte_len(), input.byte_len());
                let mut params = RmsNormParams {
                    input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                    weight: CudaBufferSpan::new(&weight, CudaDType::BF16, 0, weight.byte_len())?,
                    output: CudaBufferSpanMut::new(
                        &mut result,
                        CudaDType::BF16,
                        0,
                        expected.len() as u64 + extra_bytes,
                    )?,
                    row_count: rows + if padded_norm { 7 } else { 0 },
                    hidden_size: width,
                    epsilon: case["epsilon"].as_f64().ok_or("epsilon missing")? as f32,
                };
                if width == 576 {
                    hugging_face_smollm2_rms_norm_row_groups(
                        &mut params,
                        &vec![0; rows as usize],
                        &mut stream,
                    )?;
                } else {
                    hugging_face_qwen_wide_rms_norm_row_groups(
                        &mut params,
                        &vec![0; rows as usize],
                        &mut stream,
                    )?;
                }
                resources.extend([input, weight]);
            }
            "silu" => {
                let input = upload(&c, &mut stream, &mut host, &read(case, "input")?)?;
                let count = number(case, "elements")?;
                assert_eq!(input.byte_len(), count * 2);
                assert_eq!(result.byte_len(), input.byte_len());
                silu(
                    &mut SiluParams {
                        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                        output: CudaBufferSpanMut::new(
                            &mut result,
                            CudaDType::BF16,
                            0,
                            expected.len() as u64,
                        )?,
                        element_count: count,
                    },
                    &mut stream,
                )?;
                resources.push(input);
            }
            "fused-silu-multiply" => {
                let input = upload(&c, &mut stream, &mut host, &read(case, "input")?)?;
                let up = upload(&c, &mut stream, &mut host, &read(case, "up")?)?;
                let count = number(case, "elements")?;
                assert_eq!(input.byte_len(), count * 2);
                assert_eq!(up.byte_len(), input.byte_len());
                assert_eq!(result.byte_len(), input.byte_len());
                riley_cuda::silu_gated_multiply(
                    &mut riley_cuda::GatedMultiplyParams {
                        activated_gate: CudaBufferSpan::new(
                            &input,
                            CudaDType::BF16,
                            0,
                            input.byte_len(),
                        )?,
                        up: CudaBufferSpan::new(&up, CudaDType::BF16, 0, up.byte_len())?,
                        output: CudaBufferSpanMut::new(
                            &mut result,
                            CudaDType::BF16,
                            0,
                            expected.len() as u64,
                        )?,
                        element_count: count,
                    },
                    &mut stream,
                )?;
                resources.extend([input, up]);
            }
            "rope" | "rope-generated" => {
                let (rows, heads, depth) = (
                    number(case, "rows")?,
                    number(case, "heads")?,
                    number(case, "depth")?,
                );
                let input = upload(&c, &mut stream, &mut host, &read(case, "input")?)?;
                let table = |bytes: Vec<u8>| -> Vec<u8> {
                    bytes
                        .chunks_exact(2)
                        .flat_map(|v| {
                            f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16)
                                .to_ne_bytes()
                        })
                        .collect()
                };
                let (cos, sin) = if case["kind"] == "rope-generated" {
                    let theta = case["theta"].as_f64().ok_or("theta missing")? as f32;
                    let offset = number(case, "position_offset")?;
                    let mut angles = Vec::new();
                    let mut cos_bytes = Vec::new();
                    let mut sin_bytes = Vec::new();
                    for row in 0..rows {
                        for pair in 0..depth / 2 {
                            let inv = 1.0_f32 / theta.powf((2 * pair) as f32 / depth as f32);
                            let angle = (offset + row) as f32 * inv;
                            angles.extend_from_slice(&angle.to_ne_bytes());
                            let (s, c) = angle.sin_cos();
                            cos_bytes.extend_from_slice(&c.to_ne_bytes());
                            sin_bytes.extend_from_slice(&s.to_ne_bytes());
                        }
                    }
                    if case["cuda_tables"] == true {
                        let mut cos = upload(&c, &mut stream, &mut host, &angles)?;
                        let mut sin = c.allocate_device_buffer(angles.len() as u64)?;
                        riley_cuda::rope_table(
                            &mut riley_cuda::RopeTableParams {
                                angles_cos: CudaBufferSpanMut::new(
                                    &mut cos,
                                    CudaDType::F32,
                                    0,
                                    angles.len() as u64,
                                )?,
                                sin: CudaBufferSpanMut::new(
                                    &mut sin,
                                    CudaDType::F32,
                                    0,
                                    angles.len() as u64,
                                )?,
                                element_count: angles.len() as u64 / 4,
                            },
                            &mut stream,
                        )?;
                        (cos, sin)
                    } else {
                        (
                            upload(&c, &mut stream, &mut host, &cos_bytes)?,
                            upload(&c, &mut stream, &mut host, &sin_bytes)?,
                        )
                    }
                } else {
                    (
                        upload(&c, &mut stream, &mut host, &table(read(case, "cos")?))?,
                        upload(&c, &mut stream, &mut host, &table(read(case, "sin")?))?,
                    )
                };
                assert_eq!(input.byte_len(), rows * heads * depth * 2);
                assert_eq!(result.byte_len(), input.byte_len());
                assert_eq!(cos.byte_len(), rows * depth / 2 * 4);
                assert_eq!(sin.byte_len(), cos.byte_len());
                riley_cuda::rope(
                    &mut riley_cuda::RopeParams {
                        input: CudaBufferSpan::new(&input, CudaDType::BF16, 0, input.byte_len())?,
                        cos: CudaBufferSpan::new(&cos, CudaDType::F32, 0, cos.byte_len())?,
                        sin: CudaBufferSpan::new(&sin, CudaDType::F32, 0, sin.byte_len())?,
                        output: CudaBufferSpanMut::new(
                            &mut result,
                            CudaDType::BF16,
                            0,
                            expected.len() as u64,
                        )?,
                        token_count: rows,
                        head_count: heads,
                        head_size: depth,
                        rotary_dimension: depth,
                        table_position_count: rows,
                        position_offset: 0,
                    },
                    &mut stream,
                )?;
                resources.extend([input, cos, sin]);
            }
            "linear" => {
                let (m, n, k) = (number(case, "m")?, number(case, "n")?, number(case, "k")?);
                let input = upload(&c, &mut stream, &mut host, &read(case, "input")?)?;
                let weight = upload(&c, &mut stream, &mut host, &read(case, "weight")?)?;
                assert_eq!(input.byte_len(), m * k * 2);
                assert_eq!(weight.byte_len(), n * k * 2);
                assert_eq!(result.byte_len(), m * n * 2);
                if case["bias"].is_string() {
                    let bias = upload(&c, &mut stream, &mut host, &read(case, "bias")?)?;
                    assert_eq!(bias.byte_len(), n * 2);
                    let cfg = CudaGemmConfig::new(m, n, k, 1 << 20)?.with_reduction_policy(
                        riley_cuda::CudaGemmReductionPolicy::HfCompatibleBiasSplitKV1,
                    );
                    let mut plan = c.prepare_hf_compatible_bias_epilogue_probe(
                        cfg,
                        CudaBufferSpan::new(&bias, CudaDType::BF16, 0, bias.byte_len())?,
                    )?;
                    println!("BIASALGO {} {:?}", case["id"], plan.algorithm_metadata());
                    let need = plan.algorithm_metadata().workspace_bytes();
                    let mut scratch = if need == 0 {
                        None
                    } else {
                        Some(c.allocate_device_buffer(need)?)
                    };
                    let workspace = match scratch.as_mut() {
                        None => None,
                        Some(b) => Some(CudaBufferSpanMut::new(b, CudaDType::U8, 0, need)?),
                    };
                    plan.execute(
                        &mut riley_cuda::BiasGemmParams {
                            input: CudaBufferSpan::new(
                                &input,
                                CudaDType::BF16,
                                0,
                                input.byte_len(),
                            )?,
                            weight: CudaBufferSpan::new(
                                &weight,
                                CudaDType::BF16,
                                0,
                                weight.byte_len(),
                            )?,
                            bias: CudaBufferSpan::new(&bias, CudaDType::BF16, 0, bias.byte_len())?,
                            output: CudaBufferSpanMut::new(
                                &mut result,
                                CudaDType::BF16,
                                0,
                                expected.len() as u64,
                            )?,
                            workspace,
                        },
                        &mut stream,
                    )?;
                    // First anchor the synchronous result, then rerun the same
                    // prepared algorithm under iteration-style completion.
                    // The independent HF fixture checks BF16 bytes, including
                    // real Qwen bias split-K arithmetic and its workspace.
                    let mut synchronous = vec![0; expected.len()];
                    result.download_to_slice(0, &mut synchronous, &mut host, &mut stream)?;
                    assert_eq!(
                        synchronous, expected,
                        "synchronous bias GEMM: {}",
                        case["id"]
                    );
                    let mut batch = stream.begin_command_batch()?;
                    {
                        let mut commands = batch.commands();
                        let workspace = match scratch.as_mut() {
                            None => None,
                            Some(b) => Some(CudaBufferSpanMut::new(b, CudaDType::U8, 0, need)?),
                        };
                        plan.execute(
                            &mut riley_cuda::BiasGemmParams {
                                input: CudaBufferSpan::new(
                                    &input,
                                    CudaDType::BF16,
                                    0,
                                    input.byte_len(),
                                )?,
                                weight: CudaBufferSpan::new(
                                    &weight,
                                    CudaDType::BF16,
                                    0,
                                    weight.byte_len(),
                                )?,
                                bias: CudaBufferSpan::new(
                                    &bias,
                                    CudaDType::BF16,
                                    0,
                                    bias.byte_len(),
                                )?,
                                output: CudaBufferSpanMut::new(
                                    &mut result,
                                    CudaDType::BF16,
                                    0,
                                    expected.len() as u64,
                                )?,
                                workspace,
                            },
                            &mut commands,
                        )?;
                    }
                    batch.finish()?;
                    plan.close()?;
                    if let Some(b) = scratch {
                        b.close()?;
                    }
                    resources.push(bias);
                } else {
                    let mut plan =
                        c.prepare_owned_hf_gemm(CudaGemmConfig::new(m, n, k, 16 << 20)?)?;
                    plan.execute(
                        &mut riley_cuda::CublasGemmProbeParams {
                            input: CudaBufferSpan::new(
                                &input,
                                CudaDType::BF16,
                                0,
                                input.byte_len(),
                            )?,
                            weight: CudaBufferSpan::new(
                                &weight,
                                CudaDType::BF16,
                                0,
                                weight.byte_len(),
                            )?,
                            output: CudaBufferSpanMut::new(
                                &mut result,
                                CudaDType::BF16,
                                0,
                                expected.len() as u64,
                            )?,
                        },
                        &mut stream,
                    )?;
                    plan.close()?;
                }
                resources.extend([input, weight]);
            }
            "attention" => {
                let (m, t, h, k, d) = (
                    number(case, "rows")?,
                    number(case, "tokens")?,
                    number(case, "qh")?,
                    number(case, "kvh")?,
                    number(case, "depth")?,
                );
                let q = read(case, "query")?;
                let keys = read(case, "key")?;
                let values = read(case, "value")?;
                assert_eq!(q.len() as u64, m * h * d * 2);
                assert_eq!(keys.len() as u64, t * k * d * 2);
                assert_eq!(values.len(), keys.len());
                assert_eq!(expected.len() as u64, m * h * d * 2);
                let pages = t.div_ceil(16);
                let physical = pages + 3;
                let ids: Vec<u32> = (0..pages)
                    .map(|p| u32::try_from(physical - 1 - p))
                    .collect::<std::result::Result<_, _>>()?;
                let pool = physical * k * 16 * d * 2;
                let mut kp = vec![0xa5; pool as usize];
                let mut vp = kp.clone();
                for token in 0..t {
                    for head in 0..k {
                        for depth in 0..d {
                            let source = ((head * t + token) * d + depth) as usize * 2;
                            let target = (((u64::from(ids[(token / 16) as usize]) * k + head) * 16
                                + token % 16)
                                * d
                                + depth) as usize
                                * 2;
                            kp[target..target + 2].copy_from_slice(&keys[source..source + 2]);
                            vp[target..target + 2].copy_from_slice(&values[source..source + 2]);
                        }
                    }
                }
                let query = upload(&c, &mut stream, &mut host, &q)?;
                let key = upload(&c, &mut stream, &mut host, &kp)?;
                let value = upload(&c, &mut stream, &mut host, &vp)?;
                let table = upload(
                    &c,
                    &mut stream,
                    &mut host,
                    &ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
                )?;
                let mut buffers = Vec::new();
                for bytes in [
                    m * h * d * 2,
                    t * h * d * 2,
                    t * h * d * 2,
                    h * m * t * 2,
                    h * m.min(128) * t * 4,
                    h * m.min(128) * t * 4,
                    m * h * d * 2,
                    16 * 1024 * 1024,
                ] {
                    buffers.push(c.allocate_device_buffer(bytes)?);
                }
                let mut scratch: [CudaDeviceBuffer; 8] = buffers
                    .try_into()
                    .map_err(|_| "eight scratch buffers required")?;
                let mut plan =
                    c.prepare_owned_hf_gemm(CudaGemmConfig::new(m, t, d, 16 * 1024 * 1024)?)?;
                plan.execute_hf_attention(
                    CudaBufferSpan::new(&query, CudaDType::BF16, 0, query.byte_len())?,
                    CudaBufferSpan::new(&key, CudaDType::BF16, 0, key.byte_len())?,
                    CudaBufferSpan::new(&value, CudaDType::BF16, 0, value.byte_len())?,
                    CudaBufferSpan::new(&table, CudaDType::U32, 0, table.byte_len())?,
                    CudaBufferSpanMut::new(&mut result, CudaDType::BF16, 0, expected.len() as u64)?,
                    &mut scratch,
                    m,
                    t,
                    physical,
                    h,
                    k,
                    d,
                    &mut stream,
                )?;
                plan.close()?;
                for b in scratch {
                    b.close()?;
                }
                resources.extend([query, key, value, table]);
            }
            other => return Err(format!("unsupported component {other}").into()),
        }
        let mut actual = vec![0; expected.len()];
        result.download_to_slice(0, &mut actual, &mut host, &mut stream)?;
        if padded_norm {
            let mut tail = vec![0; extra_bytes as usize];
            result.download_to_slice(expected.len() as u64, &mut tail, &mut host, &mut stream)?;
            assert_eq!(
                tail,
                [0x10, 0x41].repeat(extra_bytes as usize / 2),
                "inactive rows were overwritten"
            );
        }
        let mismatches = actual
            .chunks_exact(2)
            .zip(expected.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        let mut max_abs = 0.0f64;
        let mut abs = 0.0f64;
        for (a, b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
            let x = f32::from_bits(u32::from(u16::from_le_bytes([a[0], a[1]])) << 16);
            let y = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
            assert!(x.is_finite() && y.is_finite());
            let diff = f64::from((x - y).abs());
            abs += diff;
            max_abs = max_abs.max(diff);
        }
        let id = case["id"].as_str().ok_or("id missing")?;
        assert!(id.bytes().all(|v| v.is_ascii_alphanumeric() || v == b'-'));
        fs::write(output.join(format!("{id}.native.bf16")), &actual)?;
        let report = json!({"id":id,"kind":case["kind"],"elements":actual.len()/2,"mismatches":mismatches,"max_abs":max_abs,"mean_abs":abs/(actual.len()/2) as f64,"native_sha256":sha(&actual),"expected_sha256":sha(&expected)});
        println!("COMPONENT {report}");
        reports.push(report);
        result.close()?;
        for b in resources {
            b.close()?;
        }
        host.close()?;
        assert!(c.allocation_stats()?.is_zero());
    }
    stream.close()?;
    c.close()?;
    let passed = reports.iter().all(|v| v["mismatches"] == 0);
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(
            &json!({"fixture_sha256":sha(&fixture_bytes),"passed":passed,"cases":reports}),
        )?,
    )?;
    assert!(
        passed,
        "HF component byte mismatch; complete diagnostic output retained"
    );
    Ok(())
}
