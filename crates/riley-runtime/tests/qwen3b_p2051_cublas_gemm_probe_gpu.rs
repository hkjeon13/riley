//! Remote-only direct-cuBLAS P2051 raw-Q arithmetic qualifier.
//!
//! This consumes the immutable P9 PyTorch arithmetic trace but keeps the
//! direct cuBLAS plan separate from every serving selector, graph, and batch
//! execution path. Its elapsed time is intentionally not a performance metric.

#![cfg(feature = "cuda-cublas-gemm-probe")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    dead_code,
    unused_imports
)]

mod p2051_cublas_probe_contract {
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    use riley_cuda::{CublasGemmProbeMetadata, CublasGemmProbeParams, CudaPreparedCublasGemmProbe};

    const P9_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_BF16_ARITHMETIC_MANIFEST";
    const P9_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_BF16_ARITHMETIC_SIDECAR";
    const P10_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_GEMM_PROBE_OUTPUT";
    const P9_MANIFEST_SHA256: &str =
        "3079d4b1e6e3e654b92431d69cc0378ece76e87f50bb736710f95c487d61c0ed";
    const P9_SIDECAR_SHA256: &str =
        "11fe03272a5cbfcf2aed5ea7adee3232641aa181d1eb92f10736897c15d3ac08";
    const P7_MANIFEST_SHA256: &str =
        "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7";
    const P7_SIDECAR_SHA256: &str =
        "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96";
    const P7_SOURCE_REVISION: &str = "3e4f6f1ead8727cd01e52a30e4673d1c83953606";
    const P7_DEFAULT_RAW_Q_SHA256: &str =
        "9353470bc0d4110218b9c4584ea782257d5a59888db5a3c0479131e5ba46ec8f";
    const P9_INPUT_SHA256: &str =
        "e865d626ed2782bc596735628a730fdaaa93178e41ee8e91d76ba6c92872c1ab";
    const P9_WEIGHT_SHA256: &str =
        "1e74e3883d71871e847369c796437fd0496f7d066584abb13e92483d80592e3f";
    const P10_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-direct-cublas-raw-q.v1";
    const P10_ARTIFACT_KIND: &str = "qwen2.5-3b-riley-p2051-direct-cublas-raw-q";
    const P10_TRACE_ID: &str = "qwen3b-p2051-direct-cublas-raw-q-v1";
    const P10_REFERENCE_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);

    #[derive(Debug)]
    struct P9RawQArtifact {
        manifest_path: PathBuf,
        sidecar_path: PathBuf,
        input_bf16_le: Vec<u8>,
        weight_bf16_le: Vec<u8>,
        p7_default_raw_q_bf16_le: Vec<u8>,
    }

    #[derive(Debug)]
    struct ProbeExecution {
        device: Value,
        metadata: Value,
        output_bf16_le: Vec<u8>,
        repeated_output_bf16_exact: bool,
        allocation_accounting_unchanged: bool,
        reference_compute_capability_matches: bool,
    }

    fn sidecar_tensor(
        header: &Map<String, Value>,
        data: &[u8],
        key: &str,
        expected_shape: &[u64],
        expected_sha256: &str,
    ) -> TestResult<Vec<u8>> {
        let record = header
            .get(key)
            .ok_or_else(|| format!("P9 sidecar does not contain {key}"))?;
        if record.get("dtype").and_then(Value::as_str) != Some("BF16")
            || record.get("shape")
                != Some(&Value::Array(
                    expected_shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(format!("P9 sidecar {key} metadata differs").into());
        }
        let offsets = record
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("P9 sidecar {key} offsets are missing"))?;
        if offsets.len() != 2 {
            return Err(format!("P9 sidecar {key} offset count differs").into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("P9 tensor start offset")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("P9 tensor end offset")?)?;
        let expected_bytes = shape_byte_len(expected_shape)?;
        if end < start || end - start != expected_bytes || end > data.len() {
            return Err(format!("P9 sidecar {key} range differs").into());
        }
        let tensor = data[start..end].to_vec();
        if sha256_hex(&tensor) != expected_sha256 {
            return Err(format!("P9 sidecar {key} SHA-256 differs").into());
        }
        validate_finite_bf16(&tensor, key)?;
        Ok(tensor)
    }

    fn load_p9_raw_q_artifact() -> TestResult<P9RawQArtifact> {
        let manifest_path = regular_file(
            &required_path(P9_MANIFEST_VARIABLE)?,
            "immutable P9 arithmetic manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(P9_SIDECAR_VARIABLE)?,
            "immutable P9 arithmetic sidecar",
        )?;
        if sha256_file(&manifest_path)? != P9_MANIFEST_SHA256
            || sha256_file(&sidecar_path)? != P9_SIDECAR_SHA256
        {
            return Err(
                "P10 direct-cuBLAS qualifier requires the pinned P9 artifact hashes".into(),
            );
        }
        let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let p7 = manifest
            .get("p7_binding")
            .ok_or("P9 manifest lacks the immutable P7 binding")?;
        if manifest.get("quality_pass").and_then(Value::as_bool) != Some(true)
            || manifest
                .get("performance_claim_eligible")
                .and_then(Value::as_bool)
                != Some(false)
            || manifest
                .get("vllm_comparison_eligible")
                .and_then(Value::as_bool)
                != Some(false)
            || p7.get("manifest_sha256").and_then(Value::as_str) != Some(P7_MANIFEST_SHA256)
            || p7.get("sidecar_sha256").and_then(Value::as_str) != Some(P7_SIDECAR_SHA256)
            || p7.get("source_revision").and_then(Value::as_str) != Some(P7_SOURCE_REVISION)
        {
            return Err("P9 manifest identity or P7 binding differs".into());
        }

        let bytes = fs::read(&sidecar_path)?;
        if bytes.len() < 8 {
            return Err("P9 sidecar is too short".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        if header_len == 0
            || header_len > MAX_SAFETENSORS_HEADER_BYTES
            || 8_usize
                .checked_add(header_len)
                .is_none_or(|end| end > bytes.len())
        {
            return Err("P9 sidecar header length differs".into());
        }
        let data_start = 8 + header_len;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
        let header = header
            .as_object()
            .ok_or("P9 sidecar header is not an object")?;
        let input_shape = [
            u64::try_from(CONTEXT_TOKEN_COUNT)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
        ];
        let weight_shape = [
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
        ];
        let input_bf16_le = sidecar_tensor(
            header,
            &bytes[data_start..],
            "trace/p7_input_norm",
            &input_shape,
            P9_INPUT_SHA256,
        )?;
        let weight_bf16_le = sidecar_tensor(
            header,
            &bytes[data_start..],
            "trace/layer0_q_proj_weight",
            &weight_shape,
            P9_WEIGHT_SHA256,
        )?;
        let p7_default_raw_q_bf16_le = sidecar_tensor(
            header,
            &bytes[data_start..],
            "trace/raw_q/p7_default_cublas_reduced_splitk",
            &input_shape,
            P7_DEFAULT_RAW_Q_SHA256,
        )?;
        Ok(P9RawQArtifact {
            manifest_path,
            sidecar_path,
            input_bf16_le,
            weight_bf16_le,
            p7_default_raw_q_bf16_le,
        })
    }

    fn metadata_json(metadata: CublasGemmProbeMetadata) -> Value {
        json!({
            "backend_id": metadata.backend_id(),
            "math_modes": metadata.math_modes(),
            "pointer_modes": metadata.pointer_modes(),
            "atomics_modes": metadata.atomics_modes(),
            "runtime_version": metadata.runtime_version(),
            "cublas_version": metadata.cublas_version(),
            "compute_capability": metadata.compute_capability(),
            "dimensions": metadata.dimensions(),
        })
    }

    fn validate_probe_metadata(
        metadata: CublasGemmProbeMetadata,
        config: CudaGemmConfig,
        compute_capability: (u32, u32),
    ) -> TestResult {
        if metadata.backend_id() != CublasGemmProbeMetadata::BACKEND_ID
            || metadata.math_modes() != (0, 0)
            || metadata.pointer_modes() != (0, 0)
            || metadata.atomics_modes() != (0, 0)
            || metadata.runtime_version() <= 0
            || metadata.cublas_version() <= 0
            || metadata.compute_capability() != compute_capability
            || metadata.dimensions() != (config.m(), config.n(), config.k())
        {
            return Err("direct-cuBLAS probe metadata violates the pinned raw-Q contract".into());
        }
        Ok(())
    }

    fn execute_probe(
        plan: &mut CudaPreparedCublasGemmProbe,
        config: CudaGemmConfig,
        input: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
        output: &mut CudaDeviceBuffer,
        stream: &mut CudaStream,
    ) -> TestResult {
        let mut params = CublasGemmProbeParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
        };
        plan.execute(&mut params, stream)?;
        Ok(())
    }

    fn run_direct_cublas_probe(artifact: &P9RawQArtifact) -> TestResult<ProbeExecution> {
        let config = CudaGemmConfig::new(
            u64::try_from(CONTEXT_TOKEN_COUNT)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            MAX_WORKSPACE_BYTES,
        )?;
        let runtime = CudaRuntime::initialize()?;
        if runtime.device_count() == 0 {
            return Err("remote GPU runner has no CUDA device".into());
        }
        let device = runtime.device(0)?;
        let properties = device.properties().clone();
        let reference_compute_capability_matches =
            properties.compute_capability() == P10_REFERENCE_COMPUTE_CAPABILITY;
        let device_record = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
            "p10_reference_compute_capability": P10_REFERENCE_COMPUTE_CAPABILITY,
            "p10_reference_compute_capability_matches": reference_compute_capability_matches,
        });
        let context = device.create_context()?;
        let mut stream = context.create_stream()?;
        let result = (|| -> TestResult<ProbeExecution> {
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let input_native = bf16_le_to_native(&artifact.input_bf16_le)?;
            let weight_native = bf16_le_to_native(&artifact.weight_bf16_le)?;
            let mut input = context.allocate_device_buffer(config.input_bytes())?;
            input.upload_from_slice(0, &input_native, &mut staging, &mut stream)?;
            let mut weight = context.allocate_device_buffer(config.weight_bytes())?;
            weight.upload_from_slice(0, &weight_native, &mut staging, &mut stream)?;
            let mut output = context.allocate_device_buffer(config.output_bytes())?;
            let mut plan = context.prepare_cublas_gemm_probe(config)?;
            let metadata = plan.metadata();
            if plan.config() != config {
                return Err("direct-cuBLAS plan config differs after prepare".into());
            }
            validate_probe_metadata(metadata, config, properties.compute_capability())?;
            let allocations_before = context.allocation_stats()?;
            execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
            let mut first_native = vec![0_u8; usize::try_from(config.output_bytes())?];
            output.download_to_slice(0, &mut first_native, &mut staging, &mut stream)?;
            execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
            let mut repeated_native = vec![0_u8; usize::try_from(config.output_bytes())?];
            output.download_to_slice(0, &mut repeated_native, &mut staging, &mut stream)?;
            let repeated_output_bf16_exact = first_native == repeated_native;
            let allocation_accounting_unchanged = context.allocation_stats()? == allocations_before;
            let output_bf16_le = bf16_native_to_le(&first_native)?;
            plan.close()?;
            input.close()?;
            weight.close()?;
            output.close()?;
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err(
                    "direct-cuBLAS probe left native allocations before context close".into(),
                );
            }
            Ok(ProbeExecution {
                device: device_record,
                metadata: metadata_json(metadata),
                output_bf16_le,
                repeated_output_bf16_exact,
                allocation_accounting_unchanged,
                reference_compute_capability_matches,
            })
        })();
        let cleanup = close_context(stream, context);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(run_error), Err(cleanup_error)) => Err(format!(
                "direct-cuBLAS raw-Q qualifier failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only P2051 direct-cuBLAS raw-Q arithmetic qualifier"]
    fn qwen3b_p2051_direct_cublas_default_matches_p7_raw_q() -> TestResult {
        let output = required_path(P10_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let artifact = load_p9_raw_q_artifact()?;
        let execution = run_direct_cublas_probe(&artifact)?;
        let comparison = metrics(
            &artifact.p7_default_raw_q_bf16_le,
            &execution.output_bf16_le,
        )?;
        let exact = comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
        let quality_pass = execution.reference_compute_capability_matches
            && execution.repeated_output_bf16_exact
            && execution.allocation_accounting_unchanged
            && exact;
        let root = repository_root()?;
        let source_paths = [
            "crates/riley-runtime/tests/qwen3b_p2051_cublas_gemm_probe_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "crates/riley-cuda/src/ffi.rs",
            "kernels/src/cublas_gemm_probe.cu",
        ];
        let source_hashes = source_paths
            .into_iter()
            .map(|source| {
                let path = regular_file(&root.join(source), "direct-cuBLAS qualifier source")?;
                Ok((source, sha256_file(&path)?))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let receipt = json!({
            "schema_version": P10_SCHEMA_VERSION,
            "artifact_kind": P10_ARTIFACT_KIND,
            "created_at_unix_seconds": unix_seconds()?,
            "quality_pass": quality_pass,
            "performance_claim_eligible": false,
            "vllm_comparison_eligible": false,
            "immutable_p9_artifact": {
                "manifest_path": artifact.manifest_path,
                "manifest_sha256": P9_MANIFEST_SHA256,
                "sidecar_path": artifact.sidecar_path,
                "sidecar_sha256": P9_SIDECAR_SHA256,
                "p7_manifest_sha256": P7_MANIFEST_SHA256,
                "p7_sidecar_sha256": P7_SIDECAR_SHA256,
                "p7_source_revision": P7_SOURCE_REVISION,
            },
            "contract": {
                "trace_id": P10_TRACE_ID,
                "endpoint": "layer0.q_proj.raw_no_bias",
                "shape": {"m": CONTEXT_TOKEN_COUNT, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE},
                "operator": "direct cublasGemmEx OP_T(Wc), OP_N(Xc), BF16 I/O, FP32 compute, CUBLAS_DEFAULT_MATH",
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "cuda_graph": false,
                "command_batch": false,
            },
            "device": execution.device,
            "direct_cublas": {
                "metadata": execution.metadata,
                "output_bf16_le_sha256": sha256_hex(&execution.output_bf16_le),
                "repeated_output_bf16_exact": execution.repeated_output_bf16_exact,
                "allocation_accounting_unchanged": execution.allocation_accounting_unchanged,
                "p7_default_raw_q_comparison": comparison,
            },
            "source_hashes": source_hashes,
            "summary": {
                "reference_compute_capability_matches": execution.reference_compute_capability_matches,
                "direct_cublas_default_matches_p7_raw_q": exact,
                "interpretation": "A true result establishes only the pinned raw-Q arithmetic correspondence. It does not qualify Q/K/V, bias, full-forward correctness, serving behavior, or performance.",
                "serving_selector_changed": false,
                "performance_claim_eligible": false,
                "vllm_comparison_eligible": false,
            },
        });
        write_artifact_exclusive(&output, &receipt)?;
        if !quality_pass {
            return Err(
                "direct-cuBLAS raw-Q candidate did not meet its pinned quality gate".into(),
            );
        }
        println!(
            "QWEN3B_P2051_DIRECT_CUBLAS trace_id={} quality_pass=true performance_claim_eligible=false",
            P10_TRACE_ID,
        );
        Ok(())
    }
}
