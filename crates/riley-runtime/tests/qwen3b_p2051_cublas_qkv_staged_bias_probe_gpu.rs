//! Remote-only P2051 direct-cuBLAS plus staged-bias qualifier.
//!
//! P12 defines the primary profile as direct default-cuBLAS raw Q/K/V followed
//! by Riley row_bias_add_in_place.  Its HF module output is intentionally an
//! observational endpoint because the two profiles can have different BF16
//! rounding.  This test never changes a serving selector and records no timing
//! claim.

#![cfg(feature = "cuda-cublas-gemm-probe")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    dead_code,
    unused_imports
)]

mod p2051_cublas_qkv_staged_bias_contract {
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    use riley_cuda::{CublasGemmProbeMetadata, CublasGemmProbeParams, CudaPreparedCublasGemmProbe};

    const P11_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_QKV_MANIFEST";
    const P11_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_QKV_SIDECAR";
    const P12_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_STAGED_BIAS_MANIFEST";
    const P12_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_STAGED_BIAS_SIDECAR";
    const P12_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_STAGED_BIAS_PROBE_OUTPUT";

    const P7_MANIFEST_SHA256: &str =
        "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7";
    const P7_SIDECAR_SHA256: &str =
        "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96";
    const P7_SOURCE_REVISION: &str = "3e4f6f1ead8727cd01e52a30e4673d1c83953606";

    const P11_MANIFEST_SHA256: &str =
        "bff87ad504a406f5b8be98414bc4397f03a15efddb1b6b2cf11d625908bd0255";
    const P11_SIDECAR_SHA256: &str =
        "f92424889ee044678de3b316f97a731577537ecfc1ea8d2b11200165c9a53b8c";
    const P11_SOURCE_REVISION: &str = "ae72a8f8266fb79a20afa5914d81e2a40047b804";

    const P12_MANIFEST_SHA256: &str =
        "1ebefa8c2a1b5bb361bd070518bac31ec7ff0a31e463e3c3e7518e20d0782baf";
    const P12_SIDECAR_SHA256: &str =
        "8908ead9a1c05df26dcec5cfe5b31f3587397348be922a85a3031f80837a1ace";
    const P12_SOURCE_REVISION: &str = "7eba26a84e8b39a2402e5749a85264c0be53e9db";

    const P12_RESULT_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-direct-cublas-staged-bias.v1";
    const P12_RESULT_ARTIFACT_KIND: &str = "qwen2.5-3b-riley-p2051-direct-cublas-staged-bias";
    const P12_RESULT_TRACE_ID: &str = "qwen3b-p2051-direct-cublas-staged-bias-v1";
    const P12_REFERENCE_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);

    const P11_TENSOR_NAMES: [&str; 8] = [
        "p7_input_norm",
        "p7_shadow_raw_q",
        "layer0_q_proj_weight",
        "raw_q/default_cublas_reduced_splitk",
        "layer0_k_proj_weight",
        "raw_k/default_cublas_reduced_splitk",
        "layer0_v_proj_weight",
        "raw_v/default_cublas_reduced_splitk",
    ];
    const P12_TENSOR_NAMES: [&str; 13] = [
        "p11_input_norm",
        "raw_q/default_cublas_reduced_splitk",
        "layer0_q_proj_bias",
        "staged_q/explicit_fp32_bias_add_bf16",
        "p7_actual_q",
        "raw_k/default_cublas_reduced_splitk",
        "layer0_k_proj_bias",
        "staged_k/explicit_fp32_bias_add_bf16",
        "p7_actual_k",
        "raw_v/default_cublas_reduced_splitk",
        "layer0_v_proj_bias",
        "staged_v/explicit_fp32_bias_add_bf16",
        "p7_actual_v",
    ];

    #[derive(Clone, Copy, Debug)]
    struct ProjectionSpec {
        identifier: &'static str,
        module_name: &'static str,
        width: usize,
        p11_weight_name: &'static str,
        p11_raw_name: &'static str,
        p12_bias_name: &'static str,
        p12_staged_name: &'static str,
        p12_actual_name: &'static str,
        checkpoint_bias_key: &'static str,
    }

    const P12_PROJECTIONS: [ProjectionSpec; 3] = [
        ProjectionSpec {
            identifier: "q",
            module_name: "q_proj",
            width: QWEN3B_HIDDEN_SIZE,
            p11_weight_name: "layer0_q_proj_weight",
            p11_raw_name: "raw_q/default_cublas_reduced_splitk",
            p12_bias_name: "layer0_q_proj_bias",
            p12_staged_name: "staged_q/explicit_fp32_bias_add_bf16",
            p12_actual_name: "p7_actual_q",
            checkpoint_bias_key: "model.layers.0.self_attn.q_proj.bias",
        },
        ProjectionSpec {
            identifier: "k",
            module_name: "k_proj",
            width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            p11_weight_name: "layer0_k_proj_weight",
            p11_raw_name: "raw_k/default_cublas_reduced_splitk",
            p12_bias_name: "layer0_k_proj_bias",
            p12_staged_name: "staged_k/explicit_fp32_bias_add_bf16",
            p12_actual_name: "p7_actual_k",
            checkpoint_bias_key: "model.layers.0.self_attn.k_proj.bias",
        },
        ProjectionSpec {
            identifier: "v",
            module_name: "v_proj",
            width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            p11_weight_name: "layer0_v_proj_weight",
            p11_raw_name: "raw_v/default_cublas_reduced_splitk",
            p12_bias_name: "layer0_v_proj_bias",
            p12_staged_name: "staged_v/explicit_fp32_bias_add_bf16",
            p12_actual_name: "p7_actual_v",
            checkpoint_bias_key: "model.layers.0.self_attn.v_proj.bias",
        },
    ];

    #[derive(Debug)]
    struct P11ProjectionArtifact {
        weight_bf16_le: Vec<u8>,
        raw_bf16_le: Vec<u8>,
    }

    #[derive(Debug)]
    struct P11Artifact {
        manifest_path: PathBuf,
        sidecar_path: PathBuf,
        input_bf16_le: Vec<u8>,
        projections: BTreeMap<&'static str, P11ProjectionArtifact>,
    }

    #[derive(Debug)]
    struct P12ProjectionArtifact {
        raw_bf16_le: Vec<u8>,
        bias_bf16_le: Vec<u8>,
        staged_bf16_le: Vec<u8>,
        actual_bf16_le: Vec<u8>,
        staged_vs_actual_metric: Value,
    }

    #[derive(Debug)]
    struct P12Artifact {
        manifest_path: PathBuf,
        sidecar_path: PathBuf,
        projections: BTreeMap<&'static str, P12ProjectionArtifact>,
    }

    #[derive(Debug)]
    struct ProjectionExecution {
        metadata: Value,
        raw_bf16_le: Vec<u8>,
        repeated_raw_bf16_le: Vec<u8>,
        staged_bf16_le: Vec<u8>,
        repeated_staged_bf16_le: Vec<u8>,
        allocation_accounting_unchanged: bool,
    }

    #[derive(Debug)]
    struct Execution {
        device: Value,
        projections: BTreeMap<&'static str, ProjectionExecution>,
        reference_compute_capability_matches: bool,
    }

    fn p11_shape(name: &str) -> TestResult<Vec<u64>> {
        let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
        let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
        let key_value = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)?;
        match name {
            "p7_input_norm" => Ok(vec![sequence, hidden]),
            "p7_shadow_raw_q" | "raw_q/default_cublas_reduced_splitk" => Ok(vec![sequence, hidden]),
            "layer0_q_proj_weight" => Ok(vec![hidden, hidden]),
            "layer0_k_proj_weight" | "layer0_v_proj_weight" => Ok(vec![key_value, hidden]),
            "raw_k/default_cublas_reduced_splitk" | "raw_v/default_cublas_reduced_splitk" => {
                Ok(vec![sequence, key_value])
            }
            _ => Err(format!("unknown P11 tensor {name}").into()),
        }
    }

    fn p12_shape(name: &str) -> TestResult<Vec<u64>> {
        let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
        let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
        let key_value = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)?;
        match name {
            "p11_input_norm" => Ok(vec![sequence, hidden]),
            "raw_q/default_cublas_reduced_splitk"
            | "staged_q/explicit_fp32_bias_add_bf16"
            | "p7_actual_q" => Ok(vec![sequence, hidden]),
            "layer0_q_proj_bias" => Ok(vec![hidden]),
            "raw_k/default_cublas_reduced_splitk"
            | "staged_k/explicit_fp32_bias_add_bf16"
            | "p7_actual_k"
            | "raw_v/default_cublas_reduced_splitk"
            | "staged_v/explicit_fp32_bias_add_bf16"
            | "p7_actual_v" => Ok(vec![sequence, key_value]),
            "layer0_k_proj_bias" | "layer0_v_proj_bias" => Ok(vec![key_value]),
            _ => Err(format!("unknown P12 tensor {name}").into()),
        }
    }

    fn parse_sidecar(
        path: &Path,
        expected_keys: &BTreeSet<String>,
    ) -> TestResult<(Map<String, Value>, Vec<u8>)> {
        let bytes = fs::read(path)?;
        if bytes.len() < 8 {
            return Err("safetensors sidecar is too short".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        if header_len == 0
            || header_len > MAX_SAFETENSORS_HEADER_BYTES
            || 8_usize
                .checked_add(header_len)
                .is_none_or(|end| end > bytes.len())
        {
            return Err("safetensors sidecar header length differs".into());
        }
        let data_start = 8 + header_len;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
        let header = header
            .as_object()
            .ok_or("safetensors sidecar header is not an object")?;
        let actual: BTreeSet<_> = header
            .keys()
            .filter(|key| key.as_str() != "__metadata__")
            .cloned()
            .collect();
        if &actual != expected_keys {
            return Err("safetensors sidecar key set differs".into());
        }
        Ok((header.clone(), bytes[data_start..].to_vec()))
    }

    fn raw_sidecar_tensor(
        header: &Map<String, Value>,
        data: &[u8],
        key: &str,
        shape: &[u64],
        label: &str,
    ) -> TestResult<Vec<u8>> {
        let record = header
            .get(key)
            .ok_or_else(|| format!("{label} sidecar tensor is missing"))?;
        require_exact_fields(record, &["dtype", "shape", "data_offsets"], label)?;
        if record.get("dtype").and_then(Value::as_str) != Some("BF16")
            || record.get("shape")
                != Some(&Value::Array(
                    shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(format!("{label} sidecar metadata differs").into());
        }
        let offsets = record
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{label} offsets are missing"))?;
        if offsets.len() != 2 {
            return Err(format!("{label} offset count differs").into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("sidecar start offset")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("sidecar end offset")?)?;
        let expected = shape_byte_len(shape)?;
        if end < start || end - start != expected || end > data.len() {
            return Err(format!("{label} byte range differs").into());
        }
        let raw = data[start..end].to_vec();
        validate_finite_bf16(&raw, label)?;
        Ok(raw)
    }

    fn manifest_sidecar_tensor(
        header: &Map<String, Value>,
        data: &[u8],
        tensors: &Map<String, Value>,
        name: &str,
        shape: &[u64],
        label: &str,
    ) -> TestResult<Vec<u8>> {
        let reference = tensors
            .get(name)
            .ok_or_else(|| format!("{label} manifest tensor is missing"))?;
        require_exact_fields(
            reference,
            &[
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_bytes",
                "bf16_le_sha256",
            ],
            label,
        )?;
        let key = format!("trace/{name}");
        let expected_bytes = shape_byte_len(shape)?;
        if reference.get("key").and_then(Value::as_str) != Some(key.as_str())
            || reference.get("shape")
                != Some(&Value::Array(
                    shape.iter().copied().map(Value::from).collect(),
                ))
            || reference.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || reference
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || reference.get("bf16_le_bytes").and_then(Value::as_u64)
                != Some(u64::try_from(expected_bytes)?)
        {
            return Err(format!("{label} manifest metadata differs").into());
        }
        let expected_sha = json_sha256(
            reference
                .get("bf16_le_sha256")
                .ok_or("manifest BF16 hash is missing")?,
            label,
        )?;
        let raw = raw_sidecar_tensor(header, data, &key, shape, label)?;
        if sha256_hex(&raw) != expected_sha {
            return Err(format!("{label} sidecar BF16 hash differs").into());
        }
        Ok(raw)
    }

    fn load_p11_artifact() -> TestResult<P11Artifact> {
        let manifest_path = regular_file(
            &required_path(P11_MANIFEST_VARIABLE)?,
            "immutable P11 manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(P11_SIDECAR_VARIABLE)?,
            "immutable P11 sidecar",
        )?;
        if sha256_file(&manifest_path)? != P11_MANIFEST_SHA256
            || sha256_file(&sidecar_path)? != P11_SIDECAR_SHA256
        {
            return Err("P12 qualifier requires pinned P11 artifact hashes".into());
        }
        let expected_keys = P11_TENSOR_NAMES
            .iter()
            .map(|name| format!("trace/{name}"))
            .collect::<BTreeSet<_>>();
        let (header, data) = parse_sidecar(&sidecar_path, &expected_keys)?;
        let input_bf16_le = raw_sidecar_tensor(
            &header,
            &data,
            "trace/p7_input_norm",
            &p11_shape("p7_input_norm")?,
            "P11 input",
        )?;
        let mut projections = BTreeMap::new();
        for projection in P12_PROJECTIONS {
            let weight_bf16_le = raw_sidecar_tensor(
                &header,
                &data,
                &format!("trace/{}", projection.p11_weight_name),
                &p11_shape(projection.p11_weight_name)?,
                "P11 weight",
            )?;
            let raw_bf16_le = raw_sidecar_tensor(
                &header,
                &data,
                &format!("trace/{}", projection.p11_raw_name),
                &p11_shape(projection.p11_raw_name)?,
                "P11 raw output",
            )?;
            projections.insert(
                projection.identifier,
                P11ProjectionArtifact {
                    weight_bf16_le,
                    raw_bf16_le,
                },
            );
        }
        Ok(P11Artifact {
            manifest_path,
            sidecar_path,
            input_bf16_le,
            projections,
        })
    }

    fn validate_p12_binding(
        manifest: &Value,
        field: &str,
        expected_manifest: &str,
        expected_sidecar: &str,
        expected_source_revision: &str,
    ) -> TestResult {
        let binding = manifest
            .get(field)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("P12 {field} binding is missing"))?;
        if binding.get("manifest_sha256").and_then(Value::as_str) != Some(expected_manifest)
            || binding.get("sidecar_sha256").and_then(Value::as_str) != Some(expected_sidecar)
            || binding.get("source_revision").and_then(Value::as_str)
                != Some(expected_source_revision)
        {
            return Err(format!("P12 {field} binding differs").into());
        }
        Ok(())
    }

    fn load_p12_artifact() -> TestResult<P12Artifact> {
        let manifest_path = regular_file(
            &required_path(P12_MANIFEST_VARIABLE)?,
            "immutable P12 manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(P12_SIDECAR_VARIABLE)?,
            "immutable P12 sidecar",
        )?;
        if sha256_file(&manifest_path)? != P12_MANIFEST_SHA256
            || sha256_file(&sidecar_path)? != P12_SIDECAR_SHA256
        {
            return Err("P12 staged-bias artifact hashes differ".into());
        }
        let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        require_exact_fields(
            &manifest,
            &[
                "schema_version",
                "artifact_kind",
                "trace_id",
                "performance_claim_eligible",
                "vllm_comparison_eligible",
                "serving_selector_changed",
                "capture_status",
                "quality_pass",
                "created_at",
                "scope",
                "producer",
                "p7_binding",
                "p11_binding",
                "model",
                "projection_results",
                "comparisons",
                "provenance",
                "sidecar",
                "tensors",
            ],
            "P12 manifest",
        )?;
        if manifest.get("schema_version").and_then(Value::as_str)
            != Some("riley.qwen3b-hf-eager-p2051-qkv-staged-bias-trace.v1")
            || manifest.get("artifact_kind").and_then(Value::as_str)
                != Some("qwen2.5-3b-hf-eager-bf16-p2051-qkv-staged-bias-trace")
            || manifest.get("trace_id").and_then(Value::as_str)
                != Some("qwen3b-p2051-layer0-qkv-staged-bias-v1")
            || manifest.get("quality_pass").and_then(Value::as_bool) != Some(true)
            || manifest
                .get("performance_claim_eligible")
                .and_then(Value::as_bool)
                != Some(false)
            || manifest
                .get("vllm_comparison_eligible")
                .and_then(Value::as_bool)
                != Some(false)
            || manifest
                .get("serving_selector_changed")
                .and_then(Value::as_bool)
                != Some(false)
            || manifest.get("capture_status").and_then(Value::as_str) != Some("captured")
        {
            return Err("P12 manifest identity differs".into());
        }
        validate_p12_binding(
            &manifest,
            "p7_binding",
            P7_MANIFEST_SHA256,
            P7_SIDECAR_SHA256,
            P7_SOURCE_REVISION,
        )?;
        validate_p12_binding(
            &manifest,
            "p11_binding",
            P11_MANIFEST_SHA256,
            P11_SIDECAR_SHA256,
            P11_SOURCE_REVISION,
        )?;
        let source = manifest
            .get("provenance")
            .and_then(|value| value.get("source_repository"))
            .ok_or("P12 source provenance is missing")?;
        if source.get("git_revision").and_then(Value::as_str) != Some(P12_SOURCE_REVISION)
            || source.get("source_dirty").and_then(Value::as_bool) != Some(false)
        {
            return Err("P12 source provenance differs".into());
        }
        let scope = manifest.get("scope").ok_or("P12 scope is missing")?;
        if scope.get("primary_profile").and_then(Value::as_str)
            != Some("explicit_fp32_bias_add_bf16")
            || scope.get("actual_module_output").and_then(Value::as_str)
                != Some("observational-not-primary-profile")
            || scope.get("offline_only").and_then(Value::as_bool) != Some(true)
            || scope.get("serving_path").and_then(Value::as_bool) != Some(false)
        {
            return Err("P12 scope differs".into());
        }
        let expected_keys = P12_TENSOR_NAMES
            .iter()
            .map(|name| format!("trace/{name}"))
            .collect::<BTreeSet<_>>();
        let (header, data) = parse_sidecar(&sidecar_path, &expected_keys)?;
        let tensors = manifest
            .get("tensors")
            .and_then(Value::as_object)
            .ok_or("P12 tensors are missing")?;
        let actual_names: BTreeSet<_> = tensors.keys().map(String::as_str).collect();
        let expected_names: BTreeSet<_> = P12_TENSOR_NAMES.into_iter().collect();
        if actual_names != expected_names {
            return Err("P12 tensor name set differs".into());
        }
        let p11_input = manifest_sidecar_tensor(
            &header,
            &data,
            tensors,
            "p11_input_norm",
            &p12_shape("p11_input_norm")?,
            "P12 input",
        )?;
        let p11_binding = manifest
            .get("p11_binding")
            .and_then(Value::as_object)
            .ok_or("P12 P11 binding is missing")?;
        if p11_binding
            .get("input_bf16_le_sha256")
            .and_then(Value::as_str)
            != Some(sha256_hex(&p11_input).as_str())
        {
            return Err("P12 input/P11 binding differs".into());
        }
        let results = manifest
            .get("projection_results")
            .and_then(Value::as_object)
            .ok_or("P12 projection results are missing")?;
        let comparison_actual = manifest
            .get("comparisons")
            .and_then(|value| value.get("staged_vs_p7_actual_module"))
            .and_then(Value::as_object)
            .ok_or("P12 staged/actual comparison map is missing")?;
        let mut projections = BTreeMap::new();
        for projection in P12_PROJECTIONS {
            let result = results
                .get(projection.identifier)
                .and_then(Value::as_object)
                .ok_or("P12 projection result is missing")?;
            require_exact_fields(
                &Value::Object(result.clone()),
                &[
                    "identifier",
                    "module_name",
                    "raw_tensor",
                    "bias_tensor",
                    "staged_tensor",
                    "actual_tensor",
                    "checkpoint_bias_key",
                    "shape",
                    "raw_bf16_le_sha256",
                    "bias_bf16_le_sha256",
                    "staged_bf16_le_sha256",
                    "actual_bf16_le_sha256",
                    "raw_vs_p11_default_cublas",
                    "staged_repeated_bf16_exact",
                    "staged_vs_p7_actual_module",
                ],
                "P12 projection result",
            )?;
            if result.get("identifier").and_then(Value::as_str) != Some(projection.identifier)
                || result.get("module_name").and_then(Value::as_str) != Some(projection.module_name)
                || result.get("raw_tensor").and_then(Value::as_str) != Some(projection.p11_raw_name)
                || result.get("bias_tensor").and_then(Value::as_str)
                    != Some(projection.p12_bias_name)
                || result.get("staged_tensor").and_then(Value::as_str)
                    != Some(projection.p12_staged_name)
                || result.get("actual_tensor").and_then(Value::as_str)
                    != Some(projection.p12_actual_name)
                || result.get("checkpoint_bias_key").and_then(Value::as_str)
                    != Some(projection.checkpoint_bias_key)
                || result.get("shape")
                    != Some(
                        &json!({"m": CONTEXT_TOKEN_COUNT, "n": projection.width, "k": QWEN3B_HIDDEN_SIZE}),
                    )
                || result
                    .get("staged_repeated_bf16_exact")
                    .and_then(Value::as_bool)
                    != Some(true)
            {
                return Err(
                    format!("P12 {} result contract differs", projection.identifier).into(),
                );
            }
            let raw_bf16_le = manifest_sidecar_tensor(
                &header,
                &data,
                tensors,
                projection.p11_raw_name,
                &p12_shape(projection.p11_raw_name)?,
                "P12 raw",
            )?;
            let bias_bf16_le = manifest_sidecar_tensor(
                &header,
                &data,
                tensors,
                projection.p12_bias_name,
                &p12_shape(projection.p12_bias_name)?,
                "P12 bias",
            )?;
            let staged_bf16_le = manifest_sidecar_tensor(
                &header,
                &data,
                tensors,
                projection.p12_staged_name,
                &p12_shape(projection.p12_staged_name)?,
                "P12 staged",
            )?;
            let actual_bf16_le = manifest_sidecar_tensor(
                &header,
                &data,
                tensors,
                projection.p12_actual_name,
                &p12_shape(projection.p12_actual_name)?,
                "P12 actual",
            )?;
            for (field, bytes) in [
                ("raw_bf16_le_sha256", &raw_bf16_le),
                ("bias_bf16_le_sha256", &bias_bf16_le),
                ("staged_bf16_le_sha256", &staged_bf16_le),
                ("actual_bf16_le_sha256", &actual_bf16_le),
            ] {
                if result.get(field).and_then(Value::as_str) != Some(sha256_hex(bytes).as_str()) {
                    return Err(
                        format!("P12 {} {field} binding differs", projection.identifier).into(),
                    );
                }
            }
            let raw_metric = result
                .get("raw_vs_p11_default_cublas")
                .ok_or("P12 raw/P11 metric is missing")?;
            if raw_metric.get("bf16_exact").and_then(Value::as_bool) != Some(true)
                || raw_metric.get("unequal_elements").and_then(Value::as_u64) != Some(0)
            {
                return Err(format!("P12 {} raw/P11 metric differs", projection.identifier).into());
            }
            let actual_metric = result
                .get("staged_vs_p7_actual_module")
                .ok_or("P12 staged/actual metric is missing")?;
            if comparison_actual.get(projection.identifier) != Some(actual_metric) {
                return Err(format!(
                    "P12 {} actual comparison binding differs",
                    projection.identifier
                )
                .into());
            }
            projections.insert(
                projection.identifier,
                P12ProjectionArtifact {
                    raw_bf16_le,
                    bias_bf16_le,
                    staged_bf16_le,
                    actual_bf16_le,
                    staged_vs_actual_metric: actual_metric.clone(),
                },
            );
        }
        Ok(P12Artifact {
            manifest_path,
            sidecar_path,
            projections,
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
        capability: (u32, u32),
    ) -> TestResult {
        if metadata.backend_id() != CublasGemmProbeMetadata::BACKEND_ID
            || metadata.math_modes() != (0, 0)
            || metadata.pointer_modes() != (0, 0)
            || metadata.atomics_modes() != (0, 0)
            || metadata.runtime_version() <= 0
            || metadata.cublas_version() <= 0
            || metadata.compute_capability() != capability
            || metadata.dimensions() != (config.m(), config.n(), config.k())
        {
            return Err("direct-cuBLAS metadata violates the P12 contract".into());
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

    fn execute_row_bias(
        config: CudaGemmConfig,
        output: &mut CudaDeviceBuffer,
        bias: &CudaDeviceBuffer,
        stream: &mut CudaStream,
    ) -> TestResult {
        let mut params = RowBiasAddInPlaceParams {
            matrix: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            bias: CudaBufferSpan::new(bias, CudaDType::BF16, 0, config.bias_bytes())?,
            row_count: config.m(),
            column_count: config.n(),
        };
        row_bias_add_in_place(&mut params, stream)?;
        Ok(())
    }

    fn run_direct_staged_bias(p11: &P11Artifact, p12: &P12Artifact) -> TestResult<Execution> {
        let runtime = CudaRuntime::initialize()?;
        if runtime.device_count() == 0 {
            return Err("remote GPU runner has no CUDA device".into());
        }
        let device = runtime.device(0)?;
        let properties = device.properties().clone();
        let reference_compute_capability_matches =
            properties.compute_capability() == P12_REFERENCE_COMPUTE_CAPABILITY;
        let device_record = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
            "p12_reference_compute_capability": P12_REFERENCE_COMPUTE_CAPABILITY,
            "p12_reference_compute_capability_matches": reference_compute_capability_matches,
        });
        let context = device.create_context()?;
        let mut stream = context.create_stream()?;
        let result = (|| -> TestResult<Execution> {
            let input_config = CudaGemmConfig::new(
                u64::try_from(CONTEXT_TOKEN_COUNT)?,
                u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                MAX_WORKSPACE_BYTES,
            )?;
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let input_native = bf16_le_to_native(&p11.input_bf16_le)?;
            if input_native.len() != usize::try_from(input_config.input_bytes())? {
                return Err("P11 input byte length differs from P12 config".into());
            }
            let mut input = context.allocate_device_buffer(input_config.input_bytes())?;
            input.upload_from_slice(0, &input_native, &mut staging, &mut stream)?;
            let base_allocations = context.allocation_stats()?;
            let mut projections = BTreeMap::new();
            for projection in P12_PROJECTIONS {
                let p11_projection = p11
                    .projections
                    .get(projection.identifier)
                    .ok_or("P11 projection is missing")?;
                let p12_projection = p12
                    .projections
                    .get(projection.identifier)
                    .ok_or("P12 projection is missing")?;
                let config = CudaGemmConfig::new(
                    u64::try_from(CONTEXT_TOKEN_COUNT)?,
                    u64::try_from(projection.width)?,
                    u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                    MAX_WORKSPACE_BYTES,
                )?;
                let weight_native = bf16_le_to_native(&p11_projection.weight_bf16_le)?;
                let bias_native = bf16_le_to_native(&p12_projection.bias_bf16_le)?;
                if weight_native.len() != usize::try_from(config.weight_bytes())?
                    || bias_native.len() != usize::try_from(config.bias_bytes())?
                {
                    return Err(format!(
                        "P12 {} operand byte length differs",
                        projection.identifier
                    )
                    .into());
                }
                let mut weight = context.allocate_device_buffer(config.weight_bytes())?;
                let mut bias = context.allocate_device_buffer(config.bias_bytes())?;
                let mut output = context.allocate_device_buffer(config.output_bytes())?;
                weight.upload_from_slice(0, &weight_native, &mut staging, &mut stream)?;
                bias.upload_from_slice(0, &bias_native, &mut staging, &mut stream)?;
                let mut plan = context.prepare_cublas_gemm_probe(config)?;
                let metadata = plan.metadata();
                if plan.config() != config {
                    return Err(format!("P12 {} plan config differs", projection.identifier).into());
                }
                validate_probe_metadata(metadata, config, properties.compute_capability())?;
                let allocations_before_execute = context.allocation_stats()?;
                execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
                let mut raw_native = vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(0, &mut raw_native, &mut staging, &mut stream)?;
                execute_row_bias(config, &mut output, &bias, &mut stream)?;
                let mut staged_native = vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(0, &mut staged_native, &mut staging, &mut stream)?;
                execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
                let mut repeated_raw_native = vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(0, &mut repeated_raw_native, &mut staging, &mut stream)?;
                execute_row_bias(config, &mut output, &bias, &mut stream)?;
                let mut repeated_staged_native =
                    vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(
                    0,
                    &mut repeated_staged_native,
                    &mut staging,
                    &mut stream,
                )?;
                let allocation_accounting_unchanged =
                    context.allocation_stats()? == allocations_before_execute;
                let raw_bf16_le = bf16_native_to_le(&raw_native)?;
                let repeated_raw_bf16_le = bf16_native_to_le(&repeated_raw_native)?;
                let staged_bf16_le = bf16_native_to_le(&staged_native)?;
                let repeated_staged_bf16_le = bf16_native_to_le(&repeated_staged_native)?;
                plan.close()?;
                weight.close()?;
                bias.close()?;
                output.close()?;
                if context.allocation_stats()? != base_allocations {
                    return Err(format!(
                        "P12 {} leaked a device allocation after close",
                        projection.identifier
                    )
                    .into());
                }
                projections.insert(
                    projection.identifier,
                    ProjectionExecution {
                        metadata: metadata_json(metadata),
                        raw_bf16_le,
                        repeated_raw_bf16_le,
                        staged_bf16_le,
                        repeated_staged_bf16_le,
                        allocation_accounting_unchanged,
                    },
                );
            }
            input.close()?;
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err("P12 native qualifier left allocations before context close".into());
            }
            Ok(Execution {
                device: device_record,
                projections,
                reference_compute_capability_matches,
            })
        })();
        let cleanup = close_context(stream, context);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(run_error), Err(cleanup_error)) => Err(format!(
                "direct-cuBLAS staged-bias qualifier failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only P2051 direct-cuBLAS staged-bias quality qualifier"]
    fn qwen3b_p2051_direct_cublas_matches_p12_staged_bias_qkv() -> TestResult {
        let output = required_path(P12_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let p11 = load_p11_artifact()?;
        let p12 = load_p12_artifact()?;
        if p12
            .projections
            .get("q")
            .ok_or("P12 Q projection is missing")?
            .raw_bf16_le
            != p11
                .projections
                .get("q")
                .ok_or("P11 Q projection is missing")?
                .raw_bf16_le
        {
            return Err("P12 raw Q does not bind the pinned P11 artifact".into());
        }
        if p12.manifest_path.parent() != p12.sidecar_path.parent() {
            return Err("P12 artifact files must be siblings".into());
        }
        let execution = run_direct_staged_bias(&p11, &p12)?;
        let mut projection_receipts = Map::new();
        let mut raw_exact = true;
        let mut staged_exact = true;
        let mut raw_repeated = true;
        let mut staged_repeated = true;
        let mut allocation_unchanged = true;
        let mut actual_endpoint_exact = true;
        for projection in P12_PROJECTIONS {
            let expected = p12
                .projections
                .get(projection.identifier)
                .ok_or("P12 expected projection is missing")?;
            let p11_expected = p11
                .projections
                .get(projection.identifier)
                .ok_or("P11 expected projection is missing")?;
            let observed = execution
                .projections
                .get(projection.identifier)
                .ok_or("P12 native projection is missing")?;
            let raw_comparison = metrics(&expected.raw_bf16_le, &observed.raw_bf16_le)?;
            let raw_p11_comparison = metrics(&p11_expected.raw_bf16_le, &observed.raw_bf16_le)?;
            let staged_comparison = metrics(&expected.staged_bf16_le, &observed.staged_bf16_le)?;
            let actual_comparison = metrics(&expected.actual_bf16_le, &observed.staged_bf16_le)?;
            if actual_comparison != expected.staged_vs_actual_metric {
                return Err(format!(
                    "P12 {} native staged/actual metric differs from oracle",
                    projection.identifier
                )
                .into());
            }
            let raw_is_exact =
                raw_comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
            let staged_is_exact =
                staged_comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
            let actual_is_exact =
                actual_comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
            raw_exact = raw_exact && raw_is_exact;
            staged_exact = staged_exact && staged_is_exact;
            raw_repeated = raw_repeated && observed.raw_bf16_le == observed.repeated_raw_bf16_le;
            staged_repeated =
                staged_repeated && observed.staged_bf16_le == observed.repeated_staged_bf16_le;
            allocation_unchanged = allocation_unchanged && observed.allocation_accounting_unchanged;
            actual_endpoint_exact = actual_endpoint_exact && actual_is_exact;
            projection_receipts.insert(
                projection.identifier.to_owned(),
                json!({
                    "shape": {"m": CONTEXT_TOKEN_COUNT, "n": projection.width, "k": QWEN3B_HIDDEN_SIZE},
                    "raw_default_cublas_comparison": raw_comparison,
                    "raw_p11_comparison": raw_p11_comparison,
                    "staged_primary_comparison": staged_comparison,
                    "staged_vs_hf_actual_module_comparison": actual_comparison,
                    "repeated_raw_bf16_exact": observed.raw_bf16_le == observed.repeated_raw_bf16_le,
                    "repeated_staged_bf16_exact": observed.staged_bf16_le == observed.repeated_staged_bf16_le,
                    "allocation_accounting_unchanged": observed.allocation_accounting_unchanged,
                    "metadata": observed.metadata,
                }),
            );
        }
        let primary_quality_pass = execution.reference_compute_capability_matches
            && raw_exact
            && staged_exact
            && raw_repeated
            && staged_repeated
            && allocation_unchanged;
        let source_paths = [
            "crates/riley-runtime/tests/qwen3b_p2051_cublas_qkv_staged_bias_probe_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "crates/riley-cuda/src/ffi.rs",
            "kernels/src/cublas_gemm_probe.cu",
            "kernels/src/gemm.cu",
        ];
        let root = repository_root()?;
        let source_hashes = source_paths
            .into_iter()
            .map(|relative| {
                let path = regular_file(&root.join(relative), "P12 native qualifier source")?;
                Ok((relative, sha256_file(&path)?))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let receipt = json!({
            "schema_version": P12_RESULT_SCHEMA_VERSION,
            "artifact_kind": P12_RESULT_ARTIFACT_KIND,
            "created_at_unix_seconds": unix_seconds()?,
            "primary_quality_pass": primary_quality_pass,
            "hf_actual_module_gate_pass": actual_endpoint_exact,
            "projection_boundary_candidate_eligible": false,
            "performance_claim_eligible": false,
            "vllm_comparison_eligible": false,
            "immutable_artifacts": {
                "p7_manifest_sha256": P7_MANIFEST_SHA256,
                "p7_sidecar_sha256": P7_SIDECAR_SHA256,
                "p7_source_revision": P7_SOURCE_REVISION,
                "p11_manifest_path": p11.manifest_path,
                "p11_manifest_sha256": P11_MANIFEST_SHA256,
                "p11_sidecar_path": p11.sidecar_path,
                "p11_sidecar_sha256": P11_SIDECAR_SHA256,
                "p11_source_revision": P11_SOURCE_REVISION,
                "p12_manifest_path": p12.manifest_path,
                "p12_manifest_sha256": P12_MANIFEST_SHA256,
                "p12_sidecar_path": p12.sidecar_path,
                "p12_sidecar_sha256": P12_SIDECAR_SHA256,
                "p12_source_revision": P12_SOURCE_REVISION,
            },
            "contract": {
                "trace_id": P12_RESULT_TRACE_ID,
                "primary_profile": "direct default cublasGemmEx BF16 raw plus riley_cuda::row_bias_add_in_place",
                "actual_module_output": "observational gate; no selector promotion if non-exact",
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "cuda_graph": false,
                "command_batch": false,
            },
            "device": execution.device,
            "projections": projection_receipts,
            "source_hashes": source_hashes,
            "summary": {
                "reference_compute_capability_matches": execution.reference_compute_capability_matches,
                "raw_qkv_exact": raw_exact,
                "staged_qkv_exact": staged_exact,
                "raw_qkv_repeated_bf16_exact": raw_repeated,
                "staged_qkv_repeated_bf16_exact": staged_repeated,
                "allocation_accounting_unchanged": allocation_unchanged,
                "hf_actual_module_qkv_exact": actual_endpoint_exact,
                "projection_boundary_candidate_eligible": false,
                "performance_claim_eligible": false,
                "vllm_comparison_eligible": false,
                "interpretation": "A true primary_quality_pass establishes only the P12 staged numerical contract. A false hf_actual_module_gate_pass blocks selector promotion, full-forward qualification, and serving performance comparison.",
            },
        });
        write_artifact_exclusive(&output, &receipt)?;
        if !primary_quality_pass {
            return Err(
                "direct-cuBLAS plus row-bias path did not meet the P12 staged primary quality gate"
                    .into(),
            );
        }
        println!(
            "QWEN3B_P2051_DIRECT_CUBLAS_STAGED_BIAS trace_id={} primary_quality_pass=true hf_actual_module_gate_pass={} performance_claim_eligible=false",
            P12_RESULT_TRACE_ID, actual_endpoint_exact,
        );
        Ok(())
    }
}
