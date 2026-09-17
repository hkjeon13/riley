//! Remote-only direct-cuBLAS P2051 raw-Q/K/V arithmetic qualifier.
//!
//! The P11 offline artifact fixes the PyTorch default-Cublas Q/K/V arithmetic
//! result. This test consumes that immutable artifact through the native/Rust
//! boundary only. It does not alter a selector, graph, command batch, or any
//! server runtime path, and it deliberately records no elapsed-time metric.

#![cfg(feature = "cuda-cublas-gemm-probe")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::too_many_lines,
    dead_code,
    unused_imports
)]

mod p2051_cublas_qkv_probe_contract {
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    use riley_cuda::{CublasGemmProbeMetadata, CublasGemmProbeParams, CudaPreparedCublasGemmProbe};

    const P11_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_QKV_MANIFEST";
    const P11_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_QKV_SIDECAR";
    const P11_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_CUBLAS_QKV_PROBE_OUTPUT";
    const P11_MANIFEST_SHA256: &str =
        "bff87ad504a406f5b8be98414bc4397f03a15efddb1b6b2cf11d625908bd0255";
    const P11_SIDECAR_SHA256: &str =
        "f92424889ee044678de3b316f97a731577537ecfc1ea8d2b11200165c9a53b8c";
    const P7_MANIFEST_SHA256: &str =
        "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7";
    const P7_SIDECAR_SHA256: &str =
        "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96";
    const P7_SOURCE_REVISION: &str = "3e4f6f1ead8727cd01e52a30e4673d1c83953606";
    const P11_SOURCE_REVISION: &str = "ae72a8f8266fb79a20afa5914d81e2a40047b804";
    const P11_Q_OUTPUT_SHA256: &str =
        "9353470bc0d4110218b9c4584ea782257d5a59888db5a3c0479131e5ba46ec8f";
    const P11_K_OUTPUT_SHA256: &str =
        "fd561bb4602e3cdcf8bcc0a960996045588ee48905af0bb7211a93ddb071edc8";
    const P11_V_OUTPUT_SHA256: &str =
        "5ecc4af8cc0b0f02e3b4e4ffa4ea828cce26c7bfa8137d88fef3ed798de16439";
    const P11_RESULT_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-direct-cublas-raw-qkv.v1";
    const P11_RESULT_ARTIFACT_KIND: &str = "qwen2.5-3b-riley-p2051-direct-cublas-raw-qkv";
    const P11_RESULT_TRACE_ID: &str = "qwen3b-p2051-direct-cublas-raw-qkv-v1";
    const P11_REFERENCE_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);
    const P11_QKV_TENSOR_NAMES: [&str; 8] = [
        "p7_input_norm",
        "p7_shadow_raw_q",
        "layer0_q_proj_weight",
        "raw_q/default_cublas_reduced_splitk",
        "layer0_k_proj_weight",
        "raw_k/default_cublas_reduced_splitk",
        "layer0_v_proj_weight",
        "raw_v/default_cublas_reduced_splitk",
    ];

    #[derive(Clone, Copy, Debug)]
    struct ProjectionSpec {
        identifier: &'static str,
        module_name: &'static str,
        width: usize,
        weight_name: &'static str,
        output_name: &'static str,
        checkpoint_weight_key: &'static str,
        expected_output_sha256: &'static str,
    }

    const PROJECTIONS: [ProjectionSpec; 3] = [
        ProjectionSpec {
            identifier: "q",
            module_name: "q_proj",
            width: QWEN3B_HIDDEN_SIZE,
            weight_name: "layer0_q_proj_weight",
            output_name: "raw_q/default_cublas_reduced_splitk",
            checkpoint_weight_key: "model.layers.0.self_attn.q_proj.weight",
            expected_output_sha256: P11_Q_OUTPUT_SHA256,
        },
        ProjectionSpec {
            identifier: "k",
            module_name: "k_proj",
            width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            weight_name: "layer0_k_proj_weight",
            output_name: "raw_k/default_cublas_reduced_splitk",
            checkpoint_weight_key: "model.layers.0.self_attn.k_proj.weight",
            expected_output_sha256: P11_K_OUTPUT_SHA256,
        },
        ProjectionSpec {
            identifier: "v",
            module_name: "v_proj",
            width: QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION,
            weight_name: "layer0_v_proj_weight",
            output_name: "raw_v/default_cublas_reduced_splitk",
            checkpoint_weight_key: "model.layers.0.self_attn.v_proj.weight",
            expected_output_sha256: P11_V_OUTPUT_SHA256,
        },
    ];

    #[derive(Debug)]
    struct P11ProjectionArtifact {
        weight_bf16_le: Vec<u8>,
        expected_output_bf16_le: Vec<u8>,
    }

    #[derive(Debug)]
    struct P11QkvArtifact {
        manifest_path: PathBuf,
        sidecar_path: PathBuf,
        input_bf16_le: Vec<u8>,
        p7_shadow_raw_q_bf16_le: Vec<u8>,
        projections: BTreeMap<&'static str, P11ProjectionArtifact>,
    }

    #[derive(Debug)]
    struct ProjectionExecution {
        metadata: Value,
        output_bf16_le: Vec<u8>,
        repeated_output_bf16_exact: bool,
        allocation_accounting_unchanged: bool,
    }

    #[derive(Debug)]
    struct ProbeExecution {
        device: Value,
        projections: BTreeMap<&'static str, ProjectionExecution>,
        reference_compute_capability_matches: bool,
    }

    fn expected_shape(name: &str) -> TestResult<Vec<u64>> {
        let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
        let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
        let kv = u64::try_from(QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION)?;
        match name {
            "p7_input_norm" => Ok(vec![sequence, hidden]),
            "p7_shadow_raw_q" | "layer0_q_proj_weight" | "raw_q/default_cublas_reduced_splitk" => {
                Ok(vec![hidden, hidden]).map(|mut shape| {
                    if name != "layer0_q_proj_weight" {
                        shape[0] = sequence;
                    }
                    shape
                })
            }
            "layer0_k_proj_weight" | "layer0_v_proj_weight" => Ok(vec![kv, hidden]),
            "raw_k/default_cublas_reduced_splitk" | "raw_v/default_cublas_reduced_splitk" => {
                Ok(vec![sequence, kv])
            }
            _ => Err(format!("unknown P11 Q/K/V tensor {name}").into()),
        }
    }

    fn require_exact_fields(value: &Value, fields: &[&str], label: &str) -> TestResult {
        let object = value
            .as_object()
            .ok_or_else(|| format!("{label} must be an object"))?;
        let expected: BTreeSet<_> = fields.iter().copied().collect();
        let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
        if actual != expected {
            return Err(
                format!("{label} fields differ: actual={actual:?} expected={expected:?}").into(),
            );
        }
        Ok(())
    }

    fn sidecar_tensor(
        header: &Map<String, Value>,
        data: &[u8],
        manifest_tensors: &Map<String, Value>,
        name: &str,
        expected_shape: &[u64],
    ) -> TestResult<Vec<u8>> {
        let manifest_record = manifest_tensors
            .get(name)
            .ok_or_else(|| format!("P11 manifest tensor {name} is missing"))?;
        require_exact_fields(
            manifest_record,
            &[
                "key",
                "shape",
                "dtype",
                "canonical_byte_order",
                "bf16_le_bytes",
                "bf16_le_sha256",
            ],
            "P11 manifest tensor",
        )?;
        let key = format!("trace/{name}");
        let expected_bytes = shape_byte_len(expected_shape)?;
        if manifest_record.get("key").and_then(Value::as_str) != Some(key.as_str())
            || manifest_record.get("shape")
                != Some(&Value::Array(
                    expected_shape.iter().copied().map(Value::from).collect(),
                ))
            || manifest_record.get("dtype").and_then(Value::as_str) != Some("bfloat16")
            || manifest_record
                .get("canonical_byte_order")
                .and_then(Value::as_str)
                != Some("little-endian-u16")
            || manifest_record.get("bf16_le_bytes").and_then(Value::as_u64)
                != Some(u64::try_from(expected_bytes)?)
        {
            return Err(format!("P11 manifest tensor {name} metadata differs").into());
        }
        let expected_sha256 = json_sha256(
            manifest_record
                .get("bf16_le_sha256")
                .ok_or("P11 manifest tensor SHA-256 is missing")?,
            "P11 manifest tensor SHA-256",
        )?;
        let record = header
            .get(&key)
            .ok_or_else(|| format!("P11 sidecar does not contain {key}"))?;
        require_exact_fields(
            record,
            &["dtype", "shape", "data_offsets"],
            "P11 sidecar tensor",
        )?;
        if record.get("dtype").and_then(Value::as_str) != Some("BF16")
            || record.get("shape")
                != Some(&Value::Array(
                    expected_shape.iter().copied().map(Value::from).collect(),
                ))
        {
            return Err(format!("P11 sidecar {key} metadata differs").into());
        }
        let offsets = record
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("P11 sidecar {key} offsets are missing"))?;
        if offsets.len() != 2 {
            return Err(format!("P11 sidecar {key} offset count differs").into());
        }
        let start = usize::try_from(offsets[0].as_u64().ok_or("P11 tensor start offset")?)?;
        let end = usize::try_from(offsets[1].as_u64().ok_or("P11 tensor end offset")?)?;
        if end < start || end - start != expected_bytes || end > data.len() {
            return Err(format!("P11 sidecar {key} range differs").into());
        }
        let tensor = data[start..end].to_vec();
        if sha256_hex(&tensor) != expected_sha256 {
            return Err(format!("P11 sidecar {key} SHA-256 differs").into());
        }
        validate_finite_bf16(&tensor, &key)?;
        Ok(tensor)
    }

    fn validate_default_policy(policy: &Value) -> TestResult {
        require_exact_fields(
            policy,
            &[
                "id",
                "preferred_blas_requested",
                "allow_bf16_reduced_precision_reduction_requested",
                "allow_bf16_reduced_precision_reduction_split_k_requested",
                "matmul_tf32_requested",
                "cudnn_tf32_requested",
                "operator",
                "bias",
                "preferred_blas_actual",
                "allow_bf16_reduced_precision_reduction_actual",
                "allow_bf16_reduced_precision_reduction_split_k_actual",
                "matmul_tf32_actual",
                "cudnn_tf32_actual",
            ],
            "P11 default policy",
        )?;
        let true_fields = [
            "allow_bf16_reduced_precision_reduction_requested",
            "allow_bf16_reduced_precision_reduction_split_k_requested",
            "allow_bf16_reduced_precision_reduction_actual",
            "allow_bf16_reduced_precision_reduction_split_k_actual",
        ];
        let false_fields = [
            "matmul_tf32_requested",
            "cudnn_tf32_requested",
            "bias",
            "matmul_tf32_actual",
            "cudnn_tf32_actual",
        ];
        if policy.get("id").and_then(Value::as_str) != Some("default_cublas_reduced_splitk")
            || policy
                .get("preferred_blas_requested")
                .and_then(Value::as_str)
                != Some("cublas")
            || policy.get("preferred_blas_actual").and_then(Value::as_str) != Some("cublas")
            || policy.get("operator").and_then(Value::as_str) != Some("torch.nn.functional.linear")
            || !true_fields
                .iter()
                .all(|field| policy.get(*field).and_then(Value::as_bool) == Some(true))
            || !false_fields
                .iter()
                .all(|field| policy.get(*field).and_then(Value::as_bool) == Some(false))
        {
            return Err("P11 default Cublas policy differs".into());
        }
        Ok(())
    }

    fn validate_projection_result(
        results: &Map<String, Value>,
        tensors: &Map<String, Value>,
        projection: ProjectionSpec,
    ) -> TestResult {
        let result = results
            .get(projection.identifier)
            .ok_or_else(|| format!("P11 result for {} is missing", projection.identifier))?;
        require_exact_fields(
            result,
            &[
                "identifier",
                "module_name",
                "weight_tensor",
                "output_tensor",
                "checkpoint_weight_key",
                "shape",
                "output_bf16_le_sha256",
                "repeated_bf16_exact",
                "p7_default_raw_q_comparison",
            ],
            "P11 projection result",
        )?;
        if result.get("identifier").and_then(Value::as_str) != Some(projection.identifier)
            || result.get("module_name").and_then(Value::as_str) != Some(projection.module_name)
            || result.get("weight_tensor").and_then(Value::as_str) != Some(projection.weight_name)
            || result.get("output_tensor").and_then(Value::as_str) != Some(projection.output_name)
            || result.get("checkpoint_weight_key").and_then(Value::as_str)
                != Some(projection.checkpoint_weight_key)
            || result.get("shape")
                != Some(
                    &json!({"m": CONTEXT_TOKEN_COUNT, "n": projection.width, "k": QWEN3B_HIDDEN_SIZE}),
                )
            || result.get("repeated_bf16_exact").and_then(Value::as_bool) != Some(true)
            || result.get("output_bf16_le_sha256").and_then(Value::as_str)
                != Some(projection.expected_output_sha256)
            || result.get("output_bf16_le_sha256")
                != tensors
                    .get(projection.output_name)
                    .and_then(|record| record.get("bf16_le_sha256"))
        {
            return Err(format!("P11 {} projection result differs", projection.identifier).into());
        }
        let comparison = result
            .get("p7_default_raw_q_comparison")
            .ok_or("P11 projection comparison is missing")?;
        if projection.identifier == "q" {
            require_exact_fields(
                comparison,
                &[
                    "bf16_exact",
                    "unequal_elements",
                    "total_elements",
                    "max_abs",
                ],
                "P11 Q/P7 comparison",
            )?;
            if comparison.get("bf16_exact").and_then(Value::as_bool) != Some(true)
                || comparison.get("unequal_elements").and_then(Value::as_u64) != Some(0)
                || comparison.get("total_elements").and_then(Value::as_u64)
                    != Some(u64::try_from(CONTEXT_TOKEN_COUNT * QWEN3B_HIDDEN_SIZE)?)
                || comparison.get("max_abs").and_then(Value::as_f64) != Some(0.0)
            {
                return Err("P11 Q/P7 exactness contract differs".into());
            }
        } else if !comparison.is_null() {
            return Err("P11 K/V must not claim a raw-Q comparison".into());
        }
        Ok(())
    }

    fn load_p11_qkv_artifact() -> TestResult<P11QkvArtifact> {
        let manifest_path = regular_file(
            &required_path(P11_MANIFEST_VARIABLE)?,
            "immutable P11 Q/K/V manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(P11_SIDECAR_VARIABLE)?,
            "immutable P11 Q/K/V sidecar",
        )?;
        if sha256_file(&manifest_path)? != P11_MANIFEST_SHA256
            || sha256_file(&sidecar_path)? != P11_SIDECAR_SHA256
        {
            return Err(
                "P11 direct-cuBLAS qualifier requires the pinned Q/K/V artifact hashes".into(),
            );
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
                "default_policy",
                "p7_binding",
                "model",
                "projection_results",
                "comparisons",
                "provenance",
                "sidecar",
                "tensors",
            ],
            "P11 manifest",
        )?;
        if manifest.get("schema_version").and_then(Value::as_str)
            != Some("riley.qwen3b-hf-eager-p2051-qkv-default-cublas-trace.v1")
            || manifest.get("artifact_kind").and_then(Value::as_str)
                != Some("qwen2.5-3b-hf-eager-bf16-p2051-raw-qkv-default-cublas-trace")
            || manifest.get("trace_id").and_then(Value::as_str)
                != Some("qwen3b-p2051-layer0-raw-qkv-default-cublas-v1")
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
            return Err("P11 manifest identity differs".into());
        }
        validate_default_policy(
            manifest
                .get("default_policy")
                .ok_or("P11 manifest default policy is missing")?,
        )?;
        let p7 = manifest
            .get("p7_binding")
            .ok_or("P11 manifest lacks the immutable P7 binding")?;
        if p7.get("manifest_sha256").and_then(Value::as_str) != Some(P7_MANIFEST_SHA256)
            || p7.get("sidecar_sha256").and_then(Value::as_str) != Some(P7_SIDECAR_SHA256)
            || p7.get("source_revision").and_then(Value::as_str) != Some(P7_SOURCE_REVISION)
            || p7.get("input_tensor_key").and_then(Value::as_str) != Some("trace/layer0/input_norm")
            || p7.get("raw_q_tensor_key").and_then(Value::as_str)
                != Some("trace/layer0/q_proj/unbiased_linear")
        {
            return Err("P11 P7 binding differs".into());
        }
        let source = manifest
            .get("provenance")
            .and_then(|value| value.get("source_repository"))
            .ok_or("P11 source provenance is missing")?;
        if source.get("git_revision").and_then(Value::as_str) != Some(P11_SOURCE_REVISION)
            || source.get("source_dirty").and_then(Value::as_bool) != Some(false)
        {
            return Err("P11 producer source provenance differs".into());
        }
        let scope = manifest.get("scope").ok_or("P11 scope is missing")?;
        if scope.get("model_id").and_then(Value::as_str) != Some(QWEN3B_MODEL_ID)
            || scope.get("model_revision").and_then(Value::as_str) != Some(QWEN3B_REVISION)
            || scope.get("operator").and_then(Value::as_str) != Some("torch.nn.functional.linear")
            || scope.get("offline_only").and_then(Value::as_bool) != Some(true)
            || scope.get("serving_path").and_then(Value::as_bool) != Some(false)
            || scope.get("shape")
                != Some(
                    &json!({"m": CONTEXT_TOKEN_COUNT, "k": QWEN3B_HIDDEN_SIZE, "q_n": QWEN3B_HIDDEN_SIZE, "kv_n": QWEN3B_KEY_VALUE_HEADS * QWEN3B_HEAD_DIMENSION}),
                )
        {
            return Err("P11 scope differs".into());
        }
        let results = manifest
            .get("projection_results")
            .and_then(Value::as_object)
            .ok_or("P11 projection results are missing")?;
        let tensors = manifest
            .get("tensors")
            .and_then(Value::as_object)
            .ok_or("P11 tensors are missing")?;
        let result_keys: BTreeSet<_> = results.keys().map(String::as_str).collect();
        let expected_result_keys: BTreeSet<_> =
            PROJECTIONS.iter().map(|item| item.identifier).collect();
        if result_keys != expected_result_keys {
            return Err("P11 projection result keys differ".into());
        }
        let tensor_keys: BTreeSet<_> = tensors.keys().map(String::as_str).collect();
        let expected_tensor_keys: BTreeSet<_> = P11_QKV_TENSOR_NAMES.into_iter().collect();
        if tensor_keys != expected_tensor_keys {
            return Err("P11 tensor keys differ".into());
        }
        for projection in PROJECTIONS {
            validate_projection_result(results, tensors, projection)?;
        }
        let q_comparison = manifest
            .get("comparisons")
            .and_then(|value| value.get("q_default_vs_p7_shadow_raw_q"))
            .ok_or("P11 Q/P7 comparison is missing")?;
        if q_comparison
            != results
                .get("q")
                .and_then(|value| value.get("p7_default_raw_q_comparison"))
            || manifest
                .get("comparisons")
                .and_then(|value| value.get("outputs_are_not_riley_results"))
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err("P11 comparison binding differs".into());
        }

        let bytes = fs::read(&sidecar_path)?;
        if bytes.len() < 8 {
            return Err("P11 sidecar is too short".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        if header_len == 0
            || header_len > MAX_SAFETENSORS_HEADER_BYTES
            || 8_usize
                .checked_add(header_len)
                .is_none_or(|end| end > bytes.len())
        {
            return Err("P11 sidecar header length differs".into());
        }
        let data_start = 8 + header_len;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
        let header = header
            .as_object()
            .ok_or("P11 sidecar header is not an object")?;
        let sidecar_keys: BTreeSet<_> = header
            .keys()
            .filter(|key| key.as_str() != "__metadata__")
            .map(String::as_str)
            .collect();
        let expected_sidecar_keys: BTreeSet<_> = P11_QKV_TENSOR_NAMES
            .iter()
            .map(|name| format!("trace/{name}"))
            .collect::<BTreeSet<_>>();
        if sidecar_keys != expected_sidecar_keys.iter().map(String::as_str).collect() {
            return Err("P11 sidecar key set differs".into());
        }
        let data = &bytes[data_start..];
        let input_bf16_le = sidecar_tensor(
            header,
            data,
            tensors,
            "p7_input_norm",
            &expected_shape("p7_input_norm")?,
        )?;
        let p7_shadow_raw_q_bf16_le = sidecar_tensor(
            header,
            data,
            tensors,
            "p7_shadow_raw_q",
            &expected_shape("p7_shadow_raw_q")?,
        )?;
        let mut projections = BTreeMap::new();
        for projection in PROJECTIONS {
            let weight_bf16_le = sidecar_tensor(
                header,
                data,
                tensors,
                projection.weight_name,
                &expected_shape(projection.weight_name)?,
            )?;
            let expected_output_bf16_le = sidecar_tensor(
                header,
                data,
                tensors,
                projection.output_name,
                &expected_shape(projection.output_name)?,
            )?;
            if sha256_hex(&expected_output_bf16_le) != projection.expected_output_sha256 {
                return Err(format!("P11 {} output hash differs", projection.identifier).into());
            }
            projections.insert(
                projection.identifier,
                P11ProjectionArtifact {
                    weight_bf16_le,
                    expected_output_bf16_le,
                },
            );
        }
        let q_expected = projections.get("q").ok_or("P11 Q projection is missing")?;
        if q_expected.expected_output_bf16_le != p7_shadow_raw_q_bf16_le {
            return Err("P11 default Q output differs from its P7 shadow raw-Q binding".into());
        }
        Ok(P11QkvArtifact {
            manifest_path,
            sidecar_path,
            input_bf16_le,
            p7_shadow_raw_q_bf16_le,
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
            return Err("direct-cuBLAS probe metadata violates the pinned Q/K/V contract".into());
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

    fn run_direct_cublas_qkv(artifact: &P11QkvArtifact) -> TestResult<ProbeExecution> {
        let runtime = CudaRuntime::initialize()?;
        if runtime.device_count() == 0 {
            return Err("remote GPU runner has no CUDA device".into());
        }
        let device = runtime.device(0)?;
        let properties = device.properties().clone();
        let reference_compute_capability_matches =
            properties.compute_capability() == P11_REFERENCE_COMPUTE_CAPABILITY;
        let device_record = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
            "p11_reference_compute_capability": P11_REFERENCE_COMPUTE_CAPABILITY,
            "p11_reference_compute_capability_matches": reference_compute_capability_matches,
        });
        let context = device.create_context()?;
        let mut stream = context.create_stream()?;
        let result = (|| -> TestResult<ProbeExecution> {
            let input_config = CudaGemmConfig::new(
                u64::try_from(CONTEXT_TOKEN_COUNT)?,
                u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                MAX_WORKSPACE_BYTES,
            )?;
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let input_native = bf16_le_to_native(&artifact.input_bf16_le)?;
            if input_native.len() != usize::try_from(input_config.input_bytes())? {
                return Err("P11 input byte length differs from direct-cuBLAS config".into());
            }
            let mut input = context.allocate_device_buffer(input_config.input_bytes())?;
            input.upload_from_slice(0, &input_native, &mut staging, &mut stream)?;
            let base_allocations = context.allocation_stats()?;
            let mut projections = BTreeMap::new();
            for projection in PROJECTIONS {
                let candidate = artifact
                    .projections
                    .get(projection.identifier)
                    .ok_or("P11 projection artifact is missing")?;
                let config = CudaGemmConfig::new(
                    u64::try_from(CONTEXT_TOKEN_COUNT)?,
                    u64::try_from(projection.width)?,
                    u64::try_from(QWEN3B_HIDDEN_SIZE)?,
                    MAX_WORKSPACE_BYTES,
                )?;
                let weight_native = bf16_le_to_native(&candidate.weight_bf16_le)?;
                if weight_native.len() != usize::try_from(config.weight_bytes())? {
                    return Err(format!(
                        "P11 {} weight byte length differs",
                        projection.identifier
                    )
                    .into());
                }
                let mut weight = context.allocate_device_buffer(config.weight_bytes())?;
                weight.upload_from_slice(0, &weight_native, &mut staging, &mut stream)?;
                let mut output = context.allocate_device_buffer(config.output_bytes())?;
                let mut plan = context.prepare_cublas_gemm_probe(config)?;
                let metadata = plan.metadata();
                if plan.config() != config {
                    return Err(format!(
                        "P11 {} plan config differs after prepare",
                        projection.identifier
                    )
                    .into());
                }
                validate_probe_metadata(metadata, config, properties.compute_capability())?;
                let allocations_before_execute = context.allocation_stats()?;
                execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
                let mut first_native = vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(0, &mut first_native, &mut staging, &mut stream)?;
                execute_probe(&mut plan, config, &input, &weight, &mut output, &mut stream)?;
                let mut repeated_native = vec![0_u8; usize::try_from(config.output_bytes())?];
                output.download_to_slice(0, &mut repeated_native, &mut staging, &mut stream)?;
                let repeated_output_bf16_exact = first_native == repeated_native;
                let allocation_accounting_unchanged =
                    context.allocation_stats()? == allocations_before_execute;
                let output_bf16_le = bf16_native_to_le(&first_native)?;
                plan.close()?;
                weight.close()?;
                output.close()?;
                if context.allocation_stats()? != base_allocations {
                    return Err(format!(
                        "P11 {} leaked a device allocation after close",
                        projection.identifier
                    )
                    .into());
                }
                projections.insert(
                    projection.identifier,
                    ProjectionExecution {
                        metadata: metadata_json(metadata),
                        output_bf16_le,
                        repeated_output_bf16_exact,
                        allocation_accounting_unchanged,
                    },
                );
            }
            input.close()?;
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err(
                    "P11 direct-cuBLAS qualifier left native allocations before context close"
                        .into(),
                );
            }
            Ok(ProbeExecution {
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
                "direct-cuBLAS raw-Q/K/V qualifier failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only P2051 direct-cuBLAS raw-Q/K/V arithmetic qualifier"]
    fn qwen3b_p2051_direct_cublas_default_matches_p11_raw_qkv() -> TestResult {
        let output = required_path(P11_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let artifact = load_p11_qkv_artifact()?;
        let execution = run_direct_cublas_qkv(&artifact)?;
        let mut projection_receipts = Map::new();
        let mut all_exact = true;
        let mut all_repeated = true;
        let mut all_allocations_unchanged = true;
        for projection in PROJECTIONS {
            let expected = artifact
                .projections
                .get(projection.identifier)
                .ok_or("P11 expected projection is missing")?;
            let observed = execution
                .projections
                .get(projection.identifier)
                .ok_or("P11 direct-cuBLAS projection is missing")?;
            let comparison = metrics(&expected.expected_output_bf16_le, &observed.output_bf16_le)?;
            let exact = comparison.get("bf16_exact").and_then(Value::as_bool) == Some(true);
            all_exact = all_exact && exact;
            all_repeated = all_repeated && observed.repeated_output_bf16_exact;
            all_allocations_unchanged =
                all_allocations_unchanged && observed.allocation_accounting_unchanged;
            projection_receipts.insert(
                projection.identifier.to_owned(),
                json!({
                    "shape": {"m": CONTEXT_TOKEN_COUNT, "n": projection.width, "k": QWEN3B_HIDDEN_SIZE},
                    "expected_default_cublas_bf16_le_sha256": projection.expected_output_sha256,
                    "direct_cublas_output_bf16_le_sha256": sha256_hex(&observed.output_bf16_le),
                    "default_cublas_comparison": comparison,
                    "repeated_output_bf16_exact": observed.repeated_output_bf16_exact,
                    "allocation_accounting_unchanged": observed.allocation_accounting_unchanged,
                    "metadata": observed.metadata,
                }),
            );
        }
        let q_expected = artifact
            .projections
            .get("q")
            .ok_or("P11 expected Q projection is missing")?;
        let q_p7_shadow_comparison = metrics(
            &artifact.p7_shadow_raw_q_bf16_le,
            &q_expected.expected_output_bf16_le,
        )?;
        let q_p7_exact = q_p7_shadow_comparison
            .get("bf16_exact")
            .and_then(Value::as_bool)
            == Some(true);
        let quality_pass = execution.reference_compute_capability_matches
            && all_exact
            && all_repeated
            && all_allocations_unchanged
            && q_p7_exact;
        let root = repository_root()?;
        let source_paths = [
            "crates/riley-runtime/tests/qwen3b_p2051_cublas_qkv_probe_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "crates/riley-cuda/src/ffi.rs",
            "kernels/src/cublas_gemm_probe.cu",
        ];
        let source_hashes = source_paths
            .into_iter()
            .map(|source| {
                let path =
                    regular_file(&root.join(source), "direct-cuBLAS Q/K/V qualifier source")?;
                Ok((source, sha256_file(&path)?))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let receipt = json!({
            "schema_version": P11_RESULT_SCHEMA_VERSION,
            "artifact_kind": P11_RESULT_ARTIFACT_KIND,
            "created_at_unix_seconds": unix_seconds()?,
            "quality_pass": quality_pass,
            "performance_claim_eligible": false,
            "vllm_comparison_eligible": false,
            "immutable_p11_artifact": {
                "manifest_path": artifact.manifest_path,
                "manifest_sha256": P11_MANIFEST_SHA256,
                "sidecar_path": artifact.sidecar_path,
                "sidecar_sha256": P11_SIDECAR_SHA256,
                "p7_manifest_sha256": P7_MANIFEST_SHA256,
                "p7_sidecar_sha256": P7_SIDECAR_SHA256,
                "p7_source_revision": P7_SOURCE_REVISION,
                "p11_source_revision": P11_SOURCE_REVISION,
            },
            "contract": {
                "trace_id": P11_RESULT_TRACE_ID,
                "endpoints": ["layer0.q_proj.raw_no_bias", "layer0.k_proj.raw_no_bias", "layer0.v_proj.raw_no_bias"],
                "operator": "direct cublasGemmEx OP_T(Wc), OP_N(Xc), BF16 I/O, FP32 compute, CUBLAS_DEFAULT_MATH",
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "cuda_graph": false,
                "command_batch": false,
            },
            "device": execution.device,
            "direct_cublas": {
                "projections": projection_receipts,
                "q_p7_shadow_raw_q_comparison": q_p7_shadow_comparison,
            },
            "source_hashes": source_hashes,
            "summary": {
                "reference_compute_capability_matches": execution.reference_compute_capability_matches,
                "all_default_cublas_qkv_exact": all_exact,
                "all_repeated_output_bf16_exact": all_repeated,
                "all_allocation_accounting_unchanged": all_allocations_unchanged,
                "q_default_matches_p7_shadow_raw_q": q_p7_exact,
                "interpretation": "A true result establishes only the pinned raw-Q/K/V arithmetic correspondence. It does not qualify bias, full-forward correctness, serving behavior, or performance.",
                "serving_selector_changed": false,
                "performance_claim_eligible": false,
                "vllm_comparison_eligible": false,
            },
        });
        write_artifact_exclusive(&output, &receipt)?;
        if !quality_pass {
            return Err(
                "direct-cuBLAS raw-Q/K/V candidate did not meet its pinned quality gate".into(),
            );
        }
        println!(
            "QWEN3B_P2051_DIRECT_CUBLAS_QKV trace_id={} quality_pass=true performance_claim_eligible=false",
            P11_RESULT_TRACE_ID,
        );
        Ok(())
    }
}
