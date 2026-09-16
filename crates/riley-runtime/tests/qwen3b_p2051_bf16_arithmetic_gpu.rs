//! Remote-only Qwen2.5-3B P2051 raw-Q BF16 arithmetic-policy consumer.
//!
//! This is deliberately a diagnostic-only consumer.  It binds the immutable
//! P7 projection trace and P9 offline PyTorch policy trace to the current
//! strict CUDA GEMM, but it does not change a serving selector or make a
//! performance claim.

#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

#[allow(dead_code, unused_imports)]
mod p2051_projection_contract {
    // Keep P7's historical consumer immutable.  P9 records the source hashes
    // of this consumer separately instead of changing P7's contract.
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    const P9_SCHEMA_VERSION: &str = "riley.qwen3b-hf-eager-p2051-bf16-arithmetic-trace.v1";
    const P9_ARTIFACT_KIND: &str = "qwen2.5-3b-hf-eager-bf16-p2051-raw-q-arithmetic-trace";
    const P9_RESULT_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-bf16-arithmetic-policy-consumer.v1";
    const P9_RESULT_ARTIFACT_KIND: &str =
        "qwen2.5-3b-riley-p2051-raw-q-bf16-arithmetic-policy-consumer";
    const P9_TRACE_ID: &str = "qwen3b-p2051-layer0-raw-q-bf16-arithmetic-v1";
    const P9_MANIFEST_VARIABLE: &str = "RILEY_QWEN3B_P2051_BF16_ARITHMETIC_MANIFEST";
    const P9_SIDECAR_VARIABLE: &str = "RILEY_QWEN3B_P2051_BF16_ARITHMETIC_SIDECAR";
    const P9_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_BF16_ARITHMETIC_OUTPUT";
    const P9_MANIFEST_SHA256: &str =
        "3079d4b1e6e3e654b92431d69cc0378ece76e87f50bb736710f95c487d61c0ed";
    const P9_SIDECAR_SHA256: &str =
        "11fe03272a5cbfcf2aed5ea7adee3232641aa181d1eb92f10736897c15d3ac08";
    const P9_SOURCE_REVISION: &str = "f55a91fd6b8a9a84fcc76d6eb447ba0e4d23b4a3";
    const P7_MANIFEST_SHA256: &str =
        "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7";
    const P7_SIDECAR_SHA256: &str =
        "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96";
    const P7_SOURCE_REVISION: &str = "3e4f6f1ead8727cd01e52a30e4673d1c83953606";
    const P9_REFERENCE_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);
    const P9_POLICY_OUTPUT_SHA256: &str =
        "5ddaf741ef847ffb997041b2b4cfed9811fdacc64d7fcca9a3ca27dd48166eb5";
    const P7_DEFAULT_OUTPUT_SHA256: &str =
        "9353470bc0d4110218b9c4584ea782257d5a59888db5a3c0479131e5ba46ec8f";
    const P9_REDUCED_CONTROL_UNEQUAL: u64 = 1_087_133;
    const P9_RAW_Q_ELEMENT_COUNT: u64 = 4_200_448;

    const P9_TENSOR_NAMES: [&str; 6] = [
        "p7_input_norm",
        "layer0_q_proj_weight",
        "p7_shadow_raw_q",
        "raw_q/p7_default_cublas_reduced_splitk",
        "raw_q/cublas_reduced_off_splitk_on",
        "raw_q/cublaslt_reduced_off_splitk_off",
    ];
    const P9_POLICY_IDS: [&str; 3] = [
        "p7_default_cublas_reduced_splitk",
        "cublas_reduced_off_splitk_on",
        "cublaslt_reduced_off_splitk_off",
    ];
    const P9_SOURCE_RECORDS: [(&str, &str); 8] = [
        (
            "hf_calibration",
            "tools/python/reference/riley_reference/hf_calibration.py",
        ),
        (
            "p7_projection_trace",
            "tools/python/reference/riley_reference/qwen3b_p2051_projection_trace.py",
        ),
        (
            "p9_bf16_arithmetic_trace",
            "tools/python/reference/riley_reference/qwen3b_p2051_bf16_arithmetic_trace.py",
        ),
        (
            "qwen_serving_oracle",
            "tools/python/reference/riley_reference/qwen3b_serving_oracle.py",
        ),
        ("reference_lock", "tools/python/reference/uv.lock"),
        ("reference_project", "tools/python/reference/pyproject.toml"),
        ("reference_python", "tools/python/reference/.python-version"),
        (
            "stage_trace_support",
            "tools/python/reference/riley_reference/qwen3b_stage_trace.py",
        ),
    ];

    #[derive(Debug)]
    struct P9ArithmeticArtifact {
        manifest_path: PathBuf,
        manifest_sha256: String,
        sidecar_path: PathBuf,
        sidecar_sha256: String,
        p7_manifest_sha256: String,
        p7_sidecar_sha256: String,
        p7_checkpoint_receipt_sha256: String,
        p7_source_revision: String,
        tensors: BTreeMap<String, Vec<u8>>,
    }

    #[derive(Debug)]
    struct P9StrictExecution {
        device: Value,
        algorithm: Value,
        strict_raw_q_le: Vec<u8>,
        repeated_output_bf16_exact: bool,
        allocation_accounting_unchanged: bool,
        reference_compute_capability_matches: bool,
        input_provenance: Value,
        weight_provenance: Value,
    }

    fn p9_tensor_shape(name: &str) -> TestResult<Vec<u64>> {
        let sequence = u64::try_from(CONTEXT_TOKEN_COUNT)?;
        let hidden = u64::try_from(QWEN3B_HIDDEN_SIZE)?;
        match name {
            "p7_input_norm"
            | "p7_shadow_raw_q"
            | "raw_q/p7_default_cublas_reduced_splitk"
            | "raw_q/cublas_reduced_off_splitk_on"
            | "raw_q/cublaslt_reduced_off_splitk_off" => Ok(vec![sequence, hidden]),
            "layer0_q_proj_weight" => Ok(vec![hidden, hidden]),
            _ => Err(format!("unknown P9 BF16 arithmetic tensor {name}").into()),
        }
    }

    fn p9_tensor_key(name: &str) -> String {
        format!("trace/{name}")
    }

    fn p9_metadata_json(metadata: CudaGemmAlgorithmMetadata) -> Value {
        json!({
            "backend_id": metadata.backend_id(),
            "algorithm_id": metadata.algorithm_id(),
            "tile_id": metadata.tile_id(),
            "stages_id": metadata.stages_id(),
            "split_k": metadata.split_k(),
            "reduction_scheme": metadata.reduction_scheme(),
            "cta_swizzling": metadata.cta_swizzling(),
            "custom_option": metadata.custom_option(),
            "numerical_implementation_flags": metadata.numerical_implementation_flags(),
            "workspace_bytes": metadata.workspace_bytes(),
            "deterministic": metadata.deterministic(),
            "compute_capability": metadata.compute_capability(),
            "runtime_version": metadata.runtime_version(),
            "cublaslt_version": metadata.cublaslt_version(),
        })
    }

    fn p9_source_hashes(root: &Path) -> TestResult<Value> {
        let sources = [
            "crates/riley-runtime/tests/qwen3b_p2051_bf16_arithmetic_gpu.rs",
            "crates/riley-runtime/tests/support/qwen3b_p2051_projection_p7_contract.rs",
            "crates/riley-runtime/tests/qwen3b_p2051_bias_epilogue_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "kernels/src/gemm.cu",
        ];
        let mut records = Map::new();
        for source in sources {
            let path = regular_file(&root.join(source), "P9 Rust consumer source")?;
            records.insert(
                source.to_owned(),
                json!({"path": source, "sha256": sha256_file(&path)?}),
            );
        }
        Ok(Value::Object(records))
    }

    fn p9_policy_output_name(id: &str) -> String {
        format!("raw_q/{id}")
    }

    fn p9_policy_record(
        policy: &Value,
        id: &str,
        preferred_blas: &str,
        reduced_precision: bool,
        split_k: bool,
        role: &str,
        expected_output_sha256: &str,
        expected_p7_exact: bool,
        expected_p7_unequal: u64,
        expected_p7_max_abs: f64,
    ) -> TestResult {
        require_exact_fields(
            policy,
            &[
                "allow_bf16_reduced_precision_reduction_actual",
                "allow_bf16_reduced_precision_reduction_requested",
                "allow_bf16_reduced_precision_reduction_split_k_actual",
                "allow_bf16_reduced_precision_reduction_split_k_requested",
                "bias",
                "cudnn_tf32_actual",
                "cudnn_tf32_requested",
                "error",
                "id",
                "matmul_tf32_actual",
                "matmul_tf32_requested",
                "operand_shape",
                "operator",
                "output_bf16_le_sha256",
                "output_tensor",
                "p7_default_comparison",
                "preferred_blas_actual",
                "preferred_blas_requested",
                "repeated_bf16_exact",
                "role",
                "runtime_status",
            ],
            "P9 policy record",
        )?;
        let boolean_matches = [
            "allow_bf16_reduced_precision_reduction_actual",
            "allow_bf16_reduced_precision_reduction_requested",
        ]
        .iter()
        .all(|field| policy.get(*field).and_then(Value::as_bool) == Some(reduced_precision))
            && [
                "allow_bf16_reduced_precision_reduction_split_k_actual",
                "allow_bf16_reduced_precision_reduction_split_k_requested",
            ]
            .iter()
            .all(|field| policy.get(*field).and_then(Value::as_bool) == Some(split_k))
            && [
                "bias",
                "cudnn_tf32_actual",
                "cudnn_tf32_requested",
                "matmul_tf32_actual",
                "matmul_tf32_requested",
            ]
            .iter()
            .all(|field| policy.get(*field).and_then(Value::as_bool) == Some(false));
        if !boolean_matches
            || policy.get("id").and_then(Value::as_str) != Some(id)
            || policy.get("error") != Some(&Value::Null)
            || policy.get("operator").and_then(Value::as_str) != Some("torch.nn.functional.linear")
            || policy.get("operand_shape")
                != Some(
                    &json!({"m": CONTEXT_TOKEN_COUNT, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE}),
                )
            || policy.get("output_tensor").and_then(Value::as_str)
                != Some(p9_policy_output_name(id).as_str())
            || policy.get("output_bf16_le_sha256").and_then(Value::as_str)
                != Some(expected_output_sha256)
            || policy.get("preferred_blas_actual").and_then(Value::as_str) != Some(preferred_blas)
            || policy
                .get("preferred_blas_requested")
                .and_then(Value::as_str)
                != Some(preferred_blas)
            || policy.get("repeated_bf16_exact").and_then(Value::as_bool) != Some(true)
            || policy.get("role").and_then(Value::as_str) != Some(role)
            || policy.get("runtime_status").and_then(Value::as_str) != Some("captured")
        {
            return Err(format!("P9 policy {id} contract differs").into());
        }
        require_exact_fields(
            policy
                .get("p7_default_comparison")
                .ok_or("P9 policy comparison is missing")?,
            &[
                "bf16_exact",
                "max_abs",
                "total_elements",
                "unequal_elements",
            ],
            "P9 policy P7-default comparison",
        )?;
        let comparison = &policy["p7_default_comparison"];
        if comparison.get("bf16_exact").and_then(Value::as_bool) != Some(expected_p7_exact)
            || comparison.get("unequal_elements").and_then(Value::as_u64)
                != Some(expected_p7_unequal)
            || comparison.get("total_elements").and_then(Value::as_u64)
                != Some(P9_RAW_Q_ELEMENT_COUNT)
            || comparison.get("max_abs").and_then(Value::as_f64) != Some(expected_p7_max_abs)
        {
            return Err(format!("P9 policy {id} P7-default comparison differs").into());
        }
        Ok(())
    }

    fn parse_p9_sidecar(
        manifest: &Value,
        sidecar_path: &Path,
    ) -> TestResult<BTreeMap<String, Vec<u8>>> {
        let source = regular_file(sidecar_path, "P9 BF16 arithmetic sidecar")?;
        let bytes = fs::read(&source)?;
        require_exact_fields(
            manifest
                .get("sidecar")
                .ok_or("P9 sidecar record is missing")?,
            &["format", "path", "sha256", "tensor_count"],
            "P9 sidecar record",
        )?;
        let expected_name = source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("P9 sidecar basename is invalid")?;
        if manifest["sidecar"]["format"].as_str() != Some("safetensors")
            || manifest["sidecar"]["path"].as_str() != Some(expected_name)
            || manifest["sidecar"]["tensor_count"].as_u64()
                != Some(u64::try_from(P9_TENSOR_NAMES.len())?)
            || sha256_hex(&bytes) != P9_SIDECAR_SHA256
            || manifest["sidecar"]["sha256"].as_str() != Some(P9_SIDECAR_SHA256)
        {
            return Err("P9 sidecar binding differs".into());
        }
        if bytes.len() < 8 {
            return Err("P9 sidecar is too short".into());
        }
        let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into()?))?;
        if header_len == 0
            || header_len > MAX_SAFETENSORS_HEADER_BYTES
            || 8 + header_len > bytes.len()
        {
            return Err("P9 sidecar header differs".into());
        }
        let data_start = 8_usize
            .checked_add(header_len)
            .ok_or("P9 sidecar data offset overflows")?;
        let header: Value = serde_json::from_slice(&bytes[8..data_start])?;
        let header = header
            .as_object()
            .ok_or("P9 sidecar header is not an object")?;
        let expected_keys: BTreeSet<_> = P9_TENSOR_NAMES
            .iter()
            .map(|name| p9_tensor_key(name))
            .collect();
        let actual_keys: BTreeSet<_> = header
            .keys()
            .filter(|key| key.as_str() != "__metadata__")
            .cloned()
            .collect();
        if actual_keys != expected_keys {
            return Err("P9 sidecar tensor key set differs".into());
        }
        let tensor_records = manifest["tensors"]
            .as_object()
            .ok_or("P9 tensor records are missing")?;
        let expected_names: BTreeSet<_> = P9_TENSOR_NAMES.iter().copied().collect();
        let actual_names: BTreeSet<_> = tensor_records.keys().map(String::as_str).collect();
        if actual_names != expected_names {
            return Err("P9 tensor record key set differs".into());
        }
        let mut ranges = Vec::with_capacity(P9_TENSOR_NAMES.len());
        let mut tensors = BTreeMap::new();
        for name in P9_TENSOR_NAMES {
            let shape = p9_tensor_shape(name)?;
            let key = p9_tensor_key(name);
            let record = tensor_records
                .get(name)
                .ok_or("P9 tensor record is missing")?;
            require_exact_fields(
                record,
                &[
                    "bf16_le_bytes",
                    "bf16_le_sha256",
                    "canonical_byte_order",
                    "dtype",
                    "key",
                    "shape",
                ],
                "P9 tensor record",
            )?;
            let expected_bytes = shape_byte_len(&shape)?;
            let expected_sha = json_sha256(
                record
                    .get("bf16_le_sha256")
                    .ok_or("P9 tensor SHA-256 is missing")?,
                "P9 tensor SHA-256",
            )?;
            if record.get("key").and_then(Value::as_str) != Some(key.as_str())
                || record.get("dtype").and_then(Value::as_str) != Some("bfloat16")
                || record.get("canonical_byte_order").and_then(Value::as_str)
                    != Some("little-endian-u16")
                || record.get("shape")
                    != Some(&Value::Array(
                        shape.iter().copied().map(Value::from).collect(),
                    ))
                || record.get("bf16_le_bytes").and_then(Value::as_u64)
                    != Some(u64::try_from(expected_bytes)?)
            {
                return Err(format!("P9 tensor {name} metadata differs").into());
            }
            let entry = header.get(&key).ok_or("P9 sidecar tensor is missing")?;
            require_exact_fields(
                entry,
                &["data_offsets", "dtype", "shape"],
                "P9 safetensors entry",
            )?;
            if entry.get("dtype").and_then(Value::as_str) != Some("BF16")
                || entry.get("shape")
                    != Some(&Value::Array(
                        shape.iter().copied().map(Value::from).collect(),
                    ))
            {
                return Err(format!("P9 sidecar tensor {name} metadata differs").into());
            }
            let offsets = entry
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or("P9 sidecar offsets are missing")?;
            if offsets.len() != 2 {
                return Err("P9 sidecar offset count differs".into());
            }
            let start = usize::try_from(offsets[0].as_u64().ok_or("P9 offset start")?)?;
            let end = usize::try_from(offsets[1].as_u64().ok_or("P9 offset end")?)?;
            if end < start || end - start != expected_bytes || data_start + end > bytes.len() {
                return Err(format!("P9 sidecar tensor {name} range differs").into());
            }
            let raw = bytes[data_start + start..data_start + end].to_vec();
            if sha256_hex(&raw) != expected_sha {
                return Err(format!("P9 sidecar tensor {name} SHA-256 differs").into());
            }
            validate_finite_bf16(&raw, &format!("P9 tensor {name}"))?;
            ranges.push((start, end));
            tensors.insert(name.to_owned(), raw);
        }
        ranges.sort_unstable();
        let mut expected_start = 0_usize;
        for (start, end) in ranges {
            if start != expected_start {
                return Err("P9 sidecar offsets are non-contiguous".into());
            }
            expected_start = end;
        }
        if data_start + expected_start != bytes.len() {
            return Err("P9 sidecar has trailing bytes".into());
        }
        Ok(tensors)
    }

    fn load_p9_artifact() -> TestResult<P9ArithmeticArtifact> {
        let manifest_path = regular_file(
            &required_path(P9_MANIFEST_VARIABLE)?,
            "P9 BF16 arithmetic manifest",
        )?;
        let sidecar_path = regular_file(
            &required_path(P9_SIDECAR_VARIABLE)?,
            "P9 BF16 arithmetic sidecar",
        )?;
        let payload = fs::read(&manifest_path)?;
        let manifest_sha256 = sha256_hex(&payload);
        if manifest_sha256 != P9_MANIFEST_SHA256 {
            return Err("P9 BF16 arithmetic manifest SHA-256 differs".into());
        }
        let manifest: Value = serde_json::from_slice(&payload)?;
        require_exact_fields(
            &manifest,
            &[
                "artifact_kind",
                "capture_status",
                "comparisons",
                "created_at",
                "model",
                "p7_binding",
                "performance_claim_eligible",
                "policies",
                "producer",
                "provenance",
                "quality_pass",
                "schema_version",
                "scope",
                "serving_selector_changed",
                "sidecar",
                "tensors",
                "trace_id",
                "vllm_comparison_eligible",
            ],
            "P9 BF16 arithmetic manifest",
        )?;
        if manifest["schema_version"].as_str() != Some(P9_SCHEMA_VERSION)
            || manifest["artifact_kind"].as_str() != Some(P9_ARTIFACT_KIND)
            || manifest["trace_id"].as_str() != Some(P9_TRACE_ID)
            || manifest["capture_status"].as_str() != Some("captured")
            || manifest["quality_pass"].as_bool() != Some(true)
            || manifest["performance_claim_eligible"].as_bool() != Some(false)
            || manifest["vllm_comparison_eligible"].as_bool() != Some(false)
            || manifest["serving_selector_changed"].as_bool() != Some(false)
        {
            return Err("P9 BF16 arithmetic manifest identity differs".into());
        }
        require_exact_fields(
            &manifest["model"],
            &[
                "checkpoint_path",
                "checkpoint_receipt_filename",
                "checkpoint_receipt_sha256",
            ],
            "P9 model binding",
        )?;
        if manifest["model"]["checkpoint_path"]
            .as_str()
            .is_none_or(str::is_empty)
            || manifest["model"]["checkpoint_receipt_filename"].as_str()
                != Some(CHECKPOINT_RECEIPT_FILENAME)
        {
            return Err("P9 model identity differs".into());
        }
        let p7_binding = manifest["p7_binding"]
            .as_object()
            .ok_or("P9 P7 binding is missing")?;
        require_exact_fields(
            &manifest["p7_binding"],
            &[
                "checkpoint_q_weight_bf16_le_sha256",
                "checkpoint_q_weight_key",
                "checkpoint_receipt_sha256",
                "input_bf16_le_sha256",
                "input_tensor_key",
                "manifest_filename",
                "manifest_sha256",
                "raw_q_bf16_le_sha256",
                "raw_q_tensor_key",
                "sidecar_filename",
                "sidecar_sha256",
                "source_revision",
            ],
            "P9 P7 binding",
        )?;
        if p7_binding
            .get("checkpoint_q_weight_key")
            .and_then(Value::as_str)
            != Some("model.layers.0.self_attn.q_proj.weight")
            || p7_binding.get("input_tensor_key").and_then(Value::as_str)
                != Some("trace/layer0/input_norm")
            || p7_binding.get("raw_q_tensor_key").and_then(Value::as_str)
                != Some("trace/layer0/q_proj/unbiased_linear")
            || p7_binding.get("manifest_filename").and_then(Value::as_str)
                != Some("hf-p2051-projection.json")
            || p7_binding.get("sidecar_filename").and_then(Value::as_str)
                != Some("hf-p2051-projection.safetensors")
            || p7_binding.get("manifest_sha256").and_then(Value::as_str) != Some(P7_MANIFEST_SHA256)
            || p7_binding.get("sidecar_sha256").and_then(Value::as_str) != Some(P7_SIDECAR_SHA256)
            || p7_binding.get("source_revision").and_then(Value::as_str) != Some(P7_SOURCE_REVISION)
        {
            return Err("P9 P7 binding identity differs".into());
        }
        let p7_checkpoint_receipt_sha256 = json_sha256(
            p7_binding
                .get("checkpoint_receipt_sha256")
                .ok_or("P9 P7 checkpoint receipt SHA-256 is missing")?,
            "P9 P7 checkpoint receipt SHA-256",
        )?;
        for field in [
            "checkpoint_q_weight_bf16_le_sha256",
            "input_bf16_le_sha256",
            "raw_q_bf16_le_sha256",
        ] {
            let _ = json_sha256(
                p7_binding
                    .get(field)
                    .ok_or("P9 P7 tensor SHA-256 is missing")?,
                "P9 P7 tensor SHA-256",
            )?;
        }
        require_exact_fields(
            &manifest["comparisons"],
            &[
                "p7_default_vs_p7_shadow_raw_q",
                "policy_outputs_are_not_riley_results",
            ],
            "P9 comparisons",
        )?;
        if manifest["comparisons"]["policy_outputs_are_not_riley_results"].as_bool() != Some(true) {
            return Err("P9 policy ownership marker differs".into());
        }
        require_exact_fields(
            &manifest["comparisons"]["p7_default_vs_p7_shadow_raw_q"],
            &[
                "bf16_exact",
                "max_abs",
                "total_elements",
                "unequal_elements",
            ],
            "P9 P7-default comparison",
        )?;
        let p7_default = &manifest["comparisons"]["p7_default_vs_p7_shadow_raw_q"];
        if p7_default["bf16_exact"].as_bool() != Some(true)
            || p7_default["unequal_elements"].as_u64() != Some(0)
            || p7_default["total_elements"].as_u64() != Some(P9_RAW_Q_ELEMENT_COUNT)
            || p7_default["max_abs"].as_f64() != Some(0.0)
        {
            return Err("P9 P7-default reproduction differs".into());
        }
        require_exact_fields(
            &manifest["scope"],
            &[
                "endpoint",
                "model_id",
                "model_revision",
                "offline_only",
                "operator",
                "serving_path",
                "shape",
            ],
            "P9 scope",
        )?;
        if manifest["scope"]["endpoint"].as_str() != Some("layer0.q_proj.raw_no_bias")
            || manifest["scope"]["model_id"].as_str() != Some(QWEN3B_MODEL_ID)
            || manifest["scope"]["model_revision"].as_str() != Some(QWEN3B_REVISION)
            || manifest["scope"]["offline_only"].as_bool() != Some(true)
            || manifest["scope"]["operator"].as_str() != Some("torch.nn.functional.linear")
            || manifest["scope"]["serving_path"].as_bool() != Some(false)
            || manifest["scope"]["shape"]
                != json!({"m": CONTEXT_TOKEN_COUNT, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE})
        {
            return Err("P9 scope differs".into());
        }
        let policy_records = manifest["policies"]
            .as_array()
            .ok_or("P9 policies are missing")?;
        let policy_ids: BTreeSet<_> = policy_records
            .iter()
            .map(|policy| {
                policy
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or("P9 policy id is missing")
            })
            .collect::<Result<_, _>>()?;
        let expected_policy_ids: BTreeSet<_> = P9_POLICY_IDS.iter().copied().collect();
        if policy_records.len() != P9_POLICY_IDS.len() || policy_ids != expected_policy_ids {
            return Err("P9 policy id set differs".into());
        }
        let policy = |id| {
            policy_records
                .iter()
                .find(|record| record.get("id").and_then(Value::as_str) == Some(id))
                .ok_or_else(|| format!("P9 policy {id} is missing"))
        };
        p9_policy_record(
            policy("p7_default_cublas_reduced_splitk")?,
            "p7_default_cublas_reduced_splitk",
            "cublas",
            true,
            true,
            "p7_default_reproduction",
            P7_DEFAULT_OUTPUT_SHA256,
            true,
            0,
            0.0,
        )?;
        p9_policy_record(
            policy("cublas_reduced_off_splitk_on")?,
            "cublas_reduced_off_splitk_on",
            "cublas",
            false,
            true,
            "reduced_precision_control",
            P9_POLICY_OUTPUT_SHA256,
            false,
            P9_REDUCED_CONTROL_UNEQUAL,
            0.0625,
        )?;
        p9_policy_record(
            policy("cublaslt_reduced_off_splitk_off")?,
            "cublaslt_reduced_off_splitk_off",
            false,
            false,
            "explicit_cublaslt_full_reduction_control",
            P9_POLICY_OUTPUT_SHA256,
            false,
            P9_REDUCED_CONTROL_UNEQUAL,
            0.0625,
        )?;
        let provenance = manifest["provenance"]["source_repository"]
            .as_object()
            .ok_or("P9 source provenance is missing")?;
        require_exact_fields(
            &manifest["provenance"],
            &["source_repository"],
            "P9 provenance",
        )?;
        require_exact_fields(
            &manifest["provenance"]["source_repository"],
            &[
                "git_revision",
                "source_dirty",
                "source_status_sha256",
                "sources",
            ],
            "P9 source provenance",
        )?;
        if provenance.get("git_revision").and_then(Value::as_str) != Some(P9_SOURCE_REVISION)
            || provenance.get("source_dirty").and_then(Value::as_bool) != Some(false)
            || provenance
                .get("source_status_sha256")
                .and_then(Value::as_str)
                != Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        {
            return Err("P9 source provenance identity differs".into());
        }
        let source_records = provenance
            .get("sources")
            .and_then(Value::as_object)
            .ok_or("P9 source records are missing")?;
        let expected_source_names: BTreeSet<_> =
            P9_SOURCE_RECORDS.iter().map(|(name, _)| *name).collect();
        let actual_source_names: BTreeSet<_> = source_records.keys().map(String::as_str).collect();
        if actual_source_names != expected_source_names {
            return Err("P9 source record set differs".into());
        }
        let root = repository_root()?;
        for (name, path) in P9_SOURCE_RECORDS {
            validate_source_record(
                &root,
                source_records
                    .get(name)
                    .ok_or("P9 source record is missing")?,
                path,
                &format!("P9 source {name}"),
            )?;
        }
        let tensors = parse_p9_sidecar(&manifest, &sidecar_path)?;
        if sha256_file(&sidecar_path)? != P9_SIDECAR_SHA256 {
            return Err("P9 sidecar SHA-256 differs after parsing".into());
        }
        Ok(P9ArithmeticArtifact {
            manifest_path,
            manifest_sha256,
            sidecar_path,
            sidecar_sha256: P9_SIDECAR_SHA256.to_owned(),
            p7_manifest_sha256: P7_MANIFEST_SHA256.to_owned(),
            p7_sidecar_sha256: P7_SIDECAR_SHA256.to_owned(),
            p7_checkpoint_receipt_sha256,
            p7_source_revision: P7_SOURCE_REVISION.to_owned(),
            tensors,
        })
    }

    fn p9_upload(
        context: &CudaContext,
        stream: &mut CudaStream,
        staging: &mut CudaPinnedHostBuffer,
        bytes: &[u8],
    ) -> TestResult<CudaDeviceBuffer> {
        let mut buffer = context.allocate_device_buffer(u64::try_from(bytes.len())?)?;
        buffer.upload_from_slice(0, bytes, staging, stream)?;
        Ok(buffer)
    }

    fn p9_download(
        output: &mut CudaDeviceBuffer,
        byte_len: usize,
        staging: &mut CudaPinnedHostBuffer,
        stream: &mut CudaStream,
    ) -> TestResult<Vec<u8>> {
        let mut bytes = vec![0_u8; byte_len];
        output.download_to_slice(0, &mut bytes, staging, stream)?;
        Ok(bytes)
    }

    fn p9_execute_strict(
        plan: &mut CudaPreparedGemm,
        config: CudaGemmConfig,
        metadata: CudaGemmAlgorithmMetadata,
        input: &CudaDeviceBuffer,
        weight: &CudaDeviceBuffer,
        output: &mut CudaDeviceBuffer,
        workspace: Option<&mut CudaDeviceBuffer>,
        stream: &mut CudaStream,
    ) -> TestResult {
        let mut params = GemmParams {
            input: CudaBufferSpan::new(input, CudaDType::BF16, 0, config.input_bytes())?,
            weight: CudaBufferSpan::new(weight, CudaDType::BF16, 0, config.weight_bytes())?,
            output: CudaBufferSpanMut::new(output, CudaDType::BF16, 0, config.output_bytes())?,
            workspace: workspace_span(workspace, metadata.workspace_bytes())?,
        };
        plan.execute(&mut params, stream)?;
        Ok(())
    }

    fn p9_validate_cross_bindings(
        p9: &P9ArithmeticArtifact,
        hf: &HfProjectionArtifact,
        model: &LoadedModel,
    ) -> TestResult<(Vec<u8>, Vec<u8>, Value, Value)> {
        if p9.p7_manifest_sha256 != hf.manifest_sha256
            || p9.p7_sidecar_sha256 != hf.sidecar_sha256
            || p9.p7_checkpoint_receipt_sha256 != hf.checkpoint_receipt_sha256
            || p9.p7_source_revision != P7_SOURCE_REVISION
        {
            return Err("P9 artifact does not bind the loaded immutable P7 artifact".into());
        }
        let p9_input = p9
            .tensors
            .get("p7_input_norm")
            .ok_or("P9 input tensor is missing")?;
        let p9_shadow = p9
            .tensors
            .get("p7_shadow_raw_q")
            .ok_or("P9 P7 raw-Q tensor is missing")?;
        let p9_default = p9
            .tensors
            .get("raw_q/p7_default_cublas_reduced_splitk")
            .ok_or("P9 default raw-Q tensor is missing")?;
        let p7_input = hf
            .tensors
            .get("layer0.input_norm")
            .ok_or("P7 input tensor is missing")?;
        let p7_shadow = hf
            .tensors
            .get("layer0.q_proj.unbiased_linear")
            .ok_or("P7 raw-Q tensor is missing")?;
        if p9_input != p7_input
            || p9_shadow != p7_shadow
            || p9_default != p7_shadow
            || sha256_hex(p9_input)
                != "e865d626ed2782bc596735628a730fdaaa93178e41ee8e91d76ba6c92872c1ab"
            || sha256_hex(p9_shadow) != P7_DEFAULT_OUTPUT_SHA256
        {
            return Err("P9 P7 input or raw-Q byte binding differs".into());
        }
        let q_projection = PROJECTIONS
            .iter()
            .copied()
            .find(|projection| projection.label == "q_proj")
            .ok_or("P9 Q projection contract is missing")?;
        let host = projection_host_tensors(model, q_projection)?;
        let host_weight_le = bf16_native_to_le(&host.weight)?;
        let p9_weight = p9
            .tensors
            .get("layer0_q_proj_weight")
            .ok_or("P9 Q weight tensor is missing")?;
        if host_weight_le != *p9_weight
            || sha256_hex(&host_weight_le)
                != "1e74e3883d71871e847369c796437fd0496f7d066584abb13e92483d80592e3f"
        {
            return Err("P9 Q weight does not bind the loaded checkpoint".into());
        }
        Ok((
            bf16_le_to_native(p9_input)?,
            host.weight,
            json!({
                "p9_bf16_le_sha256": sha256_hex(p9_input),
                "p7_bf16_le_sha256": sha256_hex(p7_input),
                "p9_matches_p7_bf16_exact": true,
            }),
            host.weight_provenance,
        ))
    }

    fn run_p9_strict_raw_q(
        input_native: &[u8],
        weight_native: &[u8],
        input_provenance: Value,
        weight_provenance: Value,
    ) -> TestResult<P9StrictExecution> {
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
        let gpu = runtime.device(0)?;
        let properties = gpu.properties().clone();
        let reference_compute_capability_matches =
            properties.compute_capability() == P9_REFERENCE_COMPUTE_CAPABILITY;
        let device = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
            "p9_reference_compute_capability": P9_REFERENCE_COMPUTE_CAPABILITY,
            "p9_reference_compute_capability_matches": reference_compute_capability_matches,
        });
        let context = gpu.create_context()?;
        let mut stream = context.create_stream()?;
        let result = (|| -> TestResult<P9StrictExecution> {
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let input = p9_upload(&context, &mut stream, &mut staging, input_native)?;
            let weight = p9_upload(&context, &mut stream, &mut staging, weight_native)?;
            let mut plan = context.prepare_gemm(config)?;
            let metadata = plan.algorithm_metadata();
            if plan.config() != config {
                return Err("P9 strict GEMM plan configuration differs".into());
            }
            validate_metadata(
                metadata,
                config,
                properties.compute_capability(),
                "P9 strict raw-Q",
            )?;
            let mut output = context.allocate_device_buffer(config.output_bytes())?;
            let mut workspace = if metadata.workspace_bytes() == 0 {
                None
            } else {
                Some(context.allocate_device_buffer(metadata.workspace_bytes())?)
            };
            let allocations_before = context.allocation_stats()?;
            p9_execute_strict(
                &mut plan,
                config,
                metadata,
                &input,
                &weight,
                &mut output,
                workspace.as_mut(),
                &mut stream,
            )?;
            let first_native = p9_download(
                &mut output,
                usize::try_from(config.output_bytes())?,
                &mut staging,
                &mut stream,
            )?;
            p9_execute_strict(
                &mut plan,
                config,
                metadata,
                &input,
                &weight,
                &mut output,
                workspace.as_mut(),
                &mut stream,
            )?;
            let repeated_native = p9_download(
                &mut output,
                usize::try_from(config.output_bytes())?,
                &mut staging,
                &mut stream,
            )?;
            let repeated_output_bf16_exact = first_native == repeated_native;
            let allocation_accounting_unchanged = context.allocation_stats()? == allocations_before;
            let strict_raw_q_le = bf16_native_to_le(&first_native)?;
            if strict_raw_q_le.len() != shape_byte_len(&p9_tensor_shape("p7_shadow_raw_q")?)? {
                return Err("P9 strict raw-Q output byte count differs".into());
            }
            plan.close()?;
            input.close()?;
            weight.close()?;
            output.close()?;
            if let Some(workspace) = workspace {
                workspace.close()?;
            }
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err("P9 strict raw-Q left a CUDA allocation before close".into());
            }
            Ok(P9StrictExecution {
                device,
                algorithm: p9_metadata_json(metadata),
                strict_raw_q_le,
                repeated_output_bf16_exact,
                allocation_accounting_unchanged,
                reference_compute_capability_matches,
                input_provenance,
                weight_provenance,
            })
        })();
        let cleanup = close_context(stream, context);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(run_error), Err(cleanup_error)) => Err(format!(
                "P9 strict raw-Q arithmetic consumer failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only Qwen2.5-3B P2051 BF16 arithmetic-policy consumer"]
    fn qwen3b_p2051_strict_raw_q_matches_bf16_policy_trace() -> TestResult {
        let output = required_path(P9_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let workload = load_workload()?;
        let teacher = load_teacher_prefix()?;
        let hf = load_hf_projection_artifact(&teacher, &workload)?;
        if hf.manifest_sha256 != P7_MANIFEST_SHA256 || hf.sidecar_sha256 != P7_SIDECAR_SHA256 {
            return Err("P9 arithmetic consumer must consume the immutable P7 artifact".into());
        }
        let p9 = load_p9_artifact()?;
        let model = load_model(&hf)?;
        let (input_native, weight_native, input_provenance, weight_provenance) =
            p9_validate_cross_bindings(&p9, &hf, &model)?;
        let execution = run_p9_strict_raw_q(
            &input_native,
            &weight_native,
            input_provenance,
            weight_provenance,
        )?;
        let mut policy_comparisons = Map::new();
        for id in P9_POLICY_IDS {
            let name = p9_policy_output_name(id);
            let expected = p9
                .tensors
                .get(&name)
                .ok_or("P9 policy output tensor is missing")?;
            policy_comparisons.insert(name, metrics(expected, &execution.strict_raw_q_le)?);
        }
        let strict_vs_p7_default = policy_comparisons
            .get("raw_q/p7_default_cublas_reduced_splitk")
            .ok_or("P9 default policy comparison is missing")?;
        let strict_vs_cublas_reduced_off = policy_comparisons
            .get("raw_q/cublas_reduced_off_splitk_on")
            .ok_or("P9 Cublas reduced-off comparison is missing")?;
        let strict_vs_cublaslt_reduced_off = policy_comparisons
            .get("raw_q/cublaslt_reduced_off_splitk_off")
            .ok_or("P9 CublasLt reduced-off comparison is missing")?;
        let strict_matches_cublas_reduced_off = bf16_exact(
            strict_vs_cublas_reduced_off,
            "P9 strict vs Cublas reduced-off",
        )?;
        let strict_matches_cublaslt_reduced_off = bf16_exact(
            strict_vs_cublaslt_reduced_off,
            "P9 strict vs CublasLt reduced-off",
        )?;
        let strict_differs_from_p7_default =
            !bf16_exact(strict_vs_p7_default, "P9 strict vs P7 default")?;
        let policy_correspondence_observed = execution.reference_compute_capability_matches
            && execution.repeated_output_bf16_exact
            && execution.allocation_accounting_unchanged
            && strict_matches_cublas_reduced_off
            && strict_matches_cublaslt_reduced_off
            && strict_differs_from_p7_default;
        let root = repository_root()?;
        let receipt = json!({
            "schema_version": P9_RESULT_SCHEMA_VERSION,
            "artifact_kind": P9_RESULT_ARTIFACT_KIND,
            "created_at_unix_seconds": unix_seconds()?,
            "quality_pass": policy_correspondence_observed,
            "performance_claim_eligible": false,
            "vllm_comparison_eligible": false,
            "immutable_p9_arithmetic_artifact": {
                "manifest_path": p9.manifest_path,
                "manifest_sha256": p9.manifest_sha256,
                "sidecar_path": p9.sidecar_path,
                "sidecar_sha256": p9.sidecar_sha256,
                "source_revision": P9_SOURCE_REVISION,
                "artifact_source_provenance_validated": true,
            },
            "immutable_p7_projection_artifact": {
                "manifest_path": hf.manifest_path,
                "manifest_sha256": hf.manifest_sha256,
                "sidecar_path": hf.sidecar_path,
                "sidecar_sha256": hf.sidecar_sha256,
                "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
                "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
                "source_revision": P7_SOURCE_REVISION,
                "artifact_source_provenance_validated": true,
                "checkpoint_model_binding_validated": true,
            },
            "source_hashes": p9_source_hashes(&root)?,
            "contract": {
                "trace_id": P9_TRACE_ID,
                "model_id": QWEN3B_MODEL_ID,
                "model_revision": QWEN3B_REVISION,
                "workload_case": QWEN3B_WORKLOAD_CASE,
                "shape": {"m": CONTEXT_TOKEN_COUNT, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE},
                "endpoint": "layer0.q_proj.raw_no_bias",
                "strict_operator": "riley_cuda::CudaPreparedGemm BF16 I/O with FP32 compute",
                "policy_oracles": "offline PyTorch torch.nn.functional.linear; not Riley or vLLM results",
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "cuda_graph": false,
            },
            "device": execution.device,
            "strict_raw_q": {
                "algorithm": execution.algorithm,
                "output_bf16_le_sha256": sha256_hex(&execution.strict_raw_q_le),
                "repeated_output_bf16_exact": execution.repeated_output_bf16_exact,
                "allocation_accounting_unchanged": execution.allocation_accounting_unchanged,
                "input": execution.input_provenance,
                "weight": execution.weight_provenance,
            },
            "policy_comparisons": policy_comparisons,
            "summary": {
                "reference_compute_capability_matches": execution.reference_compute_capability_matches,
                "strict_matches_cublas_reduced_off_splitk_on": strict_matches_cublas_reduced_off,
                "strict_matches_cublaslt_reduced_off_splitk_off": strict_matches_cublaslt_reduced_off,
                "strict_differs_from_p7_default_cublas_reduced_splitk": strict_differs_from_p7_default,
                "policy_correspondence_observed": policy_correspondence_observed,
                "interpretation": "A true result establishes only this bound raw-Q correspondence on the recorded GPU and operands; it does not qualify full-forward correctness, serving behavior, or performance.",
                "serving_selector_changed": false,
                "performance_claim_eligible": false,
                "vllm_comparison_eligible": false,
            },
        });
        // Preserve a create-only receipt before a numerical-gate failure so
        // the exact strict algorithm and all policy comparisons remain
        // available for the next diagnosis.
        write_artifact_exclusive(&output, &receipt)?;
        if !policy_correspondence_observed {
            return Err(
                "P9 strict raw-Q did not establish the recorded BF16 policy correspondence".into(),
            );
        }
        println!(
            "QWEN3B_P2051_BF16_ARITHMETIC trace_id={} policy_correspondence_observed=true performance_claim_eligible=false",
            P9_TRACE_ID,
        );
        Ok(())
    }
}
