//! Remote-only Qwen2.5-3B P2051 raw-Q geometry discriminator.
//!
//! This test deliberately reuses the immutable P7 projection artifact.  It
//! varies only the direct raw-Q GEMM descriptor and storage-span controls, so
//! it can distinguish M geometry, heuristic selection, and span layout before
//! any serving selector is changed.

#![cfg(feature = "cuda")]
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

#[allow(dead_code, unused_imports)]
mod p2051_projection_contract {
    // This is the frozen helper surface from the P7 consumer, deliberately
    // excluding P7's ignored test.  The P7 artifact validates its original
    // consumer hash; this separate test records both hashes in its own
    // receipt rather than modifying that historical consumer.
    include!("support/qwen3b_p2051_projection_p7_contract.rs");

    const GEOMETRY_SCHEMA_VERSION: &str = "riley.qwen3b-p2051-raw-q-geometry.v2";
    const GEOMETRY_ARTIFACT_KIND: &str = "qwen2.5-3b-riley-p2051-raw-q-geometry-discriminator";
    const GEOMETRY_OUTPUT_VARIABLE: &str = "RILEY_QWEN3B_P2051_RAW_Q_GEOMETRY_OUTPUT";
    const P7_MANIFEST_SHA256: &str =
        "d469e6fc0695e5fc8ec21c0c94bc7665d0d79c447f4d60e58c38ddf72fcd60f7";
    const P7_SIDECAR_SHA256: &str =
        "fffdebe4123a434ce572a6201b1d81c0bdb146aaad355db56342fc299acd6f96";
    const EXPECTED_P7_COMPUTE_CAPABILITY: (u32, u32) = (8, 9);
    const SPAN_OFFSET_BYTES: u64 = 256;
    const MAX_GEOMETRY_M: usize = 2_304;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum GeometryTail {
        Prefix,
        ZeroPad,
        RepeatLastInputRow,
    }

    impl GeometryTail {
        const fn label(self) -> &'static str {
            match self {
                Self::Prefix => "artifact-prefix",
                Self::ZeroPad => "explicit-bf16-zero",
                Self::RepeatLastInputRow => "repeat-last-artifact-input-row",
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct FreshGeometryCase {
        label: &'static str,
        m: usize,
        tail: GeometryTail,
        offset_bytes: u64,
    }

    const FRESH_CASES: [FreshGeometryCase; 6] = [
        FreshGeometryCase {
            label: "F2048",
            m: 2_048,
            tail: GeometryTail::Prefix,
            offset_bytes: 0,
        },
        FreshGeometryCase {
            label: "F2051",
            m: CONTEXT_TOKEN_COUNT,
            tail: GeometryTail::Prefix,
            offset_bytes: 0,
        },
        FreshGeometryCase {
            label: "F2052",
            m: 2_052,
            tail: GeometryTail::ZeroPad,
            offset_bytes: 0,
        },
        FreshGeometryCase {
            label: "F2080",
            m: 2_080,
            tail: GeometryTail::ZeroPad,
            offset_bytes: 0,
        },
        FreshGeometryCase {
            label: "F2176",
            m: 2_176,
            tail: GeometryTail::ZeroPad,
            offset_bytes: 0,
        },
        FreshGeometryCase {
            label: "F2304",
            m: MAX_GEOMETRY_M,
            tail: GeometryTail::ZeroPad,
            offset_bytes: 0,
        },
    ];

    #[derive(Clone, Copy, Debug)]
    struct AnchoredGeometryCase {
        label: &'static str,
        m: usize,
        anchor_label: &'static str,
        anchor_m: usize,
    }

    const ANCHORED_CASES: [AnchoredGeometryCase; 5] = [
        AnchoredGeometryCase {
            label: "A2051<-2048",
            m: CONTEXT_TOKEN_COUNT,
            anchor_label: "F2048",
            anchor_m: 2_048,
        },
        AnchoredGeometryCase {
            label: "A2052<-2051",
            m: 2_052,
            anchor_label: "F2051",
            anchor_m: CONTEXT_TOKEN_COUNT,
        },
        AnchoredGeometryCase {
            label: "A2080<-2051",
            m: 2_080,
            anchor_label: "F2051",
            anchor_m: CONTEXT_TOKEN_COUNT,
        },
        AnchoredGeometryCase {
            label: "A2176<-2051",
            m: 2_176,
            anchor_label: "F2051",
            anchor_m: CONTEXT_TOKEN_COUNT,
        },
        AnchoredGeometryCase {
            label: "A2304<-2051",
            m: MAX_GEOMETRY_M,
            anchor_label: "F2051",
            anchor_m: CONTEXT_TOKEN_COUNT,
        },
    ];

    #[derive(Debug)]
    struct GeometryObservation {
        record: Value,
        prefix_output_le: Vec<u8>,
    }

    #[derive(Debug)]
    struct GeometryExecution {
        device: Value,
        cases: Map<String, Value>,
        exact_cases: Vec<String>,
        tail_data_invariant: bool,
        admitted_offset_invariant: bool,
        anchored_algorithm_identity_invariant: bool,
        geometry_hypothesis: &'static str,
    }

    fn geometry_metadata_json(metadata: CudaGemmAlgorithmMetadata) -> Value {
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

    fn geometry_algorithm_identity_json(metadata: CudaGemmAlgorithmMetadata) -> Value {
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

    fn same_geometry_algorithm_identity(
        anchor: CudaGemmAlgorithmMetadata,
        child: CudaGemmAlgorithmMetadata,
    ) -> bool {
        geometry_algorithm_identity_json(anchor) == geometry_algorithm_identity_json(child)
    }

    fn geometry_config(m: usize) -> TestResult<CudaGemmConfig> {
        CudaGemmConfig::new(
            u64::try_from(m)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            u64::try_from(QWEN3B_HIDDEN_SIZE)?,
            MAX_WORKSPACE_BYTES,
        )
        .map_err(Into::into)
    }

    fn geometry_prefix_rows(m: usize) -> usize {
        m.min(CONTEXT_TOKEN_COUNT)
    }

    fn geometry_prefix_len(m: usize) -> TestResult<usize> {
        geometry_prefix_rows(m)
            .checked_mul(QWEN3B_HIDDEN_SIZE)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or_else(|| "geometry prefix byte count overflows".into())
    }

    fn geometry_input_len(m: usize) -> TestResult<usize> {
        m.checked_mul(QWEN3B_HIDDEN_SIZE)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or_else(|| "geometry input byte count overflows".into())
    }

    fn geometry_output_len(m: usize) -> TestResult<usize> {
        geometry_input_len(m)
    }

    fn geometry_input_payload(
        input_hf_le: &[u8],
        m: usize,
        tail: GeometryTail,
    ) -> TestResult<Vec<u8>> {
        let expected_input_len = geometry_input_len(CONTEXT_TOKEN_COUNT)?;
        if input_hf_le.len() != expected_input_len {
            return Err("P7 geometry input tensor byte count differs".into());
        }
        let output_len = geometry_input_len(m)?;
        let prefix_len = expected_input_len.min(output_len);
        let mut payload = vec![0_u8; output_len];
        payload[..prefix_len].copy_from_slice(&input_hf_le[..prefix_len]);
        if m > CONTEXT_TOKEN_COUNT && tail == GeometryTail::RepeatLastInputRow {
            let row_bytes = geometry_input_len(1)?;
            let last_row_begin = expected_input_len
                .checked_sub(row_bytes)
                .ok_or("P7 geometry input has no final row")?;
            for row in CONTEXT_TOKEN_COUNT..m {
                let begin = row
                    .checked_mul(row_bytes)
                    .ok_or("geometry repeated-tail row offset overflows")?;
                let end = begin
                    .checked_add(row_bytes)
                    .ok_or("geometry repeated-tail row end overflows")?;
                payload[begin..end]
                    .copy_from_slice(&input_hf_le[last_row_begin..expected_input_len]);
            }
        }
        Ok(payload)
    }

    fn geometry_tail_sha256(payload_le: &[u8]) -> TestResult<String> {
        let prefix_len = geometry_input_len(CONTEXT_TOKEN_COUNT)?;
        if payload_le.len() < prefix_len {
            return Ok(sha256_hex(&[]));
        }
        Ok(sha256_hex(&payload_le[prefix_len..]))
    }

    fn upload_payload(
        context: &CudaContext,
        stream: &mut CudaStream,
        staging: &mut CudaPinnedHostBuffer,
        payload: &[u8],
        offset_bytes: u64,
    ) -> TestResult<CudaDeviceBuffer> {
        let offset = usize::try_from(offset_bytes)?;
        let allocation_len = offset
            .checked_add(payload.len())
            .ok_or("geometry upload allocation byte count overflows")?;
        let mut host = vec![0_u8; allocation_len];
        host[offset..].copy_from_slice(payload);
        let mut buffer = context.allocate_device_buffer(u64::try_from(allocation_len)?)?;
        buffer.upload_from_slice(0, &host, staging, stream)?;
        Ok(buffer)
    }

    fn upload_existing_payload(
        buffer: &mut CudaDeviceBuffer,
        stream: &mut CudaStream,
        staging: &mut CudaPinnedHostBuffer,
        payload: &[u8],
        offset_bytes: u64,
    ) -> TestResult {
        let offset = usize::try_from(offset_bytes)?;
        let allocation_len = offset
            .checked_add(payload.len())
            .ok_or("geometry upload byte range overflows")?;
        if allocation_len > usize::try_from(buffer.byte_len())? {
            return Err("geometry upload payload exceeds its fixed device allocation".into());
        }
        buffer.upload_from_slice(offset_bytes, payload, staging, stream)?;
        Ok(())
    }

    fn download_geometry_output(
        output: &mut CudaDeviceBuffer,
        output_offset_bytes: u64,
        byte_len: usize,
        staging: &mut CudaPinnedHostBuffer,
        stream: &mut CudaStream,
    ) -> TestResult<Vec<u8>> {
        let mut bytes = vec![0_u8; byte_len];
        output.download_to_slice(output_offset_bytes, &mut bytes, staging, stream)?;
        Ok(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_strict_with_offsets(
        plan: &mut CudaPreparedGemm,
        config: CudaGemmConfig,
        metadata: CudaGemmAlgorithmMetadata,
        input: &CudaDeviceBuffer,
        input_offset_bytes: u64,
        weight: &CudaDeviceBuffer,
        weight_offset_bytes: u64,
        output: &mut CudaDeviceBuffer,
        output_offset_bytes: u64,
        workspace: Option<&mut CudaDeviceBuffer>,
        stream: &mut CudaStream,
    ) -> TestResult {
        let mut params = GemmParams {
            input: CudaBufferSpan::new(
                input,
                CudaDType::BF16,
                input_offset_bytes,
                config.input_bytes(),
            )?,
            weight: CudaBufferSpan::new(
                weight,
                CudaDType::BF16,
                weight_offset_bytes,
                config.weight_bytes(),
            )?,
            output: CudaBufferSpanMut::new(
                output,
                CudaDType::BF16,
                output_offset_bytes,
                config.output_bytes(),
            )?,
            workspace: workspace_span(workspace, metadata.workspace_bytes())?,
        };
        plan.execute(&mut params, stream)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_geometry_plan(
        context: &CudaContext,
        stream: &mut CudaStream,
        staging: &mut CudaPinnedHostBuffer,
        plan: &mut CudaPreparedGemm,
        config: CudaGemmConfig,
        plan_mode: &str,
        input: &CudaDeviceBuffer,
        input_offset_bytes: u64,
        input_allocation_bytes: u64,
        input_tail: GeometryTail,
        input_tail_sha256: &str,
        weight: &CudaDeviceBuffer,
        weight_offset_bytes: u64,
        weight_allocation_bytes: u64,
        output: &mut CudaDeviceBuffer,
        output_offset_bytes: u64,
        output_allocation_bytes: u64,
        expected_prefix_le: &[u8],
        anchor: Option<(&str, usize)>,
        compute_capability: (u32, u32),
        label: &str,
    ) -> TestResult<GeometryObservation> {
        let metadata = plan.algorithm_metadata();
        if plan.config() != config {
            return Err(format!("{label} plan configuration differs").into());
        }
        validate_metadata(metadata, config, compute_capability, label)?;
        let mut workspace = if metadata.workspace_bytes() == 0 {
            None
        } else {
            Some(context.allocate_device_buffer(metadata.workspace_bytes())?)
        };
        let allocations_before = context.allocation_stats()?;
        execute_strict_with_offsets(
            plan,
            config,
            metadata,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            output,
            output_offset_bytes,
            workspace.as_mut(),
            stream,
        )?;
        let first_native = download_geometry_output(
            output,
            output_offset_bytes,
            usize::try_from(config.output_bytes())?,
            staging,
            stream,
        )?;
        execute_strict_with_offsets(
            plan,
            config,
            metadata,
            input,
            input_offset_bytes,
            weight,
            weight_offset_bytes,
            output,
            output_offset_bytes,
            workspace.as_mut(),
            stream,
        )?;
        let repeated_native = download_geometry_output(
            output,
            output_offset_bytes,
            usize::try_from(config.output_bytes())?,
            staging,
            stream,
        )?;
        if first_native != repeated_native {
            return Err(format!("{label} full output changed on repetition").into());
        }
        if context.allocation_stats()? != allocations_before {
            return Err(format!("{label} execution changed allocation accounting").into());
        }
        let first_le = bf16_native_to_le(&first_native)?;
        let prefix_len = geometry_prefix_len(usize::try_from(config.m())?)?;
        let prefix_le = first_le
            .get(..prefix_len)
            .ok_or("geometry output is shorter than its compared prefix")?
            .to_vec();
        if expected_prefix_le.len() != prefix_le.len() {
            return Err(format!("{label} HF comparison prefix byte count differs").into());
        }
        let prefix_metrics = metrics(expected_prefix_le, &prefix_le)?;
        let record = json!({
            "status": "executed",
            "plan_mode": plan_mode,
            "anchor": anchor.map(|(source, m)| json!({"source_case": source, "m": m})),
            "dimensions": {"m": config.m(), "n": config.n(), "k": config.k()},
            "input_tail": input_tail.label(),
            "input_tail_bf16_le_sha256": input_tail_sha256,
            "spans": {
                "required_offset_alignment_bytes": SPAN_OFFSET_BYTES,
                "input_offset_bytes": input_offset_bytes,
                "weight_offset_bytes": weight_offset_bytes,
                "output_offset_bytes": output_offset_bytes,
                "input_span_bytes": config.input_bytes(),
                "weight_span_bytes": config.weight_bytes(),
                "output_span_bytes": config.output_bytes(),
                "input_allocation_bytes": input_allocation_bytes,
                "weight_allocation_bytes": weight_allocation_bytes,
                "output_allocation_bytes": output_allocation_bytes,
            },
            "algorithm": geometry_metadata_json(metadata),
            "repeated_full_output_bf16_exact": true,
            "full_output_bf16_le_sha256": sha256_hex(&first_le),
            "allocation_accounting_unchanged": true,
            "hf_unbiased_q_prefix_comparison": prefix_metrics,
        });
        if let Some(workspace) = workspace {
            workspace.close()?;
        }
        Ok(GeometryObservation {
            record,
            prefix_output_le: prefix_le,
        })
    }

    fn case_bf16_exact(cases: &Map<String, Value>, label: &str) -> TestResult<Option<bool>> {
        let Some(record) = cases.get(label) else {
            return Ok(None);
        };
        if record.get("status").and_then(Value::as_str) != Some("executed") {
            return Ok(None);
        }
        record
            .get("hf_unbiased_q_prefix_comparison")
            .and_then(Value::as_object)
            .and_then(|metrics| metrics.get("bf16_exact"))
            .and_then(Value::as_bool)
            .map(Some)
            .ok_or_else(|| format!("{label} geometry record has no BF16 exactness").into())
    }

    fn geometry_hypothesis(
        cases: &Map<String, Value>,
        tail_data_invariant: bool,
        admitted_offset_invariant: bool,
        anchored_algorithm_identity_invariant: bool,
    ) -> TestResult<&'static str> {
        if !tail_data_invariant
            || !admitted_offset_invariant
            || !anchored_algorithm_identity_invariant
        {
            return Ok("inconclusive");
        }
        let f2048 = case_bf16_exact(cases, "F2048")?.unwrap_or(false);
        let f2051 = case_bf16_exact(cases, "F2051")?.unwrap_or(false);
        let exact_padded_fresh = ["F2052", "F2080", "F2176", "F2304"]
            .iter()
            .map(|label| case_bf16_exact(cases, label))
            .collect::<TestResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .any(|exact| exact);
        if f2048 && !f2051 && exact_padded_fresh {
            Ok("supported")
        } else if !f2048 && !f2051 && !exact_padded_fresh {
            Ok("refuted")
        } else {
            Ok("inconclusive")
        }
    }

    fn insert_observation(
        cases: &mut Map<String, Value>,
        exact_cases: &mut Vec<String>,
        label: &str,
        observation: GeometryObservation,
    ) -> TestResult<Vec<u8>> {
        let exact = observation
            .record
            .get("hf_unbiased_q_prefix_comparison")
            .and_then(Value::as_object)
            .and_then(|metrics| metrics.get("bf16_exact"))
            .and_then(Value::as_bool)
            .ok_or("geometry observation lacks BF16 exactness")?;
        if exact {
            exact_cases.push(label.to_owned());
        }
        cases.insert(label.to_owned(), observation.record);
        Ok(observation.prefix_output_le)
    }

    fn insert_anchored_not_supported(
        cases: &mut Map<String, Value>,
        case: AnchoredGeometryCase,
        anchor: CudaGemmAlgorithmMetadata,
        error: &dyn std::fmt::Display,
    ) {
        cases.insert(
            case.label.to_owned(),
            json!({
                "status": "not-supported",
                "plan_mode": "anchored",
                "anchor": {
                    "source_case": case.anchor_label,
                    "m": case.anchor_m,
                    "algorithm_identity": geometry_algorithm_identity_json(anchor),
                },
                "dimensions": {"m": case.m, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE},
                "algorithm_identity_matches_anchor": Value::Null,
                "heuristic_fallback_permitted": false,
                "prepare_error": error.to_string(),
            }),
        );
    }

    fn insert_anchored_identity_mismatch(
        cases: &mut Map<String, Value>,
        case: AnchoredGeometryCase,
        anchor: CudaGemmAlgorithmMetadata,
        child: CudaGemmAlgorithmMetadata,
    ) {
        cases.insert(
            case.label.to_owned(),
            json!({
                "status": "algorithm-identity-mismatch",
                "plan_mode": "anchored",
                "anchor": {
                    "source_case": case.anchor_label,
                    "m": case.anchor_m,
                    "algorithm_identity": geometry_algorithm_identity_json(anchor),
                },
                "dimensions": {"m": case.m, "n": QWEN3B_HIDDEN_SIZE, "k": QWEN3B_HIDDEN_SIZE},
                "algorithm": geometry_metadata_json(child),
                "algorithm_identity_matches_anchor": false,
                "heuristic_fallback_permitted": false,
            }),
        );
    }

    fn geometry_source_hashes(root: &Path) -> TestResult<Value> {
        let sources = [
            "crates/riley-runtime/tests/qwen3b_p2051_raw_q_geometry_gpu.rs",
            "crates/riley-runtime/tests/support/qwen3b_p2051_projection_p7_contract.rs",
            "crates/riley-runtime/tests/qwen3b_p2051_bias_epilogue_gpu.rs",
            "crates/riley-cuda/src/gemm.rs",
            "kernels/src/gemm.cu",
        ];
        let mut records = Map::new();
        for source in sources {
            let path = regular_file(&root.join(source), "geometry source")?;
            records.insert(
                source.to_owned(),
                json!({"path": source, "sha256": sha256_file(&path)?}),
            );
        }
        Ok(Value::Object(records))
    }

    fn run_raw_q_geometry(
        model: &LoadedModel,
        hf: &HfProjectionArtifact,
    ) -> TestResult<GeometryExecution> {
        let input_hf_le = hf
            .tensors
            .get("layer0.input_norm")
            .ok_or("P7 geometry input_norm tensor is missing")?;
        let shadow_hf_le = hf
            .tensors
            .get("layer0.q_proj.unbiased_linear")
            .ok_or("P7 geometry Q no-bias shadow tensor is missing")?;
        if input_hf_le.len() != geometry_input_len(CONTEXT_TOKEN_COUNT)?
            || shadow_hf_le.len() != geometry_output_len(CONTEXT_TOKEN_COUNT)?
        {
            return Err("P7 geometry sidecar tensor byte count differs".into());
        }
        let projection = PROJECTIONS
            .iter()
            .copied()
            .find(|projection| projection.label == "q_proj")
            .ok_or("P7 geometry Q projection contract is missing")?;
        let host = projection_host_tensors(model, projection)?;
        let normal_input_le =
            geometry_input_payload(input_hf_le, MAX_GEOMETRY_M, GeometryTail::ZeroPad)?;
        let repeat_tail_input_le = geometry_input_payload(
            input_hf_le,
            MAX_GEOMETRY_M,
            GeometryTail::RepeatLastInputRow,
        )?;
        let normal_input_native = bf16_le_to_native(&normal_input_le)?;
        let repeat_tail_input_native = bf16_le_to_native(&repeat_tail_input_le)?;
        let offset_input_le =
            geometry_input_payload(input_hf_le, CONTEXT_TOKEN_COUNT, GeometryTail::Prefix)?;
        let offset_input_native = bf16_le_to_native(&offset_input_le)?;

        let runtime = CudaRuntime::initialize()?;
        if runtime.device_count() == 0 {
            return Err("remote GPU runner has no CUDA device".into());
        }
        let gpu = runtime.device(0)?;
        let properties = gpu.properties().clone();
        if properties.compute_capability() != EXPECTED_P7_COMPUTE_CAPABILITY {
            return Err(format!(
                "P7 raw-Q geometry artifact is an SM89 qualification; current compute capability {:?} differs",
                properties.compute_capability()
            )
            .into());
        }
        let context = gpu.create_context()?;
        let mut stream = context.create_stream()?;
        let device = json!({
            "ordinal": properties.ordinal(),
            "name": properties.name(),
            "total_memory_bytes": properties.total_memory_bytes(),
            "compute_capability": properties.compute_capability(),
            "multiprocessor_count": properties.multiprocessor_count(),
            "driver_version": properties.driver_version(),
            "runtime_version": properties.runtime_version(),
        });

        let result = (|| -> TestResult<GeometryExecution> {
            let mut staging = context.allocate_pinned_host_buffer(UPLOAD_STAGING_BYTES)?;
            let mut normal_input =
                upload_payload(&context, &mut stream, &mut staging, &normal_input_native, 0)?;
            let normal_weight =
                upload_payload(&context, &mut stream, &mut staging, &host.weight, 0)?;
            let mut normal_output = context
                .allocate_device_buffer(u64::try_from(geometry_output_len(MAX_GEOMETRY_M)?)?)?;
            let offset_input = upload_payload(
                &context,
                &mut stream,
                &mut staging,
                &offset_input_native,
                SPAN_OFFSET_BYTES,
            )?;
            let offset_weight = upload_payload(
                &context,
                &mut stream,
                &mut staging,
                &host.weight,
                SPAN_OFFSET_BYTES,
            )?;
            let mut offset_output = context.allocate_device_buffer(
                SPAN_OFFSET_BYTES
                    .checked_add(u64::try_from(geometry_output_len(CONTEXT_TOKEN_COUNT)?)?)
                    .ok_or("geometry offset output byte count overflows")?,
            )?;
            let normal_input_allocation_bytes = normal_input.byte_len();
            let normal_weight_allocation_bytes = normal_weight.byte_len();
            let normal_output_allocation_bytes = normal_output.byte_len();
            let offset_input_allocation_bytes = offset_input.byte_len();
            let offset_weight_allocation_bytes = offset_weight.byte_len();
            let offset_output_allocation_bytes = offset_output.byte_len();

            let f2048 = FRESH_CASES[0];
            let f2051 = FRESH_CASES[1];
            let mut f2048_plan = context.prepare_gemm(geometry_config(f2048.m)?)?;
            let mut f2051_plan = context.prepare_gemm(geometry_config(f2051.m)?)?;
            let mut cases = Map::new();
            let mut exact_cases = Vec::new();
            let mut prefixes = BTreeMap::new();
            let mut anchored_algorithm_identity_invariant = true;

            for case in [f2048, f2051] {
                let config = geometry_config(case.m)?;
                let plan = if case.label == f2048.label {
                    &mut f2048_plan
                } else {
                    &mut f2051_plan
                };
                let expected_len = geometry_prefix_len(case.m)?;
                let expected = shadow_hf_le
                    .get(..expected_len)
                    .ok_or("P7 geometry HF Q prefix is shorter than expected")?;
                let payload = &normal_input_le[..geometry_input_len(case.m)?];
                let observation = run_geometry_plan(
                    &context,
                    &mut stream,
                    &mut staging,
                    plan,
                    config,
                    "fresh",
                    &normal_input,
                    0,
                    normal_input_allocation_bytes,
                    case.tail,
                    &geometry_tail_sha256(payload)?,
                    &normal_weight,
                    0,
                    normal_weight_allocation_bytes,
                    &mut normal_output,
                    0,
                    normal_output_allocation_bytes,
                    expected,
                    None,
                    properties.compute_capability(),
                    case.label,
                )?;
                prefixes.insert(
                    case.label.to_owned(),
                    insert_observation(&mut cases, &mut exact_cases, case.label, observation)?,
                );
            }

            for case in FRESH_CASES.into_iter().skip(2) {
                let config = geometry_config(case.m)?;
                let mut plan = context.prepare_gemm(config)?;
                let expected_len = geometry_prefix_len(case.m)?;
                let expected = shadow_hf_le
                    .get(..expected_len)
                    .ok_or("P7 geometry HF Q prefix is shorter than expected")?;
                let payload = &normal_input_le[..geometry_input_len(case.m)?];
                let observation = run_geometry_plan(
                    &context,
                    &mut stream,
                    &mut staging,
                    &mut plan,
                    config,
                    "fresh",
                    &normal_input,
                    0,
                    normal_input_allocation_bytes,
                    case.tail,
                    &geometry_tail_sha256(payload)?,
                    &normal_weight,
                    0,
                    normal_weight_allocation_bytes,
                    &mut normal_output,
                    0,
                    normal_output_allocation_bytes,
                    expected,
                    None,
                    properties.compute_capability(),
                    case.label,
                )?;
                prefixes.insert(
                    case.label.to_owned(),
                    insert_observation(&mut cases, &mut exact_cases, case.label, observation)?,
                );

                if case.label == "F2080" {
                    upload_existing_payload(
                        &mut normal_input,
                        &mut stream,
                        &mut staging,
                        &repeat_tail_input_native,
                        0,
                    )?;
                    let tail_observation = run_geometry_plan(
                        &context,
                        &mut stream,
                        &mut staging,
                        &mut plan,
                        config,
                        "fresh-tail-control",
                        &normal_input,
                        0,
                        normal_input_allocation_bytes,
                        GeometryTail::RepeatLastInputRow,
                        &geometry_tail_sha256(
                            &repeat_tail_input_le[..geometry_input_len(case.m)?],
                        )?,
                        &normal_weight,
                        0,
                        normal_weight_allocation_bytes,
                        &mut normal_output,
                        0,
                        normal_output_allocation_bytes,
                        expected,
                        None,
                        properties.compute_capability(),
                        "T2080",
                    )?;
                    prefixes.insert(
                        "T2080".to_owned(),
                        insert_observation(
                            &mut cases,
                            &mut exact_cases,
                            "T2080",
                            tail_observation,
                        )?,
                    );
                    upload_existing_payload(
                        &mut normal_input,
                        &mut stream,
                        &mut staging,
                        &normal_input_native,
                        0,
                    )?;
                }
                plan.close()?;
            }

            let offset_case = FreshGeometryCase {
                label: "O2051",
                m: CONTEXT_TOKEN_COUNT,
                tail: GeometryTail::Prefix,
                offset_bytes: SPAN_OFFSET_BYTES,
            };
            let offset_config = geometry_config(offset_case.m)?;
            let offset_observation = run_geometry_plan(
                &context,
                &mut stream,
                &mut staging,
                &mut f2051_plan,
                offset_config,
                "F2051-reused-offset-control",
                &offset_input,
                offset_case.offset_bytes,
                offset_input_allocation_bytes,
                offset_case.tail,
                &geometry_tail_sha256(&offset_input_le)?,
                &offset_weight,
                offset_case.offset_bytes,
                offset_weight_allocation_bytes,
                &mut offset_output,
                offset_case.offset_bytes,
                offset_output_allocation_bytes,
                shadow_hf_le,
                None,
                properties.compute_capability(),
                offset_case.label,
            )?;
            let prefix_matches_fresh_case = prefixes
                .get("F2051")
                .map(|baseline| baseline == &offset_observation.prefix_output_le);
            let mut offset_record = offset_observation.record;
            offset_record["reused_fresh_case"] = json!("F2051");
            offset_record["prefix_matches_fresh_case"] = json!(prefix_matches_fresh_case);
            let offset_observation = GeometryObservation {
                record: offset_record,
                prefix_output_le: offset_observation.prefix_output_le,
            };
            prefixes.insert(
                offset_case.label.to_owned(),
                insert_observation(
                    &mut cases,
                    &mut exact_cases,
                    offset_case.label,
                    offset_observation,
                )?,
            );

            for case in ANCHORED_CASES {
                let config = geometry_config(case.m)?;
                let anchor = if case.anchor_m == f2048.m {
                    &f2048_plan
                } else {
                    &f2051_plan
                };
                let anchor_metadata = anchor.algorithm_metadata();
                let mut plan = match context.prepare_gemm_anchored(config, anchor) {
                    Ok(plan) => plan,
                    Err(error) => {
                        insert_anchored_not_supported(&mut cases, case, anchor_metadata, &error);
                        continue;
                    }
                };
                let child_metadata = plan.algorithm_metadata();
                if !same_geometry_algorithm_identity(anchor_metadata, child_metadata) {
                    anchored_algorithm_identity_invariant = false;
                    insert_anchored_identity_mismatch(
                        &mut cases,
                        case,
                        anchor_metadata,
                        child_metadata,
                    );
                    plan.close()?;
                    continue;
                }
                let expected_len = geometry_prefix_len(case.m)?;
                let expected = shadow_hf_le
                    .get(..expected_len)
                    .ok_or("P7 geometry HF Q prefix is shorter than expected")?;
                let payload = &normal_input_le[..geometry_input_len(case.m)?];
                let tail = if case.m == CONTEXT_TOKEN_COUNT {
                    GeometryTail::Prefix
                } else {
                    GeometryTail::ZeroPad
                };
                let observation = run_geometry_plan(
                    &context,
                    &mut stream,
                    &mut staging,
                    &mut plan,
                    config,
                    "anchored",
                    &normal_input,
                    0,
                    normal_input_allocation_bytes,
                    tail,
                    &geometry_tail_sha256(payload)?,
                    &normal_weight,
                    0,
                    normal_weight_allocation_bytes,
                    &mut normal_output,
                    0,
                    normal_output_allocation_bytes,
                    expected,
                    Some((case.anchor_label, case.anchor_m)),
                    properties.compute_capability(),
                    case.label,
                )?;
                let fresh_label = format!("F{}", case.m);
                let prefix_matches_fresh = prefixes
                    .get(&fresh_label)
                    .map(|fresh| fresh == &observation.prefix_output_le);
                let mut record = observation.record;
                record["prefix_matches_fresh_case"] = json!(prefix_matches_fresh);
                record["anchor_algorithm_identity"] =
                    geometry_algorithm_identity_json(anchor_metadata);
                record["algorithm_identity_matches_anchor"] = json!(true);
                let anchored = GeometryObservation {
                    record,
                    prefix_output_le: observation.prefix_output_le,
                };
                prefixes.insert(
                    case.label.to_owned(),
                    insert_observation(&mut cases, &mut exact_cases, case.label, anchored)?,
                );
                plan.close()?;
            }

            let tail_data_invariant = prefixes
                .get("F2080")
                .zip(prefixes.get("T2080"))
                .is_some_and(|(fresh, control)| fresh == control);
            let admitted_offset_invariant = prefixes
                .get("F2051")
                .zip(prefixes.get("O2051"))
                .is_some_and(|(baseline, control)| baseline == control);
            let geometry_hypothesis = geometry_hypothesis(
                &cases,
                tail_data_invariant,
                admitted_offset_invariant,
                anchored_algorithm_identity_invariant,
            )?;

            f2048_plan.close()?;
            f2051_plan.close()?;
            normal_input.close()?;
            normal_weight.close()?;
            normal_output.close()?;
            offset_input.close()?;
            offset_weight.close()?;
            offset_output.close()?;
            staging.close()?;
            if !context.allocation_stats()?.is_zero() {
                return Err("P7 geometry discriminator left a CUDA allocation before close".into());
            }
            Ok(GeometryExecution {
                device,
                cases,
                exact_cases,
                tail_data_invariant,
                admitted_offset_invariant,
                anchored_algorithm_identity_invariant,
                geometry_hypothesis,
            })
        })();
        let cleanup = close_context(stream, context);
        match (result, cleanup) {
            (Ok(result), Ok(())) => Ok(result),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(run_error), Err(cleanup_error)) => Err(format!(
                "P7 raw-Q geometry discriminator failed: {run_error}; cleanup also failed: {cleanup_error}"
            )
            .into()),
        }
    }

    #[test]
    #[ignore = "remote-only Qwen2.5-3B P2051 raw-Q geometry discriminator"]
    fn qwen3b_p2051_raw_q_geometry_discriminator() -> TestResult {
        let output = required_path(GEOMETRY_OUTPUT_VARIABLE)?;
        validate_output_destination(&output)?;
        let workload = load_workload()?;
        let teacher = load_teacher_prefix()?;
        let hf = load_hf_projection_artifact(&teacher, &workload)?;
        if hf.manifest_sha256 != P7_MANIFEST_SHA256 || hf.sidecar_sha256 != P7_SIDECAR_SHA256 {
            return Err(
                "raw-Q geometry discriminator must consume the immutable P7 artifact".into(),
            );
        }
        let model = load_model(&hf)?;
        let execution = run_raw_q_geometry(&model, &hf)?;
        let root = repository_root()?;
        let anchored_algo_check = execution
            .cases
            .iter()
            .filter(|(label, _)| label.starts_with('A'))
            .map(|(label, record)| (label.clone(), record.clone()))
            .collect::<Map<_, _>>();
        let receipt = json!({
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "artifact_kind": GEOMETRY_ARTIFACT_KIND,
            "performance_claim_eligible": false,
            "created_at_unix_seconds": unix_seconds()?,
            "immutable_p7_projection_artifact": {
                "manifest_path": hf.manifest_path,
                "manifest_sha256": hf.manifest_sha256,
                "sidecar_path": hf.sidecar_path,
                "sidecar_sha256": hf.sidecar_sha256,
                "artifact_source_provenance_validated": true,
                "checkpoint_receipt_filename": hf.checkpoint_receipt_filename,
                "checkpoint_receipt_sha256": hf.checkpoint_receipt_sha256,
                "checkpoint_model_binding_validated": true,
            },
            "source_hashes": geometry_source_hashes(&root)?,
            "contract": {
                "trace_id": PROJECTION_TRACE_ID,
                "model_id": QWEN3B_MODEL_ID,
                "model_revision": QWEN3B_REVISION,
                "input_tensor": "layer0.input_norm",
                "hf_reference_tensor": "layer0.q_proj.unbiased_linear",
                "operator": "strict cuBLASLt BF16 raw-Q GEMM with F32 compute",
                "m_values": [2048, 2051, 2052, 2080, 2176, 2304],
                "tail_control": "F2080 explicit BF16-zero tail versus T2080 repeated-last-input-row tail",
                "offset_control": "O2051 input/weight/output BF16 spans begin at byte offset 256",
                "anchored_plans_fail_closed_without_heuristic_fallback": true,
                "python_in_hot_path": false,
                "serving_selector_changed": false,
                "vllm_comparison_eligible": false,
            },
            "device": execution.device,
            "summary": {
                "geometry_hypothesis": execution.geometry_hypothesis,
                "raw_q_prefix_exact_cases": execution.exact_cases,
                "tail_data_invariant": execution.tail_data_invariant,
                "admitted_offset_invariant": execution.admitted_offset_invariant,
                "anchored_algorithm_identity_invariant": execution.anchored_algorithm_identity_invariant,
                "full_forward_padded_m_required": true,
                "serving_selector_changed": false,
                "vllm_comparison_eligible": false,
            },
            "anchored_algo_check": anchored_algo_check,
            "cases": execution.cases,
        });
        write_artifact_exclusive(&output, &receipt)?;
        if !execution.tail_data_invariant || !execution.admitted_offset_invariant {
            return Err(
                "raw-Q geometry control changed prefix output; inspect receipt before heuristic conclusions"
                    .into(),
            );
        }
        Ok(())
    }
}
