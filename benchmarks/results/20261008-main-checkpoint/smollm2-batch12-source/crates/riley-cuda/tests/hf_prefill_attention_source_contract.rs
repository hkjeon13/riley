#![cfg(not(feature = "cuda"))]

#[test]
fn plan_create_rejects_a_poisoned_context_before_acquiring_ownership() {
    let native = include_str!("../../../kernels/src/attention_cublaslt.cu");
    let (_, create_and_after) = native
        .split_once("riley_cuda_hf_prefill_attention_plan_create(")
        .expect("HF attention create entry point must exist");
    let (create, _) = create_and_after
        .split_once("riley_cuda_hf_prefill_attention_plan_info(")
        .expect("HF attention info entry point must follow create");
    let poisoned = create
        .find("context->restoration_failed.load(std::memory_order_acquire)")
        .expect("create must reject an already-poisoned context");
    let allocate = create
        .find("std::calloc")
        .expect("create must allocate an owning plan");
    let retain = create
        .find("retain_child(context)")
        .expect("create must retain the context child lease");

    assert!(poisoned < allocate);
    assert!(poisoned < retain);

    let prepare_attempted = create
        .find("bool prepare_attempted = false")
        .expect("create must distinguish entry rejection from preparation");
    let safe_entry_rejection = create
        .find("entry_rejected_without_context_change")
        .expect("create must identify side-effect-free entry rejection");
    let safe_cleanup = create
        .find("if (entry_rejected_without_context_change)")
        .expect("create must clean up the host-only plan after entry rejection");
    let restored_guard = create
        .find("const bool restored")
        .expect("ambiguous context restoration must retain ownership");

    assert!(retain < prepare_attempted);
    assert!(prepare_attempted < safe_entry_rejection);
    assert!(safe_entry_rejection < safe_cleanup);
    assert!(safe_cleanup < restored_guard);
}

#[test]
fn softmax_dispatch_uses_the_reviewed_pytorch_f32_boundaries() {
    let native = include_str!("../../../kernels/src/attention_cublaslt.cu");
    assert!(native.contains("case 10:"));
    assert!(native.contains("case 11:"));
    assert!(!native.contains("case 2:\n      hf_regular_softmax_kernel<2>"));
    assert!(native.contains("case 3:\n      hf_regular_softmax_kernel<3>"));
}

#[test]
fn native_plan_fail_closes_on_the_generated_environment_and_algorithm_allowlist() {
    let native = include_str!("../../../kernels/src/attention_cublaslt.cu");
    let allowlist = include_str!("../../../kernels/src/hf_eager_algorithm_allowlist.inc");

    assert!(allowlist.contains("kReviewedRuntimeVersion = 12080"));
    assert!(allowlist.contains("kReviewedCublasLtVersion = 120804"));
    assert!(allowlist.contains("kReviewedAttentionAlgorithms"));
    assert!(allowlist.contains("kReviewedTokenClasses"));
    for field in [
        "qk_algorithm_id",
        "qk_tile_id",
        "qk_stages_id",
        "qk_cta_swizzling",
        "qk_custom_option",
        "qk_workspace_bytes",
        "qk_numerical_implementation_flags",
        "av_algorithm_id",
        "av_tile_id",
        "av_stages_id",
        "av_cta_swizzling",
        "av_custom_option",
        "av_workspace_bytes",
        "av_numerical_implementation_flags",
    ] {
        assert!(native.contains(field), "native validator omits {field}");
    }
    assert!(native.contains("actual.qk_split_k == 1"));
    assert!(native.contains("actual.av_split_k == 1"));
    assert!(native.contains("validate_reviewed_plan_provenance(plan, error)"));
}
