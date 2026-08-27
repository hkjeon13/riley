//! PR12 source-level guard against model-family branches in CUDA hot paths.

const HOT_PATHS: [(&str, &str); 3] = [
    ("llama/forward.rs", include_str!("../src/llama/forward.rs")),
    ("llama/decode.rs", include_str!("../src/llama/decode.rs")),
    (
        "llama/generation.rs",
        include_str!("../src/llama/generation.rs"),
    ),
];

#[test]
fn hot_execution_paths_do_not_dispatch_on_qwen_identity() {
    for (path, source) in HOT_PATHS {
        let lowercase = source.to_ascii_lowercase();
        for forbidden in [
            "qwen",
            "model_name",
            "source_architecture",
            "modelfamily",
            "family()",
            "architecture()",
        ] {
            assert!(
                !lowercase.contains(forbidden),
                "{path} contains forbidden hot-path family discriminator {forbidden:?}"
            );
        }
    }
}
