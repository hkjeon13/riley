use riley_model::{PatternId, SEMANTIC_PATTERN_SCHEMA_VERSION, SemanticPattern};

#[test]
fn public_v1_schema_has_fixed_closed_ids() {
    assert_eq!(SEMANTIC_PATTERN_SCHEMA_VERSION, 1);
    assert_eq!(
        SemanticPattern::ALL,
        [
            SemanticPattern::Norm,
            SemanticPattern::AttentionPrepare,
            SemanticPattern::PrefillAttention,
            SemanticPattern::DecodeAttention,
            SemanticPattern::MlpSwiGlu,
            SemanticPattern::ResidualNorm,
            SemanticPattern::LmHead,
            SemanticPattern::GreedyOutput,
        ]
    );

    let expected: [(SemanticPattern, u16); 8] = [
        (SemanticPattern::Norm, 1),
        (SemanticPattern::AttentionPrepare, 2),
        (SemanticPattern::PrefillAttention, 3),
        (SemanticPattern::DecodeAttention, 4),
        (SemanticPattern::MlpSwiGlu, 5),
        (SemanticPattern::ResidualNorm, 6),
        (SemanticPattern::LmHead, 7),
        (SemanticPattern::GreedyOutput, 8),
    ];
    for (pattern, numeric_id) in expected {
        let id = PatternId::from_u16(numeric_id).expect("known schema-v1 ID");
        assert_eq!(id.as_u16(), numeric_id);
        assert_eq!(SemanticPattern::from_id(id), Some(pattern));
        assert_eq!(pattern.id(), id);
    }
}

#[test]
fn unknown_and_reserved_numeric_values_fail_closed() {
    for unknown in [0, 9, u16::MAX] {
        assert_eq!(PatternId::from_u16(unknown), None);
    }
}

#[test]
fn semantic_schema_does_not_depend_on_runtime_implementation_layers() {
    let pattern_source = include_str!("../src/pattern.rs");
    let model_manifest = include_str!("../Cargo.toml");

    for forbidden_dependency in ["riley_runtime", "riley_cuda", "riley-runtime", "riley-cuda"] {
        assert!(
            !pattern_source.contains(forbidden_dependency),
            "pattern schema must not import {forbidden_dependency}"
        );
        assert!(
            !model_manifest.contains(forbidden_dependency),
            "model crate must not depend on {forbidden_dependency}"
        );
    }
}
