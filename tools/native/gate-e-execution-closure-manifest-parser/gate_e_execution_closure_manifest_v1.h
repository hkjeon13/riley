#ifndef RILEY_GATE_E_EXECUTION_CLOSURE_MANIFEST_V1_H
#define RILEY_GATE_E_EXECUTION_CLOSURE_MANIFEST_V1_H

#include <stddef.h>
#include <stdint.h>

#define GATE_E_EXECUTION_CLOSURE_SCHEMA_VERSION \
    "riley.rc3-gate-e-execution-closure-manifest.v1"

enum {
    GATE_E_EXECUTION_CLOSURE_SHA256_BYTES = 32,
    GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES = 128,
};

#define GATE_E_EXECUTION_CLOSURE_MAX_MANIFEST_BYTES ((size_t)64U * 1024U)
#define GATE_E_EXECUTION_CLOSURE_MAX_AUDIT_PATH_BYTES ((size_t)512U)
#define GATE_E_EXECUTION_CLOSURE_MAX_LEAF_BYTES UINT64_C(536870912)
#define GATE_E_EXECUTION_CLOSURE_MAX_CLOSURE_BYTES UINT64_C(2147483648)

enum gate_e_execution_closure_reason {
    GATE_E_EXECUTION_CLOSURE_OK = 0,
    GATE_E_EXECUTION_CLOSURE_INVALID_ARGUMENT,
    GATE_E_EXECUTION_CLOSURE_INVALID_MANIFEST_BYTE_LENGTH,
    GATE_E_EXECUTION_CLOSURE_INVALID_TERMINAL_NEWLINE,
    GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON,
    GATE_E_EXECUTION_CLOSURE_UNSUPPORTED_SCHEMA_VERSION,
    GATE_E_EXECUTION_CLOSURE_INVALID_AUDIT_PATH,
    GATE_E_EXECUTION_CLOSURE_INVALID_BYTE_LENGTH,
    GATE_E_EXECUTION_CLOSURE_INVALID_SHA256,
    GATE_E_EXECUTION_CLOSURE_ZERO_SHA256,
    GATE_E_EXECUTION_CLOSURE_INVALID_RUNTIME_LEAVES,
    GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAVES_NOT_STRICTLY_SORTED,
    GATE_E_EXECUTION_CLOSURE_DUPLICATE_AUDIT_PATH,
    GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAF_BUDGET_EXCEEDED,
    GATE_E_EXECUTION_CLOSURE_CLOSURE_BYTE_BUDGET_EXCEEDED,
};

struct gate_e_execution_closure_leaf {
    unsigned char audit_path[GATE_E_EXECUTION_CLOSURE_MAX_AUDIT_PATH_BYTES + 1];
    size_t audit_path_length;
    uint64_t byte_length;
    unsigned char sha256[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
};

struct gate_e_execution_closure_manifest {
    uint64_t initialized_state;
    struct gate_e_execution_closure_leaf dynamic_loader;
    struct gate_e_execution_closure_leaf interpreter;
    size_t runtime_leaf_count;
    struct gate_e_execution_closure_leaf
        runtime_leaves[GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES];
    unsigned char runtime_closure_sha256[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
};

/*
 * Initialize an output before its first parse. This library reads raw bytes
 * only; it does not open or inspect any declared audit path.
 */
void gate_e_execution_closure_manifest_v1_init(
    struct gate_e_execution_closure_manifest *output
);

/*
 * Parse exactly one canonical, newline-terminated execution-closure manifest.
 * On every input failure, a valid initialized output is cleared so a stale
 * trusted result cannot be reused. The returned raw SHA-256 covers the exact
 * supplied manifest bytes, including the terminal newline.
 */
enum gate_e_execution_closure_reason
gate_e_parse_execution_closure_manifest_v1(
    const unsigned char *raw,
    size_t raw_length,
    struct gate_e_execution_closure_manifest *output
);

#endif
