#ifndef RILEY_GATE_E_GUARDIAN_CONTROL_PACKET_V1_H
#define RILEY_GATE_E_GUARDIAN_CONTROL_PACKET_V1_H

#include <stddef.h>
#include <stdint.h>

#define GATE_E_GUARDIAN_CONTROL_PACKET_SCHEMA_VERSION \
    "riley.rc3-gate-e-guardian-control.v1"

enum {
    GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES = 32,
};

#define GATE_E_GUARDIAN_CONTROL_PACKET_MAX_BYTES ((size_t)4U * 1024U)

enum gate_e_guardian_control_packet_kind {
    GATE_E_GUARDIAN_CONTROL_PACKET_KIND_INVALID = 0,
    GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
    GATE_E_GUARDIAN_CONTROL_PACKET_KIND_NO_ACTION_COMPLETE,
};

enum gate_e_guardian_control_packet_reason {
    GATE_E_GUARDIAN_CONTROL_PACKET_OK = 0,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_PACKET_BYTE_LENGTH,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON,
    GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_SCHEMA_VERSION,
    GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_KIND,
    GATE_E_GUARDIAN_CONTROL_PACKET_UNEXPECTED_KIND,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION,
    GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256,
    GATE_E_GUARDIAN_CONTROL_PACKET_ZERO_SHA256,
    GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH,
};

/*
 * Caller-owned active-session values. Initialize and populate this with the
 * setter before validation. Each digest is the binary form of one nonzero
 * lowercase SHA-256 session field; this library does not authenticate or
 * acquire the session that supplied them.
 */
struct gate_e_guardian_control_packet_binding {
    uint64_t initialized_state;
    uint64_t generation;
    unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char guardian_contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
};

void gate_e_guardian_control_packet_binding_v1_init(
    struct gate_e_guardian_control_packet_binding *binding
);

/*
 * Copy one active-session binding. On every failure a valid initialized
 * binding is cleared so a stale session cannot be reused.
 */
enum gate_e_guardian_control_packet_reason
gate_e_guardian_control_packet_binding_v1_set(
    struct gate_e_guardian_control_packet_binding *binding,
    uint64_t generation,
    const unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char guardian_contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES]
);

/*
 * Validate one complete raw canonical packet against the caller-provided
 * active session binding and required phase-specific kind. This is an
 * in-memory parser only: it neither receives from a transport nor validates
 * credentials, cgroups, ancillary FDs, a ledger, or any execution boundary.
 */
enum gate_e_guardian_control_packet_reason
gate_e_validate_guardian_control_packet_v1(
    const unsigned char *raw,
    size_t raw_length,
    enum gate_e_guardian_control_packet_kind expected_kind,
    const struct gate_e_guardian_control_packet_binding *expected_binding
);

#endif
