#ifndef RILEY_GATE_E_GUARDIAN_START_FENCE_V1_H
#define RILEY_GATE_E_GUARDIAN_START_FENCE_V1_H

#include <stdint.h>

enum {
    GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES = 32,
};

enum gate_e_guardian_start_fence_mode {
    GATE_E_GUARDIAN_START_FENCE_MODE_INVALID = 0,
    GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED,
    GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
};

enum gate_e_guardian_start_fence_reason {
    GATE_E_GUARDIAN_START_FENCE_OK = 0,
    GATE_E_GUARDIAN_START_FENCE_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE,
    GATE_E_GUARDIAN_START_FENCE_INVALID_CANDIDATE,
    GATE_E_GUARDIAN_START_FENCE_DURABLE_RECOVERY_REQUIRED,
    GATE_E_GUARDIAN_START_FENCE_GENERATION_REPLAYED,
};

/*
 * Caller-owned projection of the durable START fence. An unfenced state
 * represents no prior boot identity and a zero high-water generation. A
 * fenced state carries one opaque nonzero boot-id digest and its high-water
 * generation. This is never a ledger representation or authority to mutate it.
 */
struct gate_e_guardian_start_fence_binding {
    uint64_t initialized_state;
    enum gate_e_guardian_start_fence_mode mode;
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    uint64_t highest_generation;
};

/* Caller-owned projection from an already independently validated session. */
struct gate_e_guardian_start_fence_candidate {
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    uint64_t generation;
};

void gate_e_guardian_start_fence_binding_v1_init(
    struct gate_e_guardian_start_fence_binding *binding
);

/*
 * Copy one normalized durable-fence projection. Failure clears a valid
 * initialized binding so stale fencing claims cannot be reused.
 */
enum gate_e_guardian_start_fence_reason
gate_e_guardian_start_fence_binding_v1_set(
    struct gate_e_guardian_start_fence_binding *binding,
    enum gate_e_guardian_start_fence_mode mode,
    const unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES],
    uint64_t highest_generation
);

/*
 * Compare a normalized START candidate with an immutable caller-owned
 * projection only. This neither parses a session nor reads/writes a durable
 * ledger, changes phase/admission, or creates PID1/cgroup/FD authority.
 */
enum gate_e_guardian_start_fence_reason
gate_e_match_guardian_start_fence_v1(
    const struct gate_e_guardian_start_fence_binding *expected_binding,
    const struct gate_e_guardian_start_fence_candidate *candidate
);

#endif
