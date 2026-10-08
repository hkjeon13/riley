#ifndef RILEY_GATE_E_GUARDIAN_CONTROL_ENVELOPE_V1_H
#define RILEY_GATE_E_GUARDIAN_CONTROL_ENVELOPE_V1_H

#include <stdbool.h>
#include <stdint.h>

enum {
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES = 32,
};

#define GATE_E_GUARDIAN_CONTROL_ENVELOPE_MAX_PID UINT32_C(2147483647)

enum gate_e_guardian_control_envelope_ancillary_claim {
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_INVALID = 0,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_EMPTY,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_PRESENT,
};

enum gate_e_guardian_control_envelope_reason {
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK = 0,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PID_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_STARTTIME_TICKS_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PIDFD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_UID_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_GID_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_SUPPLIED_CGROUP_NOT_NON_DELEGATED,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_SUPPLIED_CGROUP,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_DEV_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_INO_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_HELD_FD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ANCILLARY_CLAIM,
    GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_FDS_PRESENT,
};

/* Fixed-width normalized claims, never live kernel identifiers or FDs. */
struct gate_e_guardian_process_identity {
    uint32_t pid;
    uint64_t starttime_ticks;
    unsigned char pidfd_token[GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES];
    uint32_t uid;
    uint32_t gid;
};

struct gate_e_guardian_cgroup_identity {
    uint64_t st_dev;
    uint64_t st_ino;
    unsigned char held_fd_token[GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES];
};

struct gate_e_guardian_cgroup_claim {
    struct gate_e_guardian_cgroup_identity identity;
    bool non_delegated;
};

/*
 * Caller-owned expected active-session claims. Initialize and populate this
 * before matching. The setter requires an unprivileged worker and a declared
 * non-delegated cgroup; it neither authenticates either claim nor opens an FD.
 */
struct gate_e_guardian_control_envelope_binding {
    uint64_t initialized_state;
    struct gate_e_guardian_process_identity registered_worker;
    struct gate_e_guardian_cgroup_identity held_cgroup;
};

/*
 * Caller-owned normalized report. The ancillary value is a declaration only:
 * this library never receives, counts, closes, or inspects an actual FD.
 */
struct gate_e_guardian_control_envelope_report {
    struct gate_e_guardian_process_identity credentials;
    struct gate_e_guardian_cgroup_claim cgroup;
    enum gate_e_guardian_control_envelope_ancillary_claim ancillary;
};

void gate_e_guardian_control_envelope_binding_v1_init(
    struct gate_e_guardian_control_envelope_binding *binding
);

/*
 * Copy one expected worker/cgroup binding. Failure clears a valid initialized
 * binding so a stale active-session identity cannot be reused.
 */
enum gate_e_guardian_control_envelope_reason
gate_e_guardian_control_envelope_binding_v1_set(
    struct gate_e_guardian_control_envelope_binding *binding,
    const struct gate_e_guardian_process_identity *worker,
    const struct gate_e_guardian_cgroup_claim *held_cgroup
);

/*
 * Compare normalized caller-owned claims only. This does not parse a packet,
 * authenticate a sender, inspect a cgroup/pidfd, track phase, or transition
 * admission state. It must be paired with separate packet and transport work.
 */
enum gate_e_guardian_control_envelope_reason
gate_e_match_guardian_control_envelope_v1(
    const struct gate_e_guardian_control_envelope_binding *expected_binding,
    const struct gate_e_guardian_control_envelope_report *reported
);

#endif
