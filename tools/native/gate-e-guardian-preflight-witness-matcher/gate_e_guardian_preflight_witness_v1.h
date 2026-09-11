#ifndef RILEY_GATE_E_GUARDIAN_PREFLIGHT_WITNESS_V1_H
#define RILEY_GATE_E_GUARDIAN_PREFLIGHT_WITNESS_V1_H

#include <stdbool.h>
#include <stdint.h>

enum {
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_TOKEN_BYTES = 32,
};

#define GATE_E_GUARDIAN_PREFLIGHT_WITNESS_MAX_PID UINT32_C(2147483647)

enum gate_e_guardian_preflight_witness_population_claim {
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_POPULATION_INVALID = 0,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_POPULATION_EMPTY,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_POPULATION_PRESENT,
};

enum gate_e_guardian_preflight_witness_reason {
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK = 0,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_EXPECTED_BINDING,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_GUARDIAN,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_GUARDIAN_IDENTITY_MISMATCH,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_WARDEN,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_WARDEN_IDENTITY_MISMATCH,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_CONTROLLER,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_CONTROLLER_IDENTITY_MISMATCH,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_SUPPLIED_CGROUP,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_POPULATION_CLAIM,
    GATE_E_GUARDIAN_PREFLIGHT_WITNESS_CGROUP_NOT_EMPTY,
};

/* Fixed-width normalized claims, never live kernel identifiers or FDs. */
struct gate_e_guardian_preflight_witness_process_identity {
    uint32_t pid;
    uint64_t starttime_ticks;
    unsigned char pidfd_token[GATE_E_GUARDIAN_PREFLIGHT_WITNESS_TOKEN_BYTES];
    uint32_t uid;
    uint32_t gid;
};

struct gate_e_guardian_preflight_witness_cgroup_identity {
    uint64_t st_dev;
    uint64_t st_ino;
    unsigned char held_fd_token[GATE_E_GUARDIAN_PREFLIGHT_WITNESS_TOKEN_BYTES];
};

struct gate_e_guardian_preflight_witness_cgroup_claim {
    struct gate_e_guardian_preflight_witness_cgroup_identity identity;
    bool non_delegated;
};

/* Caller-owned active PREFLIGHT binding. */
struct gate_e_guardian_preflight_witness_binding {
    uint64_t initialized_state;
    struct gate_e_guardian_preflight_witness_process_identity expected_guardian;
    struct gate_e_guardian_preflight_witness_process_identity expected_warden;
    struct gate_e_guardian_preflight_witness_process_identity expected_pid1_controller;
};

/*
 * Caller-owned normalized observation. Population is a declaration only: this
 * library never receives or inspects an actual cgroup, pidfd, socket, or FD.
 */
struct gate_e_guardian_preflight_witness_report {
    struct gate_e_guardian_preflight_witness_process_identity guardian;
    struct gate_e_guardian_preflight_witness_process_identity warden;
    struct gate_e_guardian_preflight_witness_process_identity controller;
    struct gate_e_guardian_preflight_witness_cgroup_claim cgroup;
    enum gate_e_guardian_preflight_witness_population_claim population;
};

void gate_e_guardian_preflight_witness_binding_v1_init(
    struct gate_e_guardian_preflight_witness_binding *binding
);

/*
 * Copy the three expected root service identities. They must be pairwise
 * distinct; failure clears a valid initialized binding so stale active-session
 * claims cannot be reused.
 */
enum gate_e_guardian_preflight_witness_reason
gate_e_guardian_preflight_witness_binding_v1_set(
    struct gate_e_guardian_preflight_witness_binding *binding,
    const struct gate_e_guardian_preflight_witness_process_identity *expected_guardian,
    const struct gate_e_guardian_preflight_witness_process_identity *expected_warden,
    const struct gate_e_guardian_preflight_witness_process_identity *expected_pid1_controller
);

/*
 * Match a normalized preflight witness only. This neither acquires a cgroup
 * nor tracks phase/ledger state, changes admission, or establishes a lease.
 * A caller must pair it with independent acquisition and controller work.
 */
enum gate_e_guardian_preflight_witness_reason
gate_e_match_guardian_preflight_witness_v1(
    const struct gate_e_guardian_preflight_witness_binding *expected_binding,
    const struct gate_e_guardian_preflight_witness_report *reported
);

#endif
