#ifndef RILEY_GATE_E_GUARDIAN_DRAIN_WITNESS_V1_H
#define RILEY_GATE_E_GUARDIAN_DRAIN_WITNESS_V1_H

#include <stdbool.h>
#include <stdint.h>

enum {
    GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES = 32,
};

#define GATE_E_GUARDIAN_DRAIN_WITNESS_MAX_PID UINT32_C(2147483647)

enum gate_e_guardian_drain_witness_population_claim {
    GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_INVALID = 0,
    GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_EMPTY,
    GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT,
};

enum gate_e_guardian_drain_witness_reason {
    GATE_E_GUARDIAN_DRAIN_WITNESS_OK = 0,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PID_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_STARTTIME_TICKS_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PIDFD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_UID_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_GID_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_SUPPLIED_CGROUP,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_DEV_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_INO_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_POPULATION_CLAIM,
    GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_STILL_POPULATED,
    GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_TERMINAL_TOKEN_COUNT,
    GATE_E_GUARDIAN_DRAIN_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH,
};

/* Fixed-width normalized claims, never live kernel identifiers or FDs. */
struct gate_e_guardian_drain_witness_process_identity {
    uint32_t pid;
    uint64_t starttime_ticks;
    unsigned char pidfd_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];
    uint32_t uid;
    uint32_t gid;
};

struct gate_e_guardian_drain_witness_cgroup_identity {
    uint64_t st_dev;
    uint64_t st_ino;
    unsigned char held_fd_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];
};

struct gate_e_guardian_drain_witness_cgroup_claim {
    struct gate_e_guardian_drain_witness_cgroup_identity identity;
    bool non_delegated;
};

/* Caller-owned active DRAINING binding. */
struct gate_e_guardian_drain_witness_binding {
    uint64_t initialized_state;
    struct gate_e_guardian_drain_witness_process_identity expected_pid1_controller;
    unsigned char registered_worker_terminal_pidfd_token[
        GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES
    ];
    struct gate_e_guardian_drain_witness_cgroup_identity held_cgroup;
};

/*
 * Caller-owned normalized observation. Its population and terminal-token
 * fields are declarations only: this library never receives or inspects an
 * actual cgroup, pidfd, socket message, or FD.
 */
struct gate_e_guardian_drain_witness_report {
    struct gate_e_guardian_drain_witness_process_identity controller;
    struct gate_e_guardian_drain_witness_cgroup_claim cgroup;
    enum gate_e_guardian_drain_witness_population_claim population;
    uint32_t terminal_pidfd_token_count;
    unsigned char terminal_pidfd_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];
};

void gate_e_guardian_drain_witness_binding_v1_init(
    struct gate_e_guardian_drain_witness_binding *binding
);

/*
 * Copy the expected PID1/root controller, worker terminal token, and held
 * non-delegated cgroup. Failure clears a valid initialized binding so stale
 * active-session claims cannot be reused.
 */
enum gate_e_guardian_drain_witness_reason
gate_e_guardian_drain_witness_binding_v1_set(
    struct gate_e_guardian_drain_witness_binding *binding,
    const struct gate_e_guardian_drain_witness_process_identity *expected_pid1_controller,
    const unsigned char registered_worker_terminal_pidfd_token[
        GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES
    ],
    const struct gate_e_guardian_drain_witness_cgroup_claim *held_cgroup
);

/*
 * Match a normalized empty-drain witness only. This does not inspect a live
 * object, track phase or a ledger, change admission, or authorize controller
 * release. A caller must pair it with independent transport and controller
 * enforcement.
 */
enum gate_e_guardian_drain_witness_reason
gate_e_match_guardian_drain_witness_v1(
    const struct gate_e_guardian_drain_witness_binding *expected_binding,
    const struct gate_e_guardian_drain_witness_report *reported
);

#endif
