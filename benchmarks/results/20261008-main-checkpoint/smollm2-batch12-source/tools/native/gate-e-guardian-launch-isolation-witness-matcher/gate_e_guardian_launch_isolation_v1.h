#ifndef RILEY_GATE_E_GUARDIAN_LAUNCH_ISOLATION_V1_H
#define RILEY_GATE_E_GUARDIAN_LAUNCH_ISOLATION_V1_H

#include <stdint.h>

enum {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES = 32,
};

#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD UINT32_C(31)
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD UINT32_C(32)
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_STDIO_FD_MASK \
    ((UINT64_C(1) << 0) | (UINT64_C(1) << 1) | (UINT64_C(1) << 2))
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MASK \
    (GATE_E_GUARDIAN_LAUNCH_ISOLATION_STDIO_FD_MASK | \
     (UINT64_C(1) << GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD) | \
     (UINT64_C(1) << GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD))
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT UINT32_C(5)
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT UINT32_C(3)
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_MAX_FD \
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD
#define GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_MAX_FD UINT32_C(2)

enum gate_e_guardian_launch_isolation_argv_claim {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_INVALID = 0,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_FUTURE_BOOTSTRAP_V1,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_OTHER,
};

enum gate_e_guardian_launch_isolation_environment_claim {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_INVALID = 0,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_EMPTY,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_PRESENT,
};

enum gate_e_guardian_launch_isolation_truth_claim {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_INVALID = 0,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES,
};

enum gate_e_guardian_launch_isolation_capability_claim {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_INVALID = 0,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_EMPTY,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_PRESENT,
};

enum gate_e_guardian_launch_isolation_reason {
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK = 0,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGV_CLAIM,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_UNEXPECTED_ARGV,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ENVIRONMENT_CLAIM,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_NOT_EMPTY,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MAXIMUM_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_MAXIMUM_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_SET_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_SET_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NUMBER_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NUMBER_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_TOKEN_MISMATCH,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NOT_SEALED,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_LEASE,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_CGROUP_CONTROL,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_SEALED,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_CONSUMED_BEFORE_WORKER,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_INHERITED_BY_WORKER,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_LEASE_FD_INHERITED,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CGROUP_CONTROL_FD_INHERITED,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_NO_NEW_PRIVS_NOT_SET,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_CAPABILITY_CLAIM,
    GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_NOT_EMPTY,
};

/*
 * Caller-owned tokens for separately authenticated bootstrap and core objects.
 * They are not live descriptors, object handles, or authority to execute them.
 */
struct gate_e_guardian_launch_isolation_binding {
    uint64_t initialized_state;
    unsigned char expected_bootstrap_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ];
    unsigned char expected_core_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ];
};

/*
 * Caller-normalized launch-profile claims. Descriptor masks represent already
 * observed descriptor sets; this library never inspects an actual FD table.
 */
struct gate_e_guardian_launch_isolation_report {
    enum gate_e_guardian_launch_isolation_argv_claim argv;
    enum gate_e_guardian_launch_isolation_environment_claim environment;
    uint64_t bootstrap_inherited_fd_mask;
    uint64_t worker_inherited_fd_mask;
    uint32_t bootstrap_inherited_fd_count;
    uint32_t worker_inherited_fd_count;
    uint32_t bootstrap_highest_inherited_fd;
    uint32_t worker_highest_inherited_fd;
    uint32_t bootstrap_fd_number;
    unsigned char bootstrap_fd_token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    enum gate_e_guardian_launch_isolation_truth_claim bootstrap_fd_is_sealed;
    enum gate_e_guardian_launch_isolation_truth_claim bootstrap_fd_carries_lease;
    enum gate_e_guardian_launch_isolation_truth_claim bootstrap_fd_carries_cgroup_control;
    uint32_t core_fd_number;
    unsigned char core_fd_token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    enum gate_e_guardian_launch_isolation_truth_claim core_fd_is_sealed;
    enum gate_e_guardian_launch_isolation_truth_claim core_fd_consumed_before_worker;
    enum gate_e_guardian_launch_isolation_truth_claim core_fd_inherited_by_worker;
    enum gate_e_guardian_launch_isolation_truth_claim lease_fd_inherited;
    enum gate_e_guardian_launch_isolation_truth_claim cgroup_control_fd_inherited;
    enum gate_e_guardian_launch_isolation_truth_claim no_new_privs;
    enum gate_e_guardian_launch_isolation_capability_claim capabilities;
};

void gate_e_guardian_launch_isolation_binding_v1_init(
    struct gate_e_guardian_launch_isolation_binding *binding
);

/*
 * Copy the expected held-object tokens. Failure clears a valid initialized
 * binding so stale launch-profile claims cannot be reused.
 */
enum gate_e_guardian_launch_isolation_reason
gate_e_guardian_launch_isolation_binding_v1_set(
    struct gate_e_guardian_launch_isolation_binding *binding,
    const unsigned char bootstrap_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ],
    const unsigned char core_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ]
);

/*
 * Match normalized isolation claims only. It does not inspect argv/env/FDs,
 * query capabilities, call prctl or execveat, open a path, change a phase or
 * ledger, or create guardian/admission/release authority.
 */
enum gate_e_guardian_launch_isolation_reason
gate_e_match_guardian_launch_isolation_v1(
    const struct gate_e_guardian_launch_isolation_binding *expected_binding,
    const struct gate_e_guardian_launch_isolation_report *reported
);

#endif
