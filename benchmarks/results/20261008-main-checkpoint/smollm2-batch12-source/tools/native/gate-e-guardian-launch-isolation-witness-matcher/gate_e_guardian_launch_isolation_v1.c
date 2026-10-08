/*
 * Pure C11 matcher for normalized Gate E sealed-launch isolation claims.
 *
 * This library only compares caller-owned values. It never inspects a process,
 * descriptor table, argv, environment, capability set, or executable object.
 */

#include "gate_e_guardian_launch_isolation_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494C594C41554E);

static void clear_binding(struct gate_e_guardian_launch_isolation_binding *const binding) {
    memset(
        binding->expected_bootstrap_held_fd_token,
        0,
        sizeof(binding->expected_bootstrap_held_fd_token)
    );
    memset(
        binding->expected_core_held_fd_token,
        0,
        sizeof(binding->expected_core_held_fd_token)
    );
}

void gate_e_guardian_launch_isolation_binding_v1_init(
    struct gate_e_guardian_launch_isolation_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool nonzero_token(
    const unsigned char token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES]
) {
    unsigned char nonzero = 0U;

    if (token == NULL) {
        return false;
    }
    for (size_t index = 0U; index < GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES; ++index) {
        nonzero |= token[index];
    }
    return nonzero != 0U;
}

static bool valid_truth(
    const enum gate_e_guardian_launch_isolation_truth_claim claim
) {
    return claim == GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO ||
           claim == GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
}

static bool valid_expected_binding(
    const struct gate_e_guardian_launch_isolation_binding *const binding
) {
    return binding != NULL && binding->initialized_state == BINDING_INITIALIZED_STATE &&
           nonzero_token(binding->expected_bootstrap_held_fd_token) &&
           nonzero_token(binding->expected_core_held_fd_token);
}

enum gate_e_guardian_launch_isolation_reason
gate_e_guardian_launch_isolation_binding_v1_set(
    struct gate_e_guardian_launch_isolation_binding *const binding,
    const unsigned char bootstrap_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ],
    const unsigned char core_held_fd_token[
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES
    ]
) {
    struct gate_e_guardian_launch_isolation_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT;
    }
    if (!nonzero_token(bootstrap_held_fd_token) || !nonzero_token(core_held_fd_token)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_launch_isolation_binding_v1_init(&candidate);
    memcpy(
        candidate.expected_bootstrap_held_fd_token,
        bootstrap_held_fd_token,
        sizeof(candidate.expected_bootstrap_held_fd_token)
    );
    memcpy(
        candidate.expected_core_held_fd_token,
        core_held_fd_token,
        sizeof(candidate.expected_core_held_fd_token)
    );
    *binding = candidate;
    return GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK;
}

static bool valid_truth_fields(
    const struct gate_e_guardian_launch_isolation_report *const reported
) {
    return valid_truth(reported->bootstrap_fd_is_sealed) &&
           valid_truth(reported->bootstrap_fd_carries_lease) &&
           valid_truth(reported->bootstrap_fd_carries_cgroup_control) &&
           valid_truth(reported->core_fd_is_sealed) &&
           valid_truth(reported->core_fd_consumed_before_worker) &&
           valid_truth(reported->core_fd_inherited_by_worker) &&
           valid_truth(reported->lease_fd_inherited) &&
           valid_truth(reported->cgroup_control_fd_inherited) &&
           valid_truth(reported->no_new_privs);
}

enum gate_e_guardian_launch_isolation_reason
gate_e_match_guardian_launch_isolation_v1(
    const struct gate_e_guardian_launch_isolation_binding *const expected_binding,
    const struct gate_e_guardian_launch_isolation_report *const reported
) {
    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING;
    }
    if (reported == NULL) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT;
    }
    if (
        reported->argv != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_FUTURE_BOOTSTRAP_V1 &&
        reported->argv != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_OTHER
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGV_CLAIM;
    }
    if (reported->argv != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_FUTURE_BOOTSTRAP_V1) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_UNEXPECTED_ARGV;
    }
    if (
        reported->environment != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_EMPTY &&
        reported->environment != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_PRESENT
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ENVIRONMENT_CLAIM;
    }
    if (reported->environment != GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_EMPTY) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_NOT_EMPTY;
    }
    if (
        reported->bootstrap_inherited_fd_count !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT_MISMATCH;
    }
    if (
        reported->worker_inherited_fd_count !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT_MISMATCH;
    }
    if (
        reported->bootstrap_highest_inherited_fd !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_MAX_FD
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MAXIMUM_MISMATCH;
    }
    if (
        reported->worker_highest_inherited_fd !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_MAX_FD
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_MAXIMUM_MISMATCH;
    }
    if (
        reported->bootstrap_inherited_fd_mask !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MASK
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_SET_MISMATCH;
    }
    if (reported->worker_inherited_fd_mask != GATE_E_GUARDIAN_LAUNCH_ISOLATION_STDIO_FD_MASK) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_SET_MISMATCH;
    }
    if (reported->bootstrap_fd_number != GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NUMBER_MISMATCH;
    }
    if (
        memcmp(
            reported->bootstrap_fd_token,
            expected_binding->expected_bootstrap_held_fd_token,
            sizeof(reported->bootstrap_fd_token)
        ) != 0
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_TOKEN_MISMATCH;
    }
    if (reported->core_fd_number != GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NUMBER_MISMATCH;
    }
    if (
        memcmp(
            reported->core_fd_token,
            expected_binding->expected_core_held_fd_token,
            sizeof(reported->core_fd_token)
        ) != 0
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_TOKEN_MISMATCH;
    }
    if (!valid_truth_fields(reported)) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM;
    }
    if (reported->bootstrap_fd_is_sealed != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NOT_SEALED;
    }
    if (reported->bootstrap_fd_carries_lease != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_LEASE;
    }
    if (
        reported->bootstrap_fd_carries_cgroup_control !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_CGROUP_CONTROL;
    }
    if (reported->core_fd_is_sealed != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_SEALED;
    }
    if (
        reported->core_fd_consumed_before_worker !=
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_CONSUMED_BEFORE_WORKER;
    }
    if (reported->core_fd_inherited_by_worker != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_INHERITED_BY_WORKER;
    }
    if (reported->lease_fd_inherited != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_LEASE_FD_INHERITED;
    }
    if (reported->cgroup_control_fd_inherited != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_CGROUP_CONTROL_FD_INHERITED;
    }
    if (reported->no_new_privs != GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_NO_NEW_PRIVS_NOT_SET;
    }
    if (
        reported->capabilities != GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_EMPTY &&
        reported->capabilities != GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_PRESENT
    ) {
        return GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_CAPABILITY_CLAIM;
    }
    return reported->capabilities == GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_EMPTY
               ? GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK
               : GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_NOT_EMPTY;
}
