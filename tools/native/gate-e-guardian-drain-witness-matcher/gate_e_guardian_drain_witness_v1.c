/*
 * Pure C11 matcher for normalized Gate E empty-drain witness claims.
 *
 * This library compares caller-owned typed values only. It does not receive a
 * socket message, inspect a live cgroup/pidfd/FD, parse JSON, or change state.
 */

#include "gate_e_guardian_drain_witness_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494c455944574d);

static void clear_binding(struct gate_e_guardian_drain_witness_binding *const binding) {
    memset(&binding->expected_pid1_controller, 0, sizeof(binding->expected_pid1_controller));
    memset(
        binding->registered_worker_terminal_pidfd_token,
        0,
        sizeof(binding->registered_worker_terminal_pidfd_token)
    );
    memset(&binding->held_cgroup, 0, sizeof(binding->held_cgroup));
}

void gate_e_guardian_drain_witness_binding_v1_init(
    struct gate_e_guardian_drain_witness_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool nonzero_token(
    const unsigned char token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES]
) {
    unsigned char nonzero = 0U;

    if (token == NULL) {
        return false;
    }
    for (size_t index = 0U; index < GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES; ++index) {
        nonzero |= token[index];
    }
    return nonzero != 0U;
}

static bool valid_process_identity(
    const struct gate_e_guardian_drain_witness_process_identity *const identity
) {
    return identity != NULL && identity->pid != 0U &&
           identity->pid <= GATE_E_GUARDIAN_DRAIN_WITNESS_MAX_PID &&
           identity->starttime_ticks != 0U && nonzero_token(identity->pidfd_token);
}

static bool valid_pid1_root_controller(
    const struct gate_e_guardian_drain_witness_process_identity *const identity
) {
    return valid_process_identity(identity) && identity->pid == UINT32_C(1) &&
           identity->uid == 0U && identity->gid == 0U;
}

static bool valid_cgroup_identity(
    const struct gate_e_guardian_drain_witness_cgroup_identity *const identity
) {
    return identity != NULL && identity->st_dev != 0U && identity->st_ino != 0U &&
           nonzero_token(identity->held_fd_token);
}

static bool valid_expected_binding(
    const struct gate_e_guardian_drain_witness_binding *const binding
) {
    return binding != NULL && binding->initialized_state == BINDING_INITIALIZED_STATE &&
           valid_pid1_root_controller(&binding->expected_pid1_controller) &&
           nonzero_token(binding->registered_worker_terminal_pidfd_token) &&
           valid_cgroup_identity(&binding->held_cgroup);
}

enum gate_e_guardian_drain_witness_reason
gate_e_guardian_drain_witness_binding_v1_set(
    struct gate_e_guardian_drain_witness_binding *const binding,
    const struct gate_e_guardian_drain_witness_process_identity *const expected_pid1_controller,
    const unsigned char registered_worker_terminal_pidfd_token[
        GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES
    ],
    const struct gate_e_guardian_drain_witness_cgroup_claim *const held_cgroup
) {
    struct gate_e_guardian_drain_witness_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT;
    }
    if (!valid_pid1_root_controller(expected_pid1_controller) ||
        !nonzero_token(registered_worker_terminal_pidfd_token) || held_cgroup == NULL ||
        !held_cgroup->non_delegated || !valid_cgroup_identity(&held_cgroup->identity)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_drain_witness_binding_v1_init(&candidate);
    candidate.expected_pid1_controller = *expected_pid1_controller;
    memcpy(
        candidate.registered_worker_terminal_pidfd_token,
        registered_worker_terminal_pidfd_token,
        sizeof(candidate.registered_worker_terminal_pidfd_token)
    );
    candidate.held_cgroup = held_cgroup->identity;
    *binding = candidate;
    return GATE_E_GUARDIAN_DRAIN_WITNESS_OK;
}

static enum gate_e_guardian_drain_witness_reason match_controller(
    const struct gate_e_guardian_drain_witness_process_identity *const expected,
    const struct gate_e_guardian_drain_witness_process_identity *const reported
) {
    if (!valid_process_identity(reported)) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER;
    }
    if (reported->pid != expected->pid) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PID_MISMATCH;
    }
    if (reported->starttime_ticks != expected->starttime_ticks) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_STARTTIME_TICKS_MISMATCH;
    }
    if (memcmp(reported->pidfd_token, expected->pidfd_token, sizeof(reported->pidfd_token)) != 0) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PIDFD_TOKEN_MISMATCH;
    }
    if (reported->uid != expected->uid) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_UID_MISMATCH;
    }
    return reported->gid == expected->gid ? GATE_E_GUARDIAN_DRAIN_WITNESS_OK
                                          : GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_GID_MISMATCH;
}

static enum gate_e_guardian_drain_witness_reason match_cgroup(
    const struct gate_e_guardian_drain_witness_cgroup_identity *const expected,
    const struct gate_e_guardian_drain_witness_cgroup_claim *const reported
) {
    if (!reported->non_delegated) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED;
    }
    if (!valid_cgroup_identity(&reported->identity)) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_SUPPLIED_CGROUP;
    }
    if (reported->identity.st_dev != expected->st_dev) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_DEV_MISMATCH;
    }
    if (reported->identity.st_ino != expected->st_ino) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_INO_MISMATCH;
    }
    return memcmp(
               reported->identity.held_fd_token,
               expected->held_fd_token,
               sizeof(reported->identity.held_fd_token)
           ) == 0
               ? GATE_E_GUARDIAN_DRAIN_WITNESS_OK
               : GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH;
}

enum gate_e_guardian_drain_witness_reason
gate_e_match_guardian_drain_witness_v1(
    const struct gate_e_guardian_drain_witness_binding *const expected_binding,
    const struct gate_e_guardian_drain_witness_report *const reported
) {
    enum gate_e_guardian_drain_witness_reason result;

    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING;
    }
    if (reported == NULL) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT;
    }
    result = match_controller(&expected_binding->expected_pid1_controller, &reported->controller);
    if (result != GATE_E_GUARDIAN_DRAIN_WITNESS_OK) {
        return result;
    }
    result = match_cgroup(&expected_binding->held_cgroup, &reported->cgroup);
    if (result != GATE_E_GUARDIAN_DRAIN_WITNESS_OK) {
        return result;
    }
    if (reported->population == GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_STILL_POPULATED;
    }
    if (reported->population != GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_EMPTY) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_POPULATION_CLAIM;
    }
    if (reported->terminal_pidfd_token_count != UINT32_C(1)) {
        return GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_TERMINAL_TOKEN_COUNT;
    }
    return memcmp(
               reported->terminal_pidfd_token,
               expected_binding->registered_worker_terminal_pidfd_token,
               sizeof(reported->terminal_pidfd_token)
           ) == 0
               ? GATE_E_GUARDIAN_DRAIN_WITNESS_OK
               : GATE_E_GUARDIAN_DRAIN_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH;
}
