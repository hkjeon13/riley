/*
 * Pure C11 matcher for normalized Gate E controller-restart witness claims.
 *
 * This library compares caller-owned typed values only. It does not receive a
 * socket message, inspect a live cgroup/pidfd/FD, parse JSON, or change state.
 */

#include "gate_e_guardian_controller_restart_witness_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494c4552435757);

static void clear_binding(
    struct gate_e_guardian_controller_restart_witness_binding *const binding
) {
    memset(&binding->registered_guardian, 0, sizeof(binding->registered_guardian));
    memset(&binding->registered_warden, 0, sizeof(binding->registered_warden));
    memset(&binding->registered_worker, 0, sizeof(binding->registered_worker));
    memset(&binding->held_cgroup, 0, sizeof(binding->held_cgroup));
}

void gate_e_guardian_controller_restart_witness_binding_v1_init(
    struct gate_e_guardian_controller_restart_witness_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool nonzero_token(
    const unsigned char token[GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES]
) {
    unsigned char nonzero = 0U;

    if (token == NULL) {
        return false;
    }
    for (
        size_t index = 0U;
        index < GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES;
        ++index
    ) {
        nonzero |= token[index];
    }
    return nonzero != 0U;
}

static bool valid_process_identity(
    const struct gate_e_guardian_controller_restart_witness_process_identity *const identity,
    const bool require_unprivileged
) {
    return identity != NULL && identity->pid != 0U &&
           identity->pid <= GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_MAX_PID &&
           identity->starttime_ticks != 0U && nonzero_token(identity->pidfd_token) &&
           (!require_unprivileged || (identity->uid != 0U && identity->gid != 0U));
}

static bool valid_root_service_identity(
    const struct gate_e_guardian_controller_restart_witness_process_identity *const identity
) {
    return valid_process_identity(identity, false) && identity->uid == 0U && identity->gid == 0U;
}

static bool valid_pid1_controller(
    const struct gate_e_guardian_controller_restart_witness_process_identity *const identity
) {
    return valid_root_service_identity(identity) && identity->pid == UINT32_C(1);
}

static bool identities_equal(
    const struct gate_e_guardian_controller_restart_witness_process_identity *const left,
    const struct gate_e_guardian_controller_restart_witness_process_identity *const right
) {
    return left->pid == right->pid && left->starttime_ticks == right->starttime_ticks &&
           left->uid == right->uid && left->gid == right->gid &&
           memcmp(left->pidfd_token, right->pidfd_token, sizeof(left->pidfd_token)) == 0;
}

static bool valid_cgroup_identity(
    const struct gate_e_guardian_controller_restart_witness_cgroup_identity *const identity
) {
    return identity != NULL && identity->st_dev != 0U && identity->st_ino != 0U &&
           nonzero_token(identity->held_fd_token);
}

static bool valid_expected_binding(
    const struct gate_e_guardian_controller_restart_witness_binding *const binding
) {
    return binding != NULL && binding->initialized_state == BINDING_INITIALIZED_STATE &&
           valid_root_service_identity(&binding->registered_guardian) &&
           valid_root_service_identity(&binding->registered_warden) &&
           valid_process_identity(&binding->registered_worker, true) &&
           valid_cgroup_identity(&binding->held_cgroup) &&
           !identities_equal(&binding->registered_guardian, &binding->registered_warden) &&
           !identities_equal(&binding->registered_guardian, &binding->registered_worker) &&
           !identities_equal(&binding->registered_warden, &binding->registered_worker);
}

enum gate_e_guardian_controller_restart_witness_reason
gate_e_guardian_controller_restart_witness_binding_v1_set(
    struct gate_e_guardian_controller_restart_witness_binding *const binding,
    const struct gate_e_guardian_controller_restart_witness_process_identity *const registered_guardian,
    const struct gate_e_guardian_controller_restart_witness_process_identity *const registered_warden,
    const struct gate_e_guardian_controller_restart_witness_process_identity *const registered_worker,
    const struct gate_e_guardian_controller_restart_witness_cgroup_claim *const held_cgroup
) {
    struct gate_e_guardian_controller_restart_witness_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_ARGUMENT;
    }
    if (!valid_root_service_identity(registered_guardian) ||
        !valid_root_service_identity(registered_warden) ||
        !valid_process_identity(registered_worker, true) || held_cgroup == NULL ||
        !held_cgroup->non_delegated || !valid_cgroup_identity(&held_cgroup->identity) ||
        identities_equal(registered_guardian, registered_warden) ||
        identities_equal(registered_guardian, registered_worker) ||
        identities_equal(registered_warden, registered_worker)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_controller_restart_witness_binding_v1_init(&candidate);
    candidate.registered_guardian = *registered_guardian;
    candidate.registered_warden = *registered_warden;
    candidate.registered_worker = *registered_worker;
    candidate.held_cgroup = held_cgroup->identity;
    *binding = candidate;
    return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK;
}

static enum gate_e_guardian_controller_restart_witness_reason match_cgroup(
    const struct gate_e_guardian_controller_restart_witness_cgroup_identity *const expected,
    const struct gate_e_guardian_controller_restart_witness_cgroup_claim *const reported
) {
    if (!reported->non_delegated) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED;
    }
    if (!valid_cgroup_identity(&reported->identity)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_SUPPLIED_CGROUP;
    }
    if (reported->identity.st_dev != expected->st_dev) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_ST_DEV_MISMATCH;
    }
    if (reported->identity.st_ino != expected->st_ino) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_ST_INO_MISMATCH;
    }
    return memcmp(
               reported->identity.held_fd_token,
               expected->held_fd_token,
               sizeof(reported->identity.held_fd_token)
           ) == 0
               ? GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
               : GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH;
}

enum gate_e_guardian_controller_restart_witness_reason
gate_e_match_guardian_controller_restart_witness_v1(
    const struct gate_e_guardian_controller_restart_witness_binding *const expected_binding,
    const struct gate_e_guardian_controller_restart_witness_report *const reported
) {
    enum gate_e_guardian_controller_restart_witness_reason result;

    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING;
    }
    if (reported == NULL) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_ARGUMENT;
    }
    if (!valid_pid1_controller(&reported->new_pid1_controller)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER;
    }
    if (identities_equal(&reported->new_pid1_controller, &expected_binding->registered_guardian)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CONTROLLER_IS_GUARDIAN;
    }
    if (identities_equal(&reported->new_pid1_controller, &expected_binding->registered_warden)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CONTROLLER_IS_WARDEN;
    }
    if (identities_equal(&reported->new_pid1_controller, &expected_binding->registered_worker)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CONTROLLER_IS_WORKER;
    }
    result = match_cgroup(&expected_binding->held_cgroup, &reported->cgroup);
    if (result != GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK) {
        return result;
    }
    if (reported->population == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_PRESENT) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_STILL_POPULATED;
    }
    if (reported->population != GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_EMPTY) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_POPULATION_CLAIM;
    }
    if (reported->terminal_pidfd_token_count != UINT32_C(1)) {
        return GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_TERMINAL_TOKEN_COUNT;
    }
    return memcmp(
               reported->terminal_pidfd_token,
               expected_binding->registered_worker.pidfd_token,
               sizeof(reported->terminal_pidfd_token)
           ) == 0
               ? GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
               : GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH;
}
