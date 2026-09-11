/*
 * Pure C11 matcher for normalized Gate E preflight witness claims.
 *
 * This library compares caller-owned typed values only. It does not receive a
 * socket message, inspect a live cgroup/pidfd/FD, parse JSON, or change state.
 */

#include "gate_e_guardian_preflight_witness_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494c4559504657);

static void clear_binding(struct gate_e_guardian_preflight_witness_binding *const binding) {
    memset(&binding->expected_guardian, 0, sizeof(binding->expected_guardian));
    memset(&binding->expected_warden, 0, sizeof(binding->expected_warden));
    memset(&binding->expected_pid1_controller, 0, sizeof(binding->expected_pid1_controller));
}

void gate_e_guardian_preflight_witness_binding_v1_init(
    struct gate_e_guardian_preflight_witness_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool nonzero_token(
    const unsigned char token[GATE_E_GUARDIAN_PREFLIGHT_WITNESS_TOKEN_BYTES]
) {
    unsigned char nonzero = 0U;

    if (token == NULL) {
        return false;
    }
    for (size_t index = 0U; index < GATE_E_GUARDIAN_PREFLIGHT_WITNESS_TOKEN_BYTES; ++index) {
        nonzero |= token[index];
    }
    return nonzero != 0U;
}

static bool valid_process_identity(
    const struct gate_e_guardian_preflight_witness_process_identity *const identity
) {
    return identity != NULL && identity->pid != 0U &&
           identity->pid <= GATE_E_GUARDIAN_PREFLIGHT_WITNESS_MAX_PID &&
           identity->starttime_ticks != 0U && nonzero_token(identity->pidfd_token);
}

static bool valid_root_service_identity(
    const struct gate_e_guardian_preflight_witness_process_identity *const identity
) {
    return valid_process_identity(identity) && identity->uid == 0U && identity->gid == 0U;
}

static bool valid_pid1_controller(
    const struct gate_e_guardian_preflight_witness_process_identity *const identity
) {
    return valid_root_service_identity(identity) && identity->pid == UINT32_C(1);
}

static bool identities_equal(
    const struct gate_e_guardian_preflight_witness_process_identity *const left,
    const struct gate_e_guardian_preflight_witness_process_identity *const right
) {
    return left->pid == right->pid && left->starttime_ticks == right->starttime_ticks &&
           left->uid == right->uid && left->gid == right->gid &&
           memcmp(left->pidfd_token, right->pidfd_token, sizeof(left->pidfd_token)) == 0;
}

static bool valid_cgroup_identity(
    const struct gate_e_guardian_preflight_witness_cgroup_identity *const identity
) {
    return identity != NULL && identity->st_dev != 0U && identity->st_ino != 0U &&
           nonzero_token(identity->held_fd_token);
}

static bool valid_expected_binding(
    const struct gate_e_guardian_preflight_witness_binding *const binding
) {
    return binding != NULL && binding->initialized_state == BINDING_INITIALIZED_STATE &&
           valid_root_service_identity(&binding->expected_guardian) &&
           valid_root_service_identity(&binding->expected_warden) &&
           valid_pid1_controller(&binding->expected_pid1_controller) &&
           !identities_equal(&binding->expected_guardian, &binding->expected_warden) &&
           !identities_equal(&binding->expected_guardian, &binding->expected_pid1_controller) &&
           !identities_equal(&binding->expected_warden, &binding->expected_pid1_controller);
}

enum gate_e_guardian_preflight_witness_reason
gate_e_guardian_preflight_witness_binding_v1_set(
    struct gate_e_guardian_preflight_witness_binding *const binding,
    const struct gate_e_guardian_preflight_witness_process_identity *const expected_guardian,
    const struct gate_e_guardian_preflight_witness_process_identity *const expected_warden,
    const struct gate_e_guardian_preflight_witness_process_identity *const expected_pid1_controller
) {
    struct gate_e_guardian_preflight_witness_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_ARGUMENT;
    }
    if (!valid_root_service_identity(expected_guardian) ||
        !valid_root_service_identity(expected_warden) ||
        !valid_pid1_controller(expected_pid1_controller) ||
        identities_equal(expected_guardian, expected_warden) ||
        identities_equal(expected_guardian, expected_pid1_controller) ||
        identities_equal(expected_warden, expected_pid1_controller)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_preflight_witness_binding_v1_init(&candidate);
    candidate.expected_guardian = *expected_guardian;
    candidate.expected_warden = *expected_warden;
    candidate.expected_pid1_controller = *expected_pid1_controller;
    *binding = candidate;
    return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK;
}

static enum gate_e_guardian_preflight_witness_reason match_identity(
    const struct gate_e_guardian_preflight_witness_process_identity *const expected,
    const struct gate_e_guardian_preflight_witness_process_identity *const reported,
    const enum gate_e_guardian_preflight_witness_reason invalid_reason,
    const enum gate_e_guardian_preflight_witness_reason mismatch_reason
) {
    if (!valid_process_identity(reported)) {
        return invalid_reason;
    }
    return identities_equal(expected, reported)
               ? GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK
               : mismatch_reason;
}

static enum gate_e_guardian_preflight_witness_reason match_cgroup(
    const struct gate_e_guardian_preflight_witness_cgroup_claim *const reported
) {
    if (!reported->non_delegated) {
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED;
    }
    return valid_cgroup_identity(&reported->identity)
               ? GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK
               : GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_SUPPLIED_CGROUP;
}

enum gate_e_guardian_preflight_witness_reason
gate_e_match_guardian_preflight_witness_v1(
    const struct gate_e_guardian_preflight_witness_binding *const expected_binding,
    const struct gate_e_guardian_preflight_witness_report *const reported
) {
    enum gate_e_guardian_preflight_witness_reason result;

    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_EXPECTED_BINDING;
    }
    if (reported == NULL) {
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_ARGUMENT;
    }
    result = match_identity(
        &expected_binding->expected_guardian,
        &reported->guardian,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_GUARDIAN,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_GUARDIAN_IDENTITY_MISMATCH
    );
    if (result != GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK) {
        return result;
    }
    result = match_identity(
        &expected_binding->expected_warden,
        &reported->warden,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_WARDEN,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_WARDEN_IDENTITY_MISMATCH
    );
    if (result != GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK) {
        return result;
    }
    result = match_identity(
        &expected_binding->expected_pid1_controller,
        &reported->controller,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_REPORTED_CONTROLLER,
        GATE_E_GUARDIAN_PREFLIGHT_WITNESS_CONTROLLER_IDENTITY_MISMATCH
    );
    if (result != GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK) {
        return result;
    }
    result = match_cgroup(&reported->cgroup);
    if (result != GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK) {
        return result;
    }
    if (reported->population == GATE_E_GUARDIAN_PREFLIGHT_WITNESS_POPULATION_PRESENT) {
        return GATE_E_GUARDIAN_PREFLIGHT_WITNESS_CGROUP_NOT_EMPTY;
    }
    return reported->population == GATE_E_GUARDIAN_PREFLIGHT_WITNESS_POPULATION_EMPTY
               ? GATE_E_GUARDIAN_PREFLIGHT_WITNESS_OK
               : GATE_E_GUARDIAN_PREFLIGHT_WITNESS_INVALID_POPULATION_CLAIM;
}
