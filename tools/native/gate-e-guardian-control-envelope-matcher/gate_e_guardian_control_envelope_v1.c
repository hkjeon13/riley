/*
 * Pure C11 matcher for normalized future Gate E control-envelope claims.
 *
 * This library compares caller-owned typed values only. It does not receive a
 * socket message, inspect a live pidfd/cgroup/FD, parse JSON, or change state.
 */

#include "gate_e_guardian_control_envelope_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494c455947454d);

static void clear_binding(struct gate_e_guardian_control_envelope_binding *const binding) {
    memset(&binding->registered_worker, 0, sizeof(binding->registered_worker));
    memset(&binding->held_cgroup, 0, sizeof(binding->held_cgroup));
}

void gate_e_guardian_control_envelope_binding_v1_init(
    struct gate_e_guardian_control_envelope_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool nonzero_token(
    const unsigned char token[GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES]
) {
    unsigned char nonzero = 0U;

    for (size_t index = 0U; index < GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES; ++index) {
        nonzero |= token[index];
    }
    return nonzero != 0U;
}

static bool valid_process_identity(
    const struct gate_e_guardian_process_identity *const identity,
    const bool require_unprivileged
) {
    return identity != NULL && identity->pid != 0U &&
           identity->pid <= GATE_E_GUARDIAN_CONTROL_ENVELOPE_MAX_PID &&
           identity->starttime_ticks != 0U && nonzero_token(identity->pidfd_token) &&
           (!require_unprivileged || (identity->uid != 0U && identity->gid != 0U));
}

static bool valid_cgroup_identity(
    const struct gate_e_guardian_cgroup_identity *const identity
) {
    return identity != NULL && identity->st_dev != 0U && identity->st_ino != 0U &&
           nonzero_token(identity->held_fd_token);
}

static bool valid_expected_binding(
    const struct gate_e_guardian_control_envelope_binding *const binding
) {
    return binding != NULL && binding->initialized_state == BINDING_INITIALIZED_STATE &&
           valid_process_identity(&binding->registered_worker, true) &&
           valid_cgroup_identity(&binding->held_cgroup);
}

enum gate_e_guardian_control_envelope_reason
gate_e_guardian_control_envelope_binding_v1_set(
    struct gate_e_guardian_control_envelope_binding *const binding,
    const struct gate_e_guardian_process_identity *const worker,
    const struct gate_e_guardian_cgroup_claim *const held_cgroup
) {
    struct gate_e_guardian_control_envelope_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT;
    }
    if (!valid_process_identity(worker, true) || held_cgroup == NULL ||
        !held_cgroup->non_delegated || !valid_cgroup_identity(&held_cgroup->identity)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_control_envelope_binding_v1_init(&candidate);
    candidate.registered_worker = *worker;
    candidate.held_cgroup = held_cgroup->identity;
    *binding = candidate;
    return GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK;
}

static enum gate_e_guardian_control_envelope_reason match_credentials(
    const struct gate_e_guardian_process_identity *const expected,
    const struct gate_e_guardian_process_identity *const reported
) {
    if (!valid_process_identity(reported, false)) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS;
    }
    if (reported->pid != expected->pid) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PID_MISMATCH;
    }
    if (reported->starttime_ticks != expected->starttime_ticks) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_STARTTIME_TICKS_MISMATCH;
    }
    if (memcmp(reported->pidfd_token, expected->pidfd_token, sizeof(reported->pidfd_token)) != 0) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PIDFD_TOKEN_MISMATCH;
    }
    if (reported->uid != expected->uid) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_UID_MISMATCH;
    }
    return reported->gid == expected->gid ? GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK
                                          : GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_GID_MISMATCH;
}

static enum gate_e_guardian_control_envelope_reason match_cgroup(
    const struct gate_e_guardian_cgroup_identity *const expected,
    const struct gate_e_guardian_cgroup_claim *const reported
) {
    if (!reported->non_delegated) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_SUPPLIED_CGROUP_NOT_NON_DELEGATED;
    }
    if (!valid_cgroup_identity(&reported->identity)) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_SUPPLIED_CGROUP;
    }
    if (reported->identity.st_dev != expected->st_dev) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_DEV_MISMATCH;
    }
    if (reported->identity.st_ino != expected->st_ino) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_INO_MISMATCH;
    }
    return memcmp(
               reported->identity.held_fd_token,
               expected->held_fd_token,
               sizeof(reported->identity.held_fd_token)
           ) == 0
               ? GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK
               : GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_HELD_FD_TOKEN_MISMATCH;
}

enum gate_e_guardian_control_envelope_reason
gate_e_match_guardian_control_envelope_v1(
    const struct gate_e_guardian_control_envelope_binding *const expected_binding,
    const struct gate_e_guardian_control_envelope_report *const reported
) {
    enum gate_e_guardian_control_envelope_reason result;

    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING;
    }
    if (reported == NULL) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT;
    }
    result = match_credentials(&expected_binding->registered_worker, &reported->credentials);
    if (result != GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK) {
        return result;
    }
    result = match_cgroup(&expected_binding->held_cgroup, &reported->cgroup);
    if (result != GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK) {
        return result;
    }
    if (reported->ancillary == GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_EMPTY) {
        return GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK;
    }
    return reported->ancillary == GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_PRESENT
               ? GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_FDS_PRESENT
               : GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ANCILLARY_CLAIM;
}
