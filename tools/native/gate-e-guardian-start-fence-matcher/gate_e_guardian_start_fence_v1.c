/*
 * Pure C11 matcher for normalized Gate E START boot/generation fence claims.
 *
 * It compares caller-owned values only; it never accesses a ledger, process,
 * cgroup, descriptor, socket, or filesystem object.
 */

#include "gate_e_guardian_start_fence_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494C455346454E);

static void clear_binding(struct gate_e_guardian_start_fence_binding *const binding) {
    binding->mode = GATE_E_GUARDIAN_START_FENCE_MODE_INVALID;
    memset(binding->boot_id, 0, sizeof(binding->boot_id));
    binding->highest_generation = 0U;
}

void gate_e_guardian_start_fence_binding_v1_init(
    struct gate_e_guardian_start_fence_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
}

static bool bytes_are_zero(
    const unsigned char bytes[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES]
) {
    unsigned char aggregate = 0U;

    if (bytes == NULL) {
        return false;
    }
    for (size_t index = 0U; index < GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES; ++index) {
        aggregate |= bytes[index];
    }
    return aggregate == 0U;
}

static bool valid_expected_binding(
    const struct gate_e_guardian_start_fence_binding *const binding
) {
    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return false;
    }
    if (binding->mode == GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED) {
        return bytes_are_zero(binding->boot_id) && binding->highest_generation == 0U;
    }
    if (binding->mode == GATE_E_GUARDIAN_START_FENCE_MODE_FENCED) {
        return !bytes_are_zero(binding->boot_id);
    }
    return false;
}

enum gate_e_guardian_start_fence_reason
gate_e_guardian_start_fence_binding_v1_set(
    struct gate_e_guardian_start_fence_binding *const binding,
    const enum gate_e_guardian_start_fence_mode mode,
    const unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES],
    const uint64_t highest_generation
) {
    struct gate_e_guardian_start_fence_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_START_FENCE_INVALID_ARGUMENT;
    }
    if (
        boot_id == NULL ||
        (mode == GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED &&
         (!bytes_are_zero(boot_id) || highest_generation != 0U)) ||
        (mode == GATE_E_GUARDIAN_START_FENCE_MODE_FENCED && bytes_are_zero(boot_id)) ||
        (mode != GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED &&
         mode != GATE_E_GUARDIAN_START_FENCE_MODE_FENCED)
    ) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE;
    }
    gate_e_guardian_start_fence_binding_v1_init(&candidate);
    candidate.mode = mode;
    memcpy(candidate.boot_id, boot_id, sizeof(candidate.boot_id));
    candidate.highest_generation = highest_generation;
    *binding = candidate;
    return GATE_E_GUARDIAN_START_FENCE_OK;
}

enum gate_e_guardian_start_fence_reason
gate_e_match_guardian_start_fence_v1(
    const struct gate_e_guardian_start_fence_binding *const expected_binding,
    const struct gate_e_guardian_start_fence_candidate *const candidate
) {
    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE;
    }
    if (candidate == NULL || bytes_are_zero(candidate->boot_id) || candidate->generation == 0U) {
        return GATE_E_GUARDIAN_START_FENCE_INVALID_CANDIDATE;
    }
    if (expected_binding->mode == GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED) {
        return GATE_E_GUARDIAN_START_FENCE_OK;
    }
    if (memcmp(expected_binding->boot_id, candidate->boot_id, sizeof(candidate->boot_id)) != 0) {
        return GATE_E_GUARDIAN_START_FENCE_DURABLE_RECOVERY_REQUIRED;
    }
    return candidate->generation > expected_binding->highest_generation
               ? GATE_E_GUARDIAN_START_FENCE_OK
               : GATE_E_GUARDIAN_START_FENCE_GENERATION_REPLAYED;
}
