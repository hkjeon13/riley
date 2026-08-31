/*
 * Source-only composition of two existing future-guardian primitives. This
 * library borrows a retained root-bundle handle and produces sealed data-only
 * bootstrap/core snapshots. It has no path, CLI, configuration, acquisition,
 * exec, process, cgroup, lock, GPU, Docker, evidence, or qualification path.
 */

#include "gate_e_root_bundle_sealed_leaves_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

static const uint64_t SEALED_LEAVES_INITIALIZED_STATE = UINT64_C(0x52494c4553424c56);

static bool digest_is_zero(const unsigned char digest[GATE_E_SEALED_LEAF_SHA256_BYTES]) {
    unsigned char combined = 0U;

    for (size_t index = 0; index < GATE_E_SEALED_LEAF_SHA256_BYTES; ++index) {
        combined |= digest[index];
    }
    return combined == 0U;
}

static bool snapshot_is_empty(const struct gate_e_sealed_leaf_snapshot *const snapshot) {
    return snapshot->descriptor == -1 && snapshot->byte_length == 0U &&
           digest_is_zero(snapshot->sha256);
}

static bool output_is_initialized_empty(
    const struct gate_e_root_bundle_sealed_leaves_v1 *const output
) {
    return output != NULL && output->initialized_state == SEALED_LEAVES_INITIALIZED_STATE &&
           output->bootstrap.initialized_state != 0U && output->core.initialized_state != 0U &&
           snapshot_is_empty(&output->bootstrap) && snapshot_is_empty(&output->core);
}

void gate_e_root_bundle_sealed_leaves_v1_init(
    struct gate_e_root_bundle_sealed_leaves_v1 *const output
) {
    if (output == NULL) {
        return;
    }
    output->initialized_state = SEALED_LEAVES_INITIALIZED_STATE;
    gate_e_sealed_leaf_snapshot_init(&output->bootstrap);
    gate_e_sealed_leaf_snapshot_init(&output->core);
}

enum gate_e_root_bundle_sealed_leaves_reason_v1
gate_e_root_bundle_sealed_leaves_v1_close(
    struct gate_e_root_bundle_sealed_leaves_v1 *const output
) {
    enum gate_e_sealed_leaf_reason bootstrap_reason;
    enum gate_e_sealed_leaf_reason core_reason;

    if (output == NULL || output->initialized_state != SEALED_LEAVES_INITIALIZED_STATE) {
        return GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1;
    }
    bootstrap_reason = gate_e_sealed_leaf_snapshot_close(&output->bootstrap);
    core_reason = gate_e_sealed_leaf_snapshot_close(&output->core);
    gate_e_root_bundle_sealed_leaves_v1_init(output);
    return bootstrap_reason == GATE_E_SEALED_LEAF_OK && core_reason == GATE_E_SEALED_LEAF_OK
               ? GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1
               : GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CLOSE_FAILED_V1;
}

static enum gate_e_root_bundle_sealed_leaves_reason_v1 clear_after_failure(
    struct gate_e_root_bundle_sealed_leaves_v1 *const output,
    const enum gate_e_root_bundle_sealed_leaves_reason_v1 failure_reason
) {
    return gate_e_root_bundle_sealed_leaves_v1_close(output) == GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1
               ? failure_reason
               : GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CLOSE_FAILED_V1;
}

static bool snapshot_matches_held_file(
    const struct gate_e_sealed_leaf_snapshot *const snapshot,
    const struct gate_e_root_bundle_held_file_v1 *const held_file
) {
    return snapshot->descriptor >= 3 && snapshot->byte_length == held_file->byte_length &&
           memcmp(snapshot->sha256, held_file->digest, sizeof(snapshot->sha256)) == 0;
}

static bool held_bundle_rechecks(const struct gate_e_root_bundle_held_v1 *const held) {
    return gate_e_root_bundle_held_v1_recheck(held) == GATE_E_ROOT_BUNDLE_OK_V1;
}

enum gate_e_root_bundle_sealed_leaves_reason_v1
gate_e_snapshot_held_root_bundle_leaves_v1(
    const struct gate_e_root_bundle_held_v1 *const held,
    struct gate_e_root_bundle_sealed_leaves_v1 *const output
) {
    enum gate_e_sealed_leaf_reason snapshot_reason;

    if (!output_is_initialized_empty(output) || held == NULL) {
        return GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1;
    }
    if (!held_bundle_rechecks(held)) {
        return GATE_E_ROOT_BUNDLE_SEALED_LEAVES_HELD_BUNDLE_RECHECK_FAILED_V1;
    }
    snapshot_reason = gate_e_snapshot_held_leaf_v1(
        held->bootstrap.descriptor,
        held->bootstrap.digest,
        held->bootstrap.byte_length,
        &output->bootstrap
    );
    if (snapshot_reason != GATE_E_SEALED_LEAF_OK) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_BOOTSTRAP_SNAPSHOT_FAILED_V1
        );
    }
    if (!snapshot_matches_held_file(&output->bootstrap, &held->bootstrap)) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_SNAPSHOT_BINDING_MISMATCH_V1
        );
    }
    if (!held_bundle_rechecks(held)) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_HELD_BUNDLE_RECHECK_FAILED_V1
        );
    }
    snapshot_reason = gate_e_snapshot_held_leaf_v1(
        held->core.descriptor,
        held->core.digest,
        held->core.byte_length,
        &output->core
    );
    if (snapshot_reason != GATE_E_SEALED_LEAF_OK) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CORE_SNAPSHOT_FAILED_V1
        );
    }
    if (!snapshot_matches_held_file(&output->core, &held->core)) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_SNAPSHOT_BINDING_MISMATCH_V1
        );
    }
    if (!held_bundle_rechecks(held)) {
        return clear_after_failure(
            output, GATE_E_ROOT_BUNDLE_SEALED_LEAVES_HELD_BUNDLE_RECHECK_FAILED_V1
        );
    }
    return GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1;
}
