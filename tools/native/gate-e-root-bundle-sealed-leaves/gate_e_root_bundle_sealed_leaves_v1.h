#ifndef RILEY_GATE_E_ROOT_BUNDLE_SEALED_LEAVES_V1_H
#define RILEY_GATE_E_ROOT_BUNDLE_SEALED_LEAVES_V1_H

/*
 * Source-only composition boundary for a future RC3 Gate E native guardian.
 * It borrows an already-authenticated root-bundle handle and creates only two
 * immutable no-exec data snapshots. It neither acquires nor closes the input
 * handle and does not arrange an exec, FD 31/32, child, lease, cgroup, GPU,
 * Docker, evidence, receipt, or qualification operation.
 */

#include <stdint.h>

#include "gate_e_root_bundle_held_v1.h"
#include "gate_e_sealed_leaf_snapshot.h"

enum gate_e_root_bundle_sealed_leaves_reason_v1 {
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_HELD_BUNDLE_RECHECK_FAILED_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_BOOTSTRAP_SNAPSHOT_FAILED_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CORE_SNAPSHOT_FAILED_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_SNAPSHOT_BINDING_MISMATCH_V1,
    GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CLOSE_FAILED_V1,
};

struct gate_e_root_bundle_sealed_leaves_v1 {
    uint64_t initialized_state;
    struct gate_e_sealed_leaf_snapshot bootstrap;
    struct gate_e_sealed_leaf_snapshot core;
};

/* Initialize a new or closed output before passing it to this API. */
void gate_e_root_bundle_sealed_leaves_v1_init(
    struct gate_e_root_bundle_sealed_leaves_v1 *output
);

/*
 * Borrow a held root-bundle handle, recheck it before/between/after copying,
 * and publish sealed bootstrap/core data snapshots on success. The input
 * remains caller-owned and is never acquired, closed, or modified. The output
 * must be initialized and empty; any failure clears only output snapshots.
 */
enum gate_e_root_bundle_sealed_leaves_reason_v1
gate_e_snapshot_held_root_bundle_leaves_v1(
    const struct gate_e_root_bundle_held_v1 *held,
    struct gate_e_root_bundle_sealed_leaves_v1 *output
);

/* Close output snapshots and reset the initialized composite. */
enum gate_e_root_bundle_sealed_leaves_reason_v1
gate_e_root_bundle_sealed_leaves_v1_close(
    struct gate_e_root_bundle_sealed_leaves_v1 *output
);

#endif
