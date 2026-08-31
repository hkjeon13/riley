#ifndef RILEY_GATE_E_EXECUTION_CLOSURE_HELD_FDS_V1_H
#define RILEY_GATE_E_EXECUTION_CLOSURE_HELD_FDS_V1_H

/*
 * Source-only retained-FD binding precursor for a future RC3 Gate E native
 * guardian. It binds a canonical execution-closure declaration to already
 * borrowed file descriptors. It has no path, root-policy, ELF, loader,
 * process, secure-exec, cgroup, ledger, GPU, Docker, evidence, or
 * qualification operation.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "gate_e_execution_closure_manifest_v1.h"

enum gate_e_execution_closure_held_fds_reason_v1 {
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1 = 0,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_MANIFEST_REJECTED_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_RUNTIME_COUNT_MISMATCH_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_ALIAS_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_PIN_FAILED_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_UNREADABLE_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_LENGTH_MISMATCH_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_ALIAS_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_BINDING_MISMATCH_V1,
    GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1,
};

struct gate_e_execution_closure_object_identity_v1 {
    dev_t device;
    ino_t inode;
    mode_t mode;
    nlink_t links;
    uid_t uid;
    gid_t gid;
    off_t size;
    struct timespec mtime;
    struct timespec ctime;
};

struct gate_e_execution_closure_held_file_v1 {
    int descriptor;
    struct gate_e_execution_closure_leaf declaration;
    struct gate_e_execution_closure_object_identity_v1 identity;
};

/*
 * The caller owns every descriptor and the runtime array. The binder only
 * borrows them for the duration of one call; it never closes or changes them.
 * Runtime descriptors must appear in exactly the canonical manifest order.
 * The caller must serialize descriptor ownership: no other thread may close,
 * reuse, or change an input or output descriptor slot while bind(), recheck(),
 * or close() is running.
 */
struct gate_e_execution_closure_borrowed_fds_v1 {
    int dynamic_loader_descriptor;
    int interpreter_descriptor;
    const int *runtime_leaf_descriptors;
    size_t runtime_leaf_count;
};

struct gate_e_execution_closure_held_fds_v1 {
    uint64_t initialized_state;
    unsigned char runtime_closure_sha256[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
    struct gate_e_execution_closure_held_file_v1 dynamic_loader;
    struct gate_e_execution_closure_held_file_v1 interpreter;
    size_t runtime_leaf_count;
    struct gate_e_execution_closure_held_file_v1
        runtime_leaves[GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES];
};

/* Initialize an output before binding or after close. */
void gate_e_execution_closure_held_fds_v1_init(
    struct gate_e_execution_closure_held_fds_v1 *output
);

/*
 * Parse one raw canonical execution-closure sidecar, duplicate and bind the
 * role-ordered borrowed descriptors, and return only binder-owned CLOEXEC
 * duplicates plus their declarations, identity, and raw sidecar SHA-256.
 * The raw sidecar and every input descriptor remain caller-owned. The output
 * must be initialized and empty; failures leave it empty. Input objects must
 * remain linked regular files for the duration of the call.
 */
enum gate_e_execution_closure_held_fds_reason_v1
gate_e_bind_execution_closure_held_fds_v1(
    const unsigned char *raw_manifest,
    size_t raw_manifest_length,
    const struct gate_e_execution_closure_borrowed_fds_v1 *borrowed,
    struct gate_e_execution_closure_held_fds_v1 *output
);

/*
 * Re-pin and recheck every output-owned descriptor against its saved
 * declaration, SHA-256, and full object identity without reopening a path.
 * This is not an atomic multi-object snapshot or a provenance/ELF check.
 */
enum gate_e_execution_closure_held_fds_reason_v1
gate_e_execution_closure_held_fds_v1_recheck(
    const struct gate_e_execution_closure_held_fds_v1 *held
);

/* Close every output-owned descriptor once and reset the output. */
enum gate_e_execution_closure_held_fds_reason_v1
gate_e_execution_closure_held_fds_v1_close(
    struct gate_e_execution_closure_held_fds_v1 *held
);

#endif
