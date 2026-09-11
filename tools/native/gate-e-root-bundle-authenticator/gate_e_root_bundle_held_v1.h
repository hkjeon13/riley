#ifndef GATE_E_ROOT_BUNDLE_HELD_V1_H
#define GATE_E_ROOT_BUNDLE_HELD_V1_H

/*
 * Source-library boundary for the future RC3 Gate E native guardian.  This
 * API retains only already-authenticated, read-only, CLOEXEC descriptors for
 * the fixed future guardian bundle.  It does not execute any descriptor or
 * create a launcher, child, cgroup, ledger, socket, lock, GPU/Docker action,
 * evidence, receipt, or qualification result.
 */

#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

enum {
    GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1 = 4,
    GATE_E_ROOT_BUNDLE_SHA256_DIGEST_BYTES_V1 = 32,
};

enum gate_e_root_bundle_reason_v1 {
    GATE_E_ROOT_BUNDLE_OK_V1,
    GATE_E_ROOT_BUNDLE_NOT_ROOT_V1,
    GATE_E_ROOT_BUNDLE_OPENAT2_UNAVAILABLE_V1,
    GATE_E_ROOT_BUNDLE_ROOT_UNREADABLE_V1,
    GATE_E_ROOT_BUNDLE_UNSAFE_DIRECTORY_V1,
    GATE_E_ROOT_BUNDLE_UNSAFE_FILESYSTEM_V1,
    GATE_E_ROOT_BUNDLE_ACL_PRESENT_V1,
    GATE_E_ROOT_BUNDLE_ACL_UNVERIFIABLE_V1,
    GATE_E_ROOT_BUNDLE_CAPABILITY_PRESENT_V1,
    GATE_E_ROOT_BUNDLE_UNSAFE_FILE_V1,
    GATE_E_ROOT_BUNDLE_FILE_UNREADABLE_V1,
    GATE_E_ROOT_BUNDLE_OBJECT_RACED_V1,
    GATE_E_ROOT_BUNDLE_MANIFEST_INVALID_V1,
    GATE_E_ROOT_BUNDLE_DIGEST_MISMATCH_V1,
    GATE_E_ROOT_BUNDLE_CLOSE_FAILED_V1,
    GATE_E_ROOT_BUNDLE_INVALID_ARGUMENT_V1,
};

struct gate_e_root_bundle_object_identity_v1 {
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

struct gate_e_root_bundle_held_directory_v1 {
    int descriptor;
    struct gate_e_root_bundle_object_identity_v1 identity;
};

struct gate_e_root_bundle_held_file_v1 {
    int descriptor;
    struct gate_e_root_bundle_object_identity_v1 identity;
    unsigned char digest[GATE_E_ROOT_BUNDLE_SHA256_DIGEST_BYTES_V1];
    size_t byte_length;
};

struct gate_e_root_bundle_held_v1 {
    struct gate_e_root_bundle_held_directory_v1
        directories[GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1];
    struct gate_e_root_bundle_held_file_v1 manifest;
    struct gate_e_root_bundle_held_file_v1 bootstrap;
    struct gate_e_root_bundle_held_file_v1 core;
};

/* Initialize before first use or after ownership has been transferred away. */
void gate_e_root_bundle_held_v1_init(struct gate_e_root_bundle_held_v1 *held);

/*
 * Acquire the fixed /opt/riley/rc3-gate-e-v1 future-guardian bundle for a
 * root-only caller.  It accepts no caller-controlled path, identity, leaf,
 * manifest, lock, or configuration. The caller must pass an initialized or
 * closed handle; a non-clear handle is rejected without being altered. For a
 * clear handle, an acquisition failure leaves it reset with no descriptor. A
 * successful result remains non-authoritative until a separately reviewed
 * static guardian performs later work.
 */
enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_acquire_fixed_v1(
    struct gate_e_root_bundle_held_v1 *held
);

/* Recheck the same held objects and their fixed parent/name bindings. */
enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_held_v1_recheck(
    const struct gate_e_root_bundle_held_v1 *held
);

/* Close every owned descriptor once and reset the caller-owned handle. */
enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_held_v1_close(
    struct gate_e_root_bundle_held_v1 *held
);

#endif
