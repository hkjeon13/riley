#ifndef RILEY_GATE_E_SEALED_LEAF_SNAPSHOT_H
#define RILEY_GATE_E_SEALED_LEAF_SNAPSHOT_H

#include <stddef.h>
#include <stdint.h>

enum {
    GATE_E_SEALED_LEAF_SHA256_BYTES = 32,
    GATE_E_SEALED_LEAF_MAX_BYTES = 2 * 1024 * 1024,
};

enum gate_e_sealed_leaf_reason {
    GATE_E_SEALED_LEAF_OK,
    GATE_E_SEALED_LEAF_INVALID_ARGUMENT,
    GATE_E_SEALED_LEAF_SOURCE_UNSAFE,
    GATE_E_SEALED_LEAF_SOURCE_UNREADABLE,
    GATE_E_SEALED_LEAF_SOURCE_PIN_FAILED,
    GATE_E_SEALED_LEAF_SOURCE_RACED,
    GATE_E_SEALED_LEAF_DIGEST_MISMATCH,
    GATE_E_SEALED_LEAF_MEMFD_UNAVAILABLE,
    GATE_E_SEALED_LEAF_MEMFD_UNSAFE,
    GATE_E_SEALED_LEAF_MEMFD_WRITE_FAILED,
    GATE_E_SEALED_LEAF_MEMFD_SEAL_FAILED,
    GATE_E_SEALED_LEAF_CLOSE_FAILED,
};

struct gate_e_sealed_leaf_snapshot {
    uint64_t initialized_state;
    int descriptor;
    size_t byte_length;
    unsigned char sha256[GATE_E_SEALED_LEAF_SHA256_BYTES];
};

/*
 * Initialize a new or already-closed snapshot object before passing it to
 * this API. It is not a destructor: callers must close a live snapshot with
 * gate_e_sealed_leaf_snapshot_close before reinitializing it.
 */
void gate_e_sealed_leaf_snapshot_init(struct gate_e_sealed_leaf_snapshot *snapshot);

/*
 * Copy an already-authenticated, held regular source FD into a new anonymous,
 * sealed, close-on-exec, no-exec memfd. The source must be an already-held,
 * O_RDONLY|O_NOATIME, single-link regular object with exactly
 * expected_byte_length bytes; upstream acquisition is responsible for the
 * caller FD's CLOEXEC/provenance contract. This primitive's first operation
 * pins that FD into a private CLOEXEC duplicate and it then uses only
 * fixed-offset reads, so it never changes the caller-visible source offset.
 * It neither authenticates the source pathname nor executes the returned FD.
 * The output must first be initialized and must not already hold a descriptor.
 */
enum gate_e_sealed_leaf_reason gate_e_snapshot_held_leaf_v1(
    int source_descriptor,
    const unsigned char expected_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES],
    size_t expected_byte_length,
    struct gate_e_sealed_leaf_snapshot *output
);

/* Close an output from gate_e_snapshot_held_leaf_v1 and clear it. */
enum gate_e_sealed_leaf_reason gate_e_sealed_leaf_snapshot_close(
    struct gate_e_sealed_leaf_snapshot *snapshot
);

#endif
