#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gate_e_root_bundle_sealed_leaves_v1.h"

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/memfd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef F_SEAL_EXEC
#define F_SEAL_EXEC 0x0020
#endif

enum {
    REQUIRED_SEALS = F_SEAL_EXEC | F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE,
};

struct fixture {
    char root[PATH_MAX];
    char opt[PATH_MAX];
    char riley[PATH_MAX];
    char bundle[PATH_MAX];
    char manifest[PATH_MAX];
    char bootstrap[PATH_MAX];
    char core[PATH_MAX];
};

static const char BOOTSTRAP_BYTES[] = "abc";
static const char CORE_BYTES[] = "def";
static const char MANIFEST_BYTES[] = "fixture manifest\n";
static const unsigned char BOOTSTRAP_SHA256[GATE_E_SEALED_LEAF_SHA256_BYTES] = {
    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
    0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
    0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
    0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
};
static const unsigned char CORE_SHA256[GATE_E_SEALED_LEAF_SHA256_BYTES] = {
    0xcb, 0x83, 0x79, 0xac, 0x20, 0x98, 0xaa, 0x16,
    0x50, 0x29, 0xe3, 0x93, 0x8a, 0x51, 0xda, 0x0b,
    0xce, 0xcf, 0xc0, 0x08, 0xfd, 0x67, 0x95, 0xf4,
    0x01, 0x17, 0x86, 0x47, 0xf9, 0x6c, 0x5b, 0x34,
};

static void write_exact_file(const char *const path, const char *const bytes, const mode_t mode) {
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode);
    const size_t length = strlen(bytes);
    size_t written = 0U;

    assert(descriptor >= 0);
    while (written < length) {
        const ssize_t count = write(descriptor, bytes + written, length - written);

        assert(count > 0);
        written += (size_t)count;
    }
    assert(fsync(descriptor) == 0);
    assert(close(descriptor) == 0);
    assert(chmod(path, mode) == 0);
}

static struct fixture make_fixture(void) {
    struct fixture fixture = {0};
    char template[] = "/tmp/riley-root-bundle-sealed-leaves.XXXXXX";
    char *const root = mkdtemp(template);

    assert(root != NULL);
    assert(strlen(root) < sizeof(fixture.root));
    memcpy(fixture.root, root, strlen(root) + 1U);
    assert(chmod(fixture.root, 0755) == 0);
    assert(snprintf(fixture.opt, sizeof(fixture.opt), "%s/opt", fixture.root) > 0);
    assert(snprintf(fixture.riley, sizeof(fixture.riley), "%s/riley", fixture.opt) > 0);
    assert(snprintf(fixture.bundle, sizeof(fixture.bundle), "%s/rc3-gate-e-v1", fixture.riley) > 0);
    assert(mkdir(fixture.opt, 0755) == 0);
    assert(mkdir(fixture.riley, 0755) == 0);
    assert(mkdir(fixture.bundle, 0755) == 0);
    assert(chmod(fixture.opt, 0755) == 0);
    assert(chmod(fixture.riley, 0755) == 0);
    assert(chmod(fixture.bundle, 0755) == 0);
    assert(snprintf(fixture.manifest, sizeof(fixture.manifest), "%s/gate-e-v3.manifest.json", fixture.bundle) > 0);
    assert(snprintf(fixture.bootstrap, sizeof(fixture.bootstrap), "%s/rc3_gate_e_guardian_bootstrap_v1.py", fixture.bundle) > 0);
    assert(snprintf(fixture.core, sizeof(fixture.core), "%s/rc3_gate_e_guardian_no_action_core_v1.py", fixture.bundle) > 0);
    write_exact_file(fixture.manifest, MANIFEST_BYTES, 0644);
    write_exact_file(fixture.bootstrap, BOOTSTRAP_BYTES, 0755);
    write_exact_file(fixture.core, CORE_BYTES, 0644);
    return fixture;
}

static void destroy_fixture(const struct fixture *const fixture) {
    (void)unlink(fixture->manifest);
    (void)unlink(fixture->bootstrap);
    (void)unlink(fixture->core);
    (void)rmdir(fixture->bundle);
    (void)rmdir(fixture->riley);
    (void)rmdir(fixture->opt);
    (void)rmdir(fixture->root);
}

static struct gate_e_root_bundle_object_identity_v1 identity_from_descriptor(const int descriptor) {
    struct stat metadata;

    assert(fstat(descriptor, &metadata) == 0);
    return (struct gate_e_root_bundle_object_identity_v1){
        .device = metadata.st_dev,
        .inode = metadata.st_ino,
        .mode = metadata.st_mode,
        .links = metadata.st_nlink,
        .uid = metadata.st_uid,
        .gid = metadata.st_gid,
        .size = metadata.st_size,
        .mtime = metadata.st_mtim,
        .ctime = metadata.st_ctim,
    };
}

static int open_held_directory(const char *const path) {
    const int descriptor = open(
        path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOATIME | O_NOFOLLOW
    );

    assert(descriptor >= 3);
    return descriptor;
}

static int open_held_file(const char *const path) {
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOATIME | O_NOFOLLOW);

    assert(descriptor >= 3);
    return descriptor;
}

static void populate_pre_authenticated_held_fixture(
    const struct fixture *const fixture,
    struct gate_e_root_bundle_held_v1 *const held
) {
    gate_e_root_bundle_held_v1_init(held);
    held->directories[0].descriptor = open_held_directory(fixture->root);
    held->directories[0].identity = identity_from_descriptor(held->directories[0].descriptor);
    held->directories[1].descriptor = open_held_directory(fixture->opt);
    held->directories[1].identity = identity_from_descriptor(held->directories[1].descriptor);
    held->directories[2].descriptor = open_held_directory(fixture->riley);
    held->directories[2].identity = identity_from_descriptor(held->directories[2].descriptor);
    held->directories[3].descriptor = open_held_directory(fixture->bundle);
    held->directories[3].identity = identity_from_descriptor(held->directories[3].descriptor);
    held->manifest.descriptor = open_held_file(fixture->manifest);
    held->manifest.identity = identity_from_descriptor(held->manifest.descriptor);
    held->manifest.byte_length = (size_t)held->manifest.identity.size;
    held->bootstrap.descriptor = open_held_file(fixture->bootstrap);
    held->bootstrap.identity = identity_from_descriptor(held->bootstrap.descriptor);
    held->bootstrap.byte_length = strlen(BOOTSTRAP_BYTES);
    memcpy(held->bootstrap.digest, BOOTSTRAP_SHA256, sizeof(held->bootstrap.digest));
    held->core.descriptor = open_held_file(fixture->core);
    held->core.identity = identity_from_descriptor(held->core.descriptor);
    held->core.byte_length = strlen(CORE_BYTES);
    memcpy(held->core.digest, CORE_SHA256, sizeof(held->core.digest));
    assert(gate_e_root_bundle_held_v1_recheck(held) == GATE_E_ROOT_BUNDLE_OK_V1);
}

static void assert_pair_empty(const struct gate_e_root_bundle_sealed_leaves_v1 *const pair) {
    static const unsigned char zero_digest[GATE_E_SEALED_LEAF_SHA256_BYTES] = {0};

    assert(pair->bootstrap.descriptor == -1);
    assert(pair->core.descriptor == -1);
    assert(pair->bootstrap.byte_length == 0U);
    assert(pair->core.byte_length == 0U);
    assert(memcmp(pair->bootstrap.sha256, zero_digest, sizeof(zero_digest)) == 0);
    assert(memcmp(pair->core.sha256, zero_digest, sizeof(zero_digest)) == 0);
}

static void assert_sealed_snapshot(
    const struct gate_e_sealed_leaf_snapshot *const snapshot,
    const char *const expected_bytes,
    const unsigned char expected_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES]
) {
    struct stat metadata;
    char observed[8] = {0};
    const int descriptor_flags = fcntl(snapshot->descriptor, F_GETFD);
    const int seals = fcntl(snapshot->descriptor, F_GET_SEALS);

    assert(snapshot->descriptor >= 3);
    assert(descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0);
    assert(seals == REQUIRED_SEALS);
    assert(snapshot->byte_length == strlen(expected_bytes));
    assert(memcmp(snapshot->sha256, expected_sha256, sizeof(snapshot->sha256)) == 0);
    assert(fstat(snapshot->descriptor, &metadata) == 0);
    assert(S_ISREG(metadata.st_mode));
    assert(metadata.st_nlink == 0);
    assert((metadata.st_mode & 0111) == 0);
    assert(pread(snapshot->descriptor, observed, strlen(expected_bytes), 0) ==
           (ssize_t)strlen(expected_bytes));
    assert(memcmp(observed, expected_bytes, strlen(expected_bytes)) == 0);
}

static void test_success_preserves_borrowed_handle(void) {
    const struct fixture fixture = make_fixture();
    struct gate_e_root_bundle_held_v1 held;
    struct gate_e_root_bundle_sealed_leaves_v1 pair;
    off_t bootstrap_offset_before;
    off_t core_offset_before;
    int bootstrap_descriptor_before;
    int core_descriptor_before;
    unsigned char bootstrap_digest_before[GATE_E_SEALED_LEAF_SHA256_BYTES];
    unsigned char core_digest_before[GATE_E_SEALED_LEAF_SHA256_BYTES];

    populate_pre_authenticated_held_fixture(&fixture, &held);
    bootstrap_descriptor_before = held.bootstrap.descriptor;
    core_descriptor_before = held.core.descriptor;
    memcpy(bootstrap_digest_before, held.bootstrap.digest, sizeof(bootstrap_digest_before));
    memcpy(core_digest_before, held.core.digest, sizeof(core_digest_before));
    gate_e_root_bundle_sealed_leaves_v1_init(&pair);
    assert_pair_empty(&pair);
    bootstrap_offset_before = lseek(held.bootstrap.descriptor, 0, SEEK_CUR);
    core_offset_before = lseek(held.core.descriptor, 0, SEEK_CUR);
    assert(bootstrap_offset_before >= 0 && core_offset_before >= 0);
    assert(gate_e_snapshot_held_root_bundle_leaves_v1(&held, &pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1);
    assert(held.bootstrap.descriptor == bootstrap_descriptor_before);
    assert(held.core.descriptor == core_descriptor_before);
    assert(memcmp(held.bootstrap.digest, bootstrap_digest_before, sizeof(bootstrap_digest_before)) == 0);
    assert(memcmp(held.core.digest, core_digest_before, sizeof(core_digest_before)) == 0);
    assert(lseek(held.bootstrap.descriptor, 0, SEEK_CUR) == bootstrap_offset_before);
    assert(lseek(held.core.descriptor, 0, SEEK_CUR) == core_offset_before);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    assert_sealed_snapshot(&pair.bootstrap, BOOTSTRAP_BYTES, BOOTSTRAP_SHA256);
    assert_sealed_snapshot(&pair.core, CORE_BYTES, CORE_SHA256);
    {
        const int bootstrap_descriptor = pair.bootstrap.descriptor;

        assert(gate_e_snapshot_held_root_bundle_leaves_v1(&held, &pair) ==
               GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1);
        assert(pair.bootstrap.descriptor == bootstrap_descriptor);
    }
    assert(gate_e_root_bundle_sealed_leaves_v1_close(&pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1);
    assert_pair_empty(&pair);
    assert(gate_e_root_bundle_sealed_leaves_v1_close(&pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_OK_V1);
    assert(gate_e_root_bundle_held_v1_close(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    destroy_fixture(&fixture);
}

static void test_rejects_held_recheck_and_core_digest_drift(void) {
    const struct fixture fixture = make_fixture();
    struct gate_e_root_bundle_held_v1 held;
    struct gate_e_root_bundle_sealed_leaves_v1 pair;
    int descriptor_flags;
    unsigned char original_digest_byte;

    populate_pre_authenticated_held_fixture(&fixture, &held);
    gate_e_root_bundle_sealed_leaves_v1_init(&pair);
    descriptor_flags = fcntl(held.bootstrap.descriptor, F_GETFD);
    assert(descriptor_flags >= 0);
    assert(fcntl(held.bootstrap.descriptor, F_SETFD, descriptor_flags & ~FD_CLOEXEC) == 0);
    assert(gate_e_snapshot_held_root_bundle_leaves_v1(&held, &pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_HELD_BUNDLE_RECHECK_FAILED_V1);
    assert_pair_empty(&pair);
    assert(fcntl(held.bootstrap.descriptor, F_SETFD, descriptor_flags) == 0);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == GATE_E_ROOT_BUNDLE_OK_V1);

    original_digest_byte = held.core.digest[0];
    held.core.digest[0] ^= 0x80U;
    assert(gate_e_snapshot_held_root_bundle_leaves_v1(&held, &pair) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_CORE_SNAPSHOT_FAILED_V1);
    assert_pair_empty(&pair);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    held.core.digest[0] = original_digest_byte;
    assert(gate_e_root_bundle_held_v1_close(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    destroy_fixture(&fixture);
}

static void test_invalid_output_is_not_touched(void) {
    const struct fixture fixture = make_fixture();
    struct gate_e_root_bundle_held_v1 held;
    struct gate_e_root_bundle_sealed_leaves_v1 uninitialized = {0};

    populate_pre_authenticated_held_fixture(&fixture, &held);
    assert(gate_e_snapshot_held_root_bundle_leaves_v1(&held, &uninitialized) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1);
    assert(uninitialized.initialized_state == 0U);
    assert(gate_e_root_bundle_sealed_leaves_v1_close(&uninitialized) ==
           GATE_E_ROOT_BUNDLE_SEALED_LEAVES_INVALID_ARGUMENT_V1);
    assert(gate_e_root_bundle_held_v1_close(&held) == GATE_E_ROOT_BUNDLE_OK_V1);
    destroy_fixture(&fixture);
}

int main(void) {
    test_success_preserves_borrowed_handle();
    test_rejects_held_recheck_and_core_digest_drift();
    test_invalid_output_is_not_touched();
    (void)puts("gate-e root bundle sealed leaves tests passed");
    return 0;
}
