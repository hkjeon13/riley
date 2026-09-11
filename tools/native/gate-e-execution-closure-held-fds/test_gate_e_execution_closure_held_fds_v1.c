#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gate_e_execution_closure_held_fds_v1.h"

#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    LARGE_RUNTIME_BYTE_LENGTH = 8193,
};

struct fixture {
    char root[PATH_MAX];
    char loader[PATH_MAX];
    char interpreter[PATH_MAX];
    char runtime[PATH_MAX];
    char runtime_second[PATH_MAX];
    char runtime_large[PATH_MAX];
};

static const char LOADER_DIGEST_HEX[] =
    "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb";
static const char INTERPRETER_DIGEST_HEX[] =
    "3e23e8160039594a33894f6564e1b1348bbd7a0088d42c4acb73eeaed59c009d";
static const char RUNTIME_DIGEST_HEX[] =
    "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6";
static const char RUNTIME_SECOND_DIGEST_HEX[] =
    "18ac3e7343f016890c510e93f935261169d9e3f565436429830faf0934f4f8e4";
static const char RUNTIME_LARGE_DIGEST_HEX[] =
    "68e3a09e97afe1dc99525a1e569b653d704bcb543cbae0b9b926307236f907d1";

static const unsigned char LOADER_DIGEST[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0xca, 0x97, 0x81, 0x12, 0xca, 0x1b, 0xbd, 0xca,
    0xfa, 0xc2, 0x31, 0xb3, 0x9a, 0x23, 0xdc, 0x4d,
    0xa7, 0x86, 0xef, 0xf8, 0x14, 0x7c, 0x4e, 0x72,
    0xb9, 0x80, 0x77, 0x85, 0xaf, 0xee, 0x48, 0xbb,
};
static const unsigned char INTERPRETER_DIGEST[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0x3e, 0x23, 0xe8, 0x16, 0x00, 0x39, 0x59, 0x4a,
    0x33, 0x89, 0x4f, 0x65, 0x64, 0xe1, 0xb1, 0x34,
    0x8b, 0xbd, 0x7a, 0x00, 0x88, 0xd4, 0x2c, 0x4a,
    0xcb, 0x73, 0xee, 0xae, 0xd5, 0x9c, 0x00, 0x9d,
};
static const unsigned char RUNTIME_DIGEST[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0x2e, 0x7d, 0x2c, 0x03, 0xa9, 0x50, 0x7a, 0xe2,
    0x65, 0xec, 0xf5, 0xb5, 0x35, 0x68, 0x85, 0xa5,
    0x33, 0x93, 0xa2, 0x02, 0x9d, 0x24, 0x13, 0x94,
    0x99, 0x72, 0x65, 0xa1, 0xa2, 0x5a, 0xef, 0xc6,
};
static const unsigned char RUNTIME_SECOND_DIGEST[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0x18, 0xac, 0x3e, 0x73, 0x43, 0xf0, 0x16, 0x89,
    0x0c, 0x51, 0x0e, 0x93, 0xf9, 0x35, 0x26, 0x11,
    0x69, 0xd9, 0xe3, 0xf5, 0x65, 0x43, 0x64, 0x29,
    0x83, 0x0f, 0xaf, 0x09, 0x34, 0xf4, 0xf8, 0xe4,
};
static const unsigned char RUNTIME_LARGE_DIGEST[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0x68, 0xe3, 0xa0, 0x9e, 0x97, 0xaf, 0xe1, 0xdc,
    0x99, 0x52, 0x5a, 0x1e, 0x56, 0x9b, 0x65, 0x3d,
    0x70, 0x4b, 0xcb, 0x54, 0x3c, 0xba, 0xe0, 0xb9,
    0xb9, 0x26, 0x30, 0x72, 0x36, 0xf9, 0x07, 0xd1,
};

static bool pread_mutation_enabled;
static bool pread_mutation_fired;
static int pread_mutation_descriptor = -1;
static struct stat pread_mutation_target;

ssize_t __real_pread(int descriptor, void *buffer, size_t count, off_t offset);
ssize_t __wrap_pread(int descriptor, void *buffer, size_t count, off_t offset);
ssize_t __wrap_pread64(int descriptor, void *buffer, size_t count, off_t offset);
ssize_t __wrap___pread_chk(
    int descriptor,
    void *buffer,
    size_t count,
    off_t offset,
    size_t buffer_length
);
ssize_t __wrap___pread64_chk(
    int descriptor,
    void *buffer,
    size_t count,
    off_t offset,
    size_t buffer_length
);

static void write_repeated_file(
    const char *const path,
    const char byte,
    size_t length
) {
    unsigned char buffer[1024];
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);

    assert(descriptor >= 3);
    memset(buffer, (unsigned char)byte, sizeof(buffer));
    while (length != 0U) {
        const size_t chunk = length < sizeof(buffer) ? length : sizeof(buffer);

        assert(write(descriptor, buffer, chunk) == (ssize_t)chunk);
        length -= chunk;
    }
    assert(fsync(descriptor) == 0);
    assert(close(descriptor) == 0);
}

static void write_exact_file(const char *const path, const char byte) {
    write_repeated_file(path, byte, 1U);
}

static void overwrite_exact_file(const char *const path, const char byte) {
    const int descriptor = open(path, O_WRONLY | O_TRUNC | O_CLOEXEC);

    assert(descriptor >= 3);
    assert(write(descriptor, &byte, 1U) == 1);
    assert(fsync(descriptor) == 0);
    assert(close(descriptor) == 0);
}

static void mutate_after_target_pread(const int descriptor, const ssize_t result) {
    if (pread_mutation_enabled && !pread_mutation_fired && result > 0) {
        struct stat metadata;

        assert(fstat(descriptor, &metadata) == 0);
        if (metadata.st_dev == pread_mutation_target.st_dev &&
            metadata.st_ino == pread_mutation_target.st_ino) {
            assert(fchmod(pread_mutation_descriptor, (mode_t)0640) == 0);
            pread_mutation_fired = true;
        }
    }
}

ssize_t __wrap_pread(int descriptor, void *buffer, size_t count, off_t offset) {
    const ssize_t result = __real_pread(descriptor, buffer, count, offset);

    mutate_after_target_pread(descriptor, result);
    return result;
}

ssize_t __wrap_pread64(int descriptor, void *buffer, size_t count, off_t offset) {
    return __wrap_pread(descriptor, buffer, count, offset);
}

ssize_t __wrap___pread_chk(
    int descriptor,
    void *buffer,
    size_t count,
    off_t offset,
    size_t buffer_length
) {
    assert(count <= buffer_length);
    return __wrap_pread(descriptor, buffer, count, offset);
}

ssize_t __wrap___pread64_chk(
    int descriptor,
    void *buffer,
    size_t count,
    off_t offset,
    size_t buffer_length
) {
    assert(count <= buffer_length);
    return __wrap_pread64(descriptor, buffer, count, offset);
}

static void arm_pread_mutation(
    const int target_descriptor,
    const int mutation_descriptor
) {
    assert(fstat(target_descriptor, &pread_mutation_target) == 0);
    pread_mutation_enabled = true;
    pread_mutation_fired = false;
    pread_mutation_descriptor = mutation_descriptor;
}

static void disarm_pread_mutation(void) {
    pread_mutation_enabled = false;
    pread_mutation_fired = false;
    pread_mutation_descriptor = -1;
    memset(&pread_mutation_target, 0, sizeof(pread_mutation_target));
}

static struct fixture make_fixture(void) {
    struct fixture fixture = {0};
    char template[] = "/tmp/riley-execution-closure-held-fds.XXXXXX";
    char *const root = mkdtemp(template);
    int written;

    assert(root != NULL);
    assert(strlen(root) < sizeof(fixture.root));
    memcpy(fixture.root, root, strlen(root) + 1U);
    written = snprintf(fixture.loader, sizeof(fixture.loader), "%s/loader", fixture.root);
    assert(written > 0 && (size_t)written < sizeof(fixture.loader));
    written = snprintf(
        fixture.interpreter, sizeof(fixture.interpreter), "%s/interpreter", fixture.root
    );
    assert(written > 0 && (size_t)written < sizeof(fixture.interpreter));
    written = snprintf(fixture.runtime, sizeof(fixture.runtime), "%s/runtime", fixture.root);
    assert(written > 0 && (size_t)written < sizeof(fixture.runtime));
    written = snprintf(
        fixture.runtime_second, sizeof(fixture.runtime_second), "%s/runtime-second", fixture.root
    );
    assert(written > 0 && (size_t)written < sizeof(fixture.runtime_second));
    written = snprintf(
        fixture.runtime_large, sizeof(fixture.runtime_large), "%s/runtime-large", fixture.root
    );
    assert(written > 0 && (size_t)written < sizeof(fixture.runtime_large));
    write_exact_file(fixture.loader, 'a');
    write_exact_file(fixture.interpreter, 'b');
    write_exact_file(fixture.runtime, 'c');
    write_exact_file(fixture.runtime_second, 'd');
    write_repeated_file(fixture.runtime_large, 'e', LARGE_RUNTIME_BYTE_LENGTH);
    return fixture;
}

static void destroy_fixture(const struct fixture *const fixture) {
    (void)unlink(fixture->loader);
    (void)unlink(fixture->interpreter);
    (void)unlink(fixture->runtime);
    (void)unlink(fixture->runtime_second);
    (void)unlink(fixture->runtime_large);
    (void)rmdir(fixture->root);
}

static int open_borrowed_file(const char *const path) {
    const int descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);

    assert(descriptor >= 3);
    return descriptor;
}

static void populate_borrowed_with_runtime(
    const struct fixture *const fixture,
    const char *const runtime_path,
    int runtime_descriptors[1],
    struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed
) {
    borrowed->dynamic_loader_descriptor = open_borrowed_file(fixture->loader);
    borrowed->interpreter_descriptor = open_borrowed_file(fixture->interpreter);
    runtime_descriptors[0] = open_borrowed_file(runtime_path);
    borrowed->runtime_leaf_descriptors = runtime_descriptors;
    borrowed->runtime_leaf_count = 1U;
}

static void populate_borrowed(
    const struct fixture *const fixture,
    int runtime_descriptors[1],
    struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed
) {
    populate_borrowed_with_runtime(fixture, fixture->runtime, runtime_descriptors, borrowed);
}

static void populate_two_runtime_borrowed(
    const struct fixture *const fixture,
    int runtime_descriptors[2],
    struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed
) {
    borrowed->dynamic_loader_descriptor = open_borrowed_file(fixture->loader);
    borrowed->interpreter_descriptor = open_borrowed_file(fixture->interpreter);
    runtime_descriptors[0] = open_borrowed_file(fixture->runtime);
    runtime_descriptors[1] = open_borrowed_file(fixture->runtime_second);
    borrowed->runtime_leaf_descriptors = runtime_descriptors;
    borrowed->runtime_leaf_count = 2U;
}

static void close_borrowed(const struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed) {
    assert(close(borrowed->dynamic_loader_descriptor) == 0);
    assert(close(borrowed->interpreter_descriptor) == 0);
    for (size_t index = 0U; index < borrowed->runtime_leaf_count; ++index) {
        assert(close(borrowed->runtime_leaf_descriptors[index]) == 0);
    }
}

static size_t make_manifest_one(
    unsigned char *const destination,
    const size_t capacity,
    const char *const runtime_audit_path,
    const uint64_t runtime_byte_length,
    const char *const runtime_digest
) {
    const int written = snprintf(
        (char *)destination,
        capacity,
        "{\"dynamic_loader\":{\"audit_path\":\"/future/loader\",\"byte_length\":1,"
        "\"sha256\":\"%s\"},\"interpreter\":{\"audit_path\":\"/future/interpreter\","
        "\"byte_length\":1,\"sha256\":\"%s\"},\"runtime_leaves\":[{\"audit_path\":"
        "\"%s\",\"byte_length\":%" PRIu64 ",\"sha256\":\"%s\"}],"
        "\"schema_version\":\"riley.rc3-gate-e-execution-closure-manifest.v1\"}\n",
        LOADER_DIGEST_HEX,
        INTERPRETER_DIGEST_HEX,
        runtime_audit_path,
        runtime_byte_length,
        runtime_digest
    );

    assert(written > 0 && (size_t)written < capacity);
    return (size_t)written;
}

static size_t make_manifest_two(unsigned char *const destination, const size_t capacity) {
    const int written = snprintf(
        (char *)destination,
        capacity,
        "{\"dynamic_loader\":{\"audit_path\":\"/future/loader\",\"byte_length\":1,"
        "\"sha256\":\"%s\"},\"interpreter\":{\"audit_path\":\"/future/interpreter\","
        "\"byte_length\":1,\"sha256\":\"%s\"},\"runtime_leaves\":[{\"audit_path\":"
        "\"/future/runtime-a\",\"byte_length\":1,\"sha256\":\"%s\"},{\"audit_path\":"
        "\"/future/runtime-b\",\"byte_length\":1,\"sha256\":\"%s\"}],"
        "\"schema_version\":\"riley.rc3-gate-e-execution-closure-manifest.v1\"}\n",
        LOADER_DIGEST_HEX,
        INTERPRETER_DIGEST_HEX,
        RUNTIME_DIGEST_HEX,
        RUNTIME_SECOND_DIGEST_HEX
    );

    assert(written > 0 && (size_t)written < capacity);
    return (size_t)written;
}

static void manifest_digest(
    const unsigned char *const raw,
    const size_t raw_length,
    unsigned char digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    struct gate_e_execution_closure_manifest parsed;

    gate_e_execution_closure_manifest_v1_init(&parsed);
    assert(gate_e_parse_execution_closure_manifest_v1(raw, raw_length, &parsed) ==
           GATE_E_EXECUTION_CLOSURE_OK);
    memcpy(digest, parsed.runtime_closure_sha256, GATE_E_EXECUTION_CLOSURE_SHA256_BYTES);
}

static void assert_zero_bytes(const unsigned char *const bytes, const size_t length) {
    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_empty(const struct gate_e_execution_closure_held_fds_v1 *const output) {
    struct gate_e_execution_closure_held_fds_v1 expected;

    gate_e_execution_closure_held_fds_v1_init(&expected);
    assert(memcmp(output, &expected, sizeof(expected)) == 0);
}

static void assert_identity_matches_descriptor(
    const struct gate_e_execution_closure_held_file_v1 *const file
) {
    struct stat metadata;

    assert(fstat(file->descriptor, &metadata) == 0);
    assert(file->identity.device == metadata.st_dev);
    assert(file->identity.inode == metadata.st_ino);
    assert(file->identity.mode == metadata.st_mode);
    assert(file->identity.links == metadata.st_nlink);
    assert(file->identity.uid == metadata.st_uid);
    assert(file->identity.gid == metadata.st_gid);
    assert(file->identity.size == metadata.st_size);
    assert(file->identity.mtime.tv_sec == metadata.st_mtim.tv_sec);
    assert(file->identity.mtime.tv_nsec == metadata.st_mtim.tv_nsec);
    assert(file->identity.ctime.tv_sec == metadata.st_ctim.tv_sec);
    assert(file->identity.ctime.tv_nsec == metadata.st_ctim.tv_nsec);
}

static void assert_held_file(
    const struct gate_e_execution_closure_held_file_v1 *const file,
    const char *const expected_audit_path,
    const char expected_byte,
    const uint64_t expected_byte_length,
    const unsigned char expected_digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    char observed = '\0';
    const size_t audit_path_length = strlen(expected_audit_path);
    const int descriptor_flags = fcntl(file->descriptor, F_GETFD);

    assert(audit_path_length < sizeof(file->declaration.audit_path));
    assert(file->descriptor >= 3);
    assert(descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0);
    assert(file->declaration.audit_path_length == audit_path_length);
    assert(memcmp(
               file->declaration.audit_path, expected_audit_path, audit_path_length + 1U
           ) == 0);
    assert_zero_bytes(
        file->declaration.audit_path + audit_path_length + 1U,
        sizeof(file->declaration.audit_path) - audit_path_length - 1U
    );
    assert(file->declaration.byte_length == expected_byte_length);
    assert(memcmp(file->declaration.sha256, expected_digest, sizeof(file->declaration.sha256)) == 0);
    assert_identity_matches_descriptor(file);
    assert(pread(file->descriptor, &observed, 1U, 0) == 1);
    assert(observed == expected_byte);
}

static void test_success_preserves_inputs_and_rechecks(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    unsigned char expected_closure_digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
    int runtime_descriptors[1];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_held_fds_v1 output;
    const size_t raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );
    int loader_flags_before;
    int interpreter_flags_before;
    int runtime_flags_before;
    off_t loader_offset_before;
    off_t interpreter_offset_before;
    off_t runtime_offset_before;
    int output_loader_descriptor;

    populate_borrowed(&fixture, runtime_descriptors, &borrowed);
    manifest_digest(raw, raw_length, expected_closure_digest);
    loader_flags_before = fcntl(borrowed.dynamic_loader_descriptor, F_GETFD);
    interpreter_flags_before = fcntl(borrowed.interpreter_descriptor, F_GETFD);
    runtime_flags_before = fcntl(borrowed.runtime_leaf_descriptors[0], F_GETFD);
    loader_offset_before = lseek(borrowed.dynamic_loader_descriptor, 0, SEEK_CUR);
    interpreter_offset_before = lseek(borrowed.interpreter_descriptor, 0, SEEK_CUR);
    runtime_offset_before = lseek(borrowed.runtime_leaf_descriptors[0], 0, SEEK_CUR);
    assert(loader_flags_before >= 0 && interpreter_flags_before >= 0 && runtime_flags_before >= 0);
    assert(loader_offset_before >= 0 && interpreter_offset_before >= 0 && runtime_offset_before >= 0);
    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert(output.runtime_leaf_count == 1U);
    assert(memcmp(
               output.runtime_closure_sha256,
               expected_closure_digest,
               sizeof(expected_closure_digest)
           ) == 0);
    assert_held_file(
        &output.dynamic_loader, "/future/loader", 'a', UINT64_C(1), LOADER_DIGEST
    );
    assert_held_file(
        &output.interpreter, "/future/interpreter", 'b', UINT64_C(1), INTERPRETER_DIGEST
    );
    assert_held_file(
        &output.runtime_leaves[0], "/future/runtime", 'c', UINT64_C(1), RUNTIME_DIGEST
    );
    assert(output.dynamic_loader.descriptor != borrowed.dynamic_loader_descriptor);
    assert(output.interpreter.descriptor != borrowed.interpreter_descriptor);
    assert(output.runtime_leaves[0].descriptor != borrowed.runtime_leaf_descriptors[0]);
    assert(fcntl(borrowed.dynamic_loader_descriptor, F_GETFD) == loader_flags_before);
    assert(fcntl(borrowed.interpreter_descriptor, F_GETFD) == interpreter_flags_before);
    assert(fcntl(borrowed.runtime_leaf_descriptors[0], F_GETFD) == runtime_flags_before);
    assert(lseek(borrowed.dynamic_loader_descriptor, 0, SEEK_CUR) == loader_offset_before);
    assert(lseek(borrowed.interpreter_descriptor, 0, SEEK_CUR) == interpreter_offset_before);
    assert(lseek(borrowed.runtime_leaf_descriptors[0], 0, SEEK_CUR) == runtime_offset_before);
    assert(gate_e_execution_closure_held_fds_v1_recheck(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    output_loader_descriptor = output.dynamic_loader.descriptor;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1);
    assert(output.dynamic_loader.descriptor == output_loader_descriptor);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_empty(&output);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

static void test_output_owns_duplicates_after_input_close(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    int runtime_descriptors[1];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_held_fds_v1 output;
    const size_t raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );

    populate_borrowed(&fixture, runtime_descriptors, &borrowed);
    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    close_borrowed(&borrowed);
    assert_held_file(
        &output.dynamic_loader, "/future/loader", 'a', UINT64_C(1), LOADER_DIGEST
    );
    assert_held_file(
        &output.interpreter, "/future/interpreter", 'b', UINT64_C(1), INTERPRETER_DIGEST
    );
    assert_held_file(
        &output.runtime_leaves[0], "/future/runtime", 'c', UINT64_C(1), RUNTIME_DIGEST
    );
    assert(gate_e_execution_closure_held_fds_v1_recheck(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_empty(&output);
    destroy_fixture(&fixture);
}

static void test_runtime_role_order(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    int runtime_descriptors[2];
    int swapped_runtime_descriptors[2];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_borrowed_fds_v1 bad;
    struct gate_e_execution_closure_held_fds_v1 output;
    const size_t raw_length = make_manifest_two(raw, sizeof(raw));
    int swapped_fixed_descriptor;

    populate_two_runtime_borrowed(&fixture, runtime_descriptors, &borrowed);
    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert(output.runtime_leaf_count == 2U);
    assert_held_file(
        &output.runtime_leaves[0], "/future/runtime-a", 'c', UINT64_C(1), RUNTIME_DIGEST
    );
    assert_held_file(
        &output.runtime_leaves[1],
        "/future/runtime-b",
        'd',
        UINT64_C(1),
        RUNTIME_SECOND_DIGEST
    );
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_empty(&output);

    bad = borrowed;
    swapped_fixed_descriptor = bad.dynamic_loader_descriptor;
    bad.dynamic_loader_descriptor = bad.interpreter_descriptor;
    bad.interpreter_descriptor = swapped_fixed_descriptor;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1);
    assert_empty(&output);

    swapped_runtime_descriptors[0] = borrowed.runtime_leaf_descriptors[1];
    swapped_runtime_descriptors[1] = borrowed.runtime_leaf_descriptors[0];
    bad = borrowed;
    bad.runtime_leaf_descriptors = swapped_runtime_descriptors;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1);
    assert_empty(&output);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

static void test_hashes_multiple_reads(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    int runtime_descriptors[1];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_held_fds_v1 output;
    const size_t raw_length = make_manifest_one(
        raw,
        sizeof(raw),
        "/future/runtime-large",
        (uint64_t)LARGE_RUNTIME_BYTE_LENGTH,
        RUNTIME_LARGE_DIGEST_HEX
    );

    populate_borrowed_with_runtime(
        &fixture, fixture.runtime_large, runtime_descriptors, &borrowed
    );
    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_held_file(
        &output.runtime_leaves[0],
        "/future/runtime-large",
        'e',
        (uint64_t)LARGE_RUNTIME_BYTE_LENGTH,
        RUNTIME_LARGE_DIGEST
    );
    assert(gate_e_execution_closure_held_fds_v1_recheck(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_empty(&output);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

static void test_rejects_invalid_manifest_and_borrowed_shapes(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    unsigned char alias_raw[2048];
    unsigned char digest_mismatch_raw[2048];
    int runtime_descriptors[1];
    int alias_runtime_descriptors[1];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_borrowed_fds_v1 bad;
    struct gate_e_execution_closure_held_fds_v1 output;
    size_t raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );
    const size_t alias_raw_length = make_manifest_one(
        alias_raw, sizeof(alias_raw), "/future/runtime", UINT64_C(1), LOADER_DIGEST_HEX
    );
    const size_t digest_mismatch_raw_length = make_manifest_one(
        digest_mismatch_raw,
        sizeof(digest_mismatch_raw),
        "/future/runtime",
        UINT64_C(1),
        INTERPRETER_DIGEST_HEX
    );
    int descriptor_flags;
    int writable_descriptor;
    int append_descriptor;
    int directory_descriptor;
    int pipe_descriptors[2];

    populate_borrowed(&fixture, runtime_descriptors, &borrowed);
    gate_e_execution_closure_held_fds_v1_init(&output);
    raw[raw_length - 1U] = (unsigned char)' ';
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_MANIFEST_REJECTED_V1);
    assert_empty(&output);
    raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );
    bad = borrowed;
    bad.runtime_leaf_count = 0U;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_RUNTIME_COUNT_MISMATCH_V1);
    assert_empty(&output);
    alias_runtime_descriptors[0] = borrowed.dynamic_loader_descriptor;
    bad = borrowed;
    bad.runtime_leaf_descriptors = alias_runtime_descriptors;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_ALIAS_V1);
    assert_empty(&output);
    descriptor_flags = fcntl(borrowed.dynamic_loader_descriptor, F_GETFD);
    assert(descriptor_flags >= 0);
    assert(fcntl(
               borrowed.dynamic_loader_descriptor, F_SETFD, descriptor_flags & ~FD_CLOEXEC
           ) == 0);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    assert(fcntl(borrowed.dynamic_loader_descriptor, F_SETFD, descriptor_flags) == 0);
    bad = borrowed;
    bad.runtime_leaf_descriptors = NULL;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1);
    assert_empty(&output);
    writable_descriptor = open(fixture.loader, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    assert(writable_descriptor >= 3);
    bad = borrowed;
    bad.dynamic_loader_descriptor = writable_descriptor;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    assert(close(writable_descriptor) == 0);
    append_descriptor = open(fixture.loader, O_RDONLY | O_APPEND | O_CLOEXEC | O_NOFOLLOW);
    assert(append_descriptor >= 3);
    bad = borrowed;
    bad.dynamic_loader_descriptor = append_descriptor;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    assert(close(append_descriptor) == 0);
#ifdef O_DIRECT
    {
        const int direct_descriptor = open(
            fixture.loader, O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW
        );

        if (direct_descriptor >= 3) {
            bad = borrowed;
            bad.dynamic_loader_descriptor = direct_descriptor;
            assert(gate_e_bind_execution_closure_held_fds_v1(
                       raw, raw_length, &bad, &output
                   ) == GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
            assert_empty(&output);
            assert(close(direct_descriptor) == 0);
        } else {
            assert(direct_descriptor == -1);
        }
    }
#endif
#ifdef O_PATH
    {
        const int path_descriptor = open(fixture.loader, O_PATH | O_CLOEXEC | O_NOFOLLOW);

        assert(path_descriptor >= 3);
        bad = borrowed;
        bad.dynamic_loader_descriptor = path_descriptor;
        assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
               GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
        assert_empty(&output);
        assert(close(path_descriptor) == 0);
    }
#endif
    directory_descriptor = open(fixture.root, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    assert(directory_descriptor >= 3);
    bad = borrowed;
    bad.dynamic_loader_descriptor = directory_descriptor;
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    assert(close(directory_descriptor) == 0);
    assert(pipe2(pipe_descriptors, O_CLOEXEC) == 0);
    bad = borrowed;
    bad.dynamic_loader_descriptor = pipe_descriptors[0];
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &bad, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    assert(close(pipe_descriptors[0]) == 0);
    assert(close(pipe_descriptors[1]) == 0);
    alias_runtime_descriptors[0] = open_borrowed_file(fixture.loader);
    bad = borrowed;
    bad.runtime_leaf_descriptors = alias_runtime_descriptors;
    assert(gate_e_bind_execution_closure_held_fds_v1(
               alias_raw, alias_raw_length, &bad, &output
           ) == GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_ALIAS_V1);
    assert_empty(&output);
    assert(close(alias_runtime_descriptors[0]) == 0);
    assert(gate_e_bind_execution_closure_held_fds_v1(
               digest_mismatch_raw, digest_mismatch_raw_length, &borrowed, &output
           ) == GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1);
    assert_empty(&output);
    assert(unlink(fixture.loader) == 0);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1);
    assert_empty(&output);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

static void test_recheck_rejects_object_drift_and_uninitialized_output(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    int runtime_descriptors[1];
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_held_fds_v1 output;
    struct gate_e_execution_closure_held_fds_v1 uninitialized = {0};
    const size_t raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );
    enum gate_e_execution_closure_held_fds_reason_v1 result;

    populate_borrowed(&fixture, runtime_descriptors, &borrowed);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &uninitialized) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1);
    assert(uninitialized.initialized_state == 0U);
    gate_e_execution_closure_held_fds_v1_init(&output);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    overwrite_exact_file(fixture.runtime, 'z');
    result = gate_e_execution_closure_held_fds_v1_recheck(&output);
    assert(result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1 ||
           result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1);
    assert(output.runtime_leaves[0].descriptor >= 3);
    assert(gate_e_execution_closure_held_fds_v1_close(&output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1);
    assert_empty(&output);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

static void test_bind_rejects_identity_drift(void) {
    const struct fixture fixture = make_fixture();
    unsigned char raw[2048];
    int runtime_descriptors[1];
    const int mutation_descriptor = open(fixture.runtime, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    struct gate_e_execution_closure_borrowed_fds_v1 borrowed;
    struct gate_e_execution_closure_held_fds_v1 output;
    const size_t raw_length = make_manifest_one(
        raw, sizeof(raw), "/future/runtime", UINT64_C(1), RUNTIME_DIGEST_HEX
    );
    int runtime_flags_before;
    off_t runtime_offset_before;

    assert(mutation_descriptor >= 3);
    populate_borrowed(&fixture, runtime_descriptors, &borrowed);
    runtime_flags_before = fcntl(borrowed.runtime_leaf_descriptors[0], F_GETFD);
    runtime_offset_before = lseek(borrowed.runtime_leaf_descriptors[0], 0, SEEK_CUR);
    assert(runtime_flags_before >= 0 && runtime_offset_before >= 0);
    gate_e_execution_closure_held_fds_v1_init(&output);
    arm_pread_mutation(borrowed.runtime_leaf_descriptors[0], mutation_descriptor);
    assert(gate_e_bind_execution_closure_held_fds_v1(raw, raw_length, &borrowed, &output) ==
           GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1);
    assert(pread_mutation_fired);
    disarm_pread_mutation();
    assert(fcntl(borrowed.runtime_leaf_descriptors[0], F_GETFD) == runtime_flags_before);
    assert(lseek(borrowed.runtime_leaf_descriptors[0], 0, SEEK_CUR) == runtime_offset_before);
    assert_empty(&output);
    assert(close(mutation_descriptor) == 0);
    close_borrowed(&borrowed);
    destroy_fixture(&fixture);
}

int main(void) {
    test_success_preserves_inputs_and_rechecks();
    test_output_owns_duplicates_after_input_close();
    test_runtime_role_order();
    test_hashes_multiple_reads();
    test_rejects_invalid_manifest_and_borrowed_shapes();
    test_recheck_rejects_object_drift_and_uninitialized_output();
    test_bind_rejects_identity_drift();
    (void)puts("gate-e execution closure held-fds tests passed");
    return 0;
}
