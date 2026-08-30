/*
 * Linux-only, no-action object inspection for the future RC3 Gate E native
 * guardian.  This source is deliberately an excluded C11 tool: it is not a
 * runtime component, installed guardian, launcher, service, or authority to
 * execute Python, a GPU workload, Docker, or a qualification campaign.
 *
 * Its production invocation has no caller-controlled paths.  The only tree
 * it can observe is the future guardian audit bundle below /opt/riley.  It
 * uses held descriptors throughout one read-only check.  Its CLI closes them
 * before reporting; its versioned source-library API can return the same
 * verified CLOEXEC descriptors to a later, separately reviewed static root
 * guardian.  That future guardian must still authenticate its own
 * loader/interpreter/runtime closure and perform the same-object secure-exec,
 * ledger, cgroup, and pidfd work.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#include "gate_e_root_bundle_held_v1.h"

enum {
    MAX_MANIFEST_BYTES = 64 * 1024,
    MAX_CODE_BYTES = 2 * 1024 * 1024,
    SHA256_BLOCK_BYTES = 64,
    SHA256_HEX_BYTES = 64,
};

#define SHA256_DIGEST_BYTES GATE_E_ROOT_BUNDLE_SHA256_DIGEST_BYTES_V1
#define BUNDLE_DIRECTORY_COUNT GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1

/* Keep the implementation's short names private while exposing versioned ABI. */
#define anchor_reason gate_e_root_bundle_reason_v1
#define ANCHOR_OK GATE_E_ROOT_BUNDLE_OK_V1
#define ANCHOR_NOT_ROOT GATE_E_ROOT_BUNDLE_NOT_ROOT_V1
#define ANCHOR_OPENAT2_UNAVAILABLE GATE_E_ROOT_BUNDLE_OPENAT2_UNAVAILABLE_V1
#define ANCHOR_ROOT_UNREADABLE GATE_E_ROOT_BUNDLE_ROOT_UNREADABLE_V1
#define ANCHOR_UNSAFE_DIRECTORY GATE_E_ROOT_BUNDLE_UNSAFE_DIRECTORY_V1
#define ANCHOR_UNSAFE_FILESYSTEM GATE_E_ROOT_BUNDLE_UNSAFE_FILESYSTEM_V1
#define ANCHOR_ACL_PRESENT GATE_E_ROOT_BUNDLE_ACL_PRESENT_V1
#define ANCHOR_ACL_UNVERIFIABLE GATE_E_ROOT_BUNDLE_ACL_UNVERIFIABLE_V1
#define ANCHOR_CAPABILITY_PRESENT GATE_E_ROOT_BUNDLE_CAPABILITY_PRESENT_V1
#define ANCHOR_UNSAFE_FILE GATE_E_ROOT_BUNDLE_UNSAFE_FILE_V1
#define ANCHOR_FILE_UNREADABLE GATE_E_ROOT_BUNDLE_FILE_UNREADABLE_V1
#define ANCHOR_OBJECT_RACED GATE_E_ROOT_BUNDLE_OBJECT_RACED_V1
#define ANCHOR_MANIFEST_INVALID GATE_E_ROOT_BUNDLE_MANIFEST_INVALID_V1
#define ANCHOR_DIGEST_MISMATCH GATE_E_ROOT_BUNDLE_DIGEST_MISMATCH_V1
#define ANCHOR_CLOSE_FAILED GATE_E_ROOT_BUNDLE_CLOSE_FAILED_V1
#define ANCHOR_INVALID_ARGUMENT GATE_E_ROOT_BUNDLE_INVALID_ARGUMENT_V1
#define object_identity gate_e_root_bundle_object_identity_v1
#define held_directory gate_e_root_bundle_held_directory_v1
#define held_file gate_e_root_bundle_held_file_v1
#define bundle_handles gate_e_root_bundle_held_v1

static const char *const ROOT_BUNDLE_COMPONENTS[BUNDLE_DIRECTORY_COUNT] = {
    "/", "opt", "riley", "rc3-gate-e-v1",
};
static const char *const MANIFEST_NAME = "gate-e-v3.manifest.json";
static const char *const BOOTSTRAP_NAME = "rc3_gate_e_guardian_bootstrap_v1.py";
static const char *const CORE_NAME = "rc3_gate_e_guardian_no_action_core_v1.py";
static const char *const MANIFEST_SCHEMA = "riley.rc3-gate-e-root-bundle.v1";

struct manifest_spec {
    char bootstrap_sha256[SHA256_HEX_BYTES + 1];
    size_t bootstrap_byte_length;
    char core_sha256[SHA256_HEX_BYTES + 1];
    size_t core_byte_length;
};

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[SHA256_BLOCK_BYTES];
    size_t block_length;
};

typedef int (*openat2_invoker)(int, const char *, const struct open_how *);
typedef int (*statfs_invoker)(int, struct statfs *);
typedef ssize_t (*xattr_invoker)(int, const char *, void *, size_t);

static int linux_openat2(const int directory_fd, const char *const path, const struct open_how *const how) {
#ifdef SYS_openat2
    return (int)syscall(SYS_openat2, directory_fd, path, how, sizeof(*how));
#else
    (void)directory_fd;
    (void)path;
    (void)how;
    errno = ENOSYS;
    return -1;
#endif
}

static openat2_invoker invoke_openat2 = linux_openat2;
static statfs_invoker invoke_fstatfs = fstatfs;
static xattr_invoker invoke_fgetxattr = fgetxattr;

static uint32_t rotate_right(const uint32_t value, const unsigned int shift) {
    return (value >> shift) | (value << (32U - shift));
}

static uint32_t load_be32(const unsigned char *const source) {
    return ((uint32_t)source[0] << 24U) | ((uint32_t)source[1] << 16U) |
           ((uint32_t)source[2] << 8U) | (uint32_t)source[3];
}

static void store_be32(unsigned char *const destination, const uint32_t value) {
    destination[0] = (unsigned char)(value >> 24U);
    destination[1] = (unsigned char)(value >> 16U);
    destination[2] = (unsigned char)(value >> 8U);
    destination[3] = (unsigned char)value;
}

static void sha256_transform(struct sha256_context *const context, const unsigned char block[SHA256_BLOCK_BYTES]) {
    static const uint32_t round_constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    uint32_t words[64] = {0};
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;

    for (size_t index = 0; index < 16; ++index) {
        words[index] = load_be32(block + (index * 4));
    }
    for (size_t index = 16; index < 64; ++index) {
        const uint32_t s0 = rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^
                            (words[index - 15] >> 3U);
        const uint32_t s1 = rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^
                            (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (size_t index = 0; index < 64; ++index) {
        const uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temporary1 = h + sigma1 + choose + round_constants[index] + words[index];
        const uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temporary2 = sigma0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(struct sha256_context *const context) {
    static const uint32_t initial_state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };

    memcpy(context->state, initial_state, sizeof(initial_state));
    context->bit_length = 0;
    context->block_length = 0;
}

static void sha256_update(struct sha256_context *const context, const unsigned char *data, size_t length) {
    while (length != 0) {
        const size_t available = SHA256_BLOCK_BYTES - context->block_length;
        const size_t chunk = length < available ? length : available;

        memcpy(context->block + context->block_length, data, chunk);
        context->block_length += chunk;
        data += chunk;
        length -= chunk;
        if (context->block_length == SHA256_BLOCK_BYTES) {
            sha256_transform(context, context->block);
            context->bit_length += UINT64_C(512);
            context->block_length = 0;
        }
    }
}

static void sha256_final(struct sha256_context *const context, unsigned char digest[SHA256_DIGEST_BYTES]) {
    const uint64_t final_bits = context->bit_length + ((uint64_t)context->block_length * UINT64_C(8));
    size_t index = context->block_length;

    context->block[index++] = 0x80U;
    if (index > 56) {
        while (index < SHA256_BLOCK_BYTES) {
            context->block[index++] = 0;
        }
        sha256_transform(context, context->block);
        index = 0;
    }
    while (index < 56) {
        context->block[index++] = 0;
    }
    for (size_t byte = 0; byte < 8; ++byte) {
        context->block[63 - byte] = (unsigned char)(final_bits >> (byte * 8U));
    }
    sha256_transform(context, context->block);
    for (size_t word = 0; word < 8; ++word) {
        store_be32(digest + (word * 4), context->state[word]);
    }
}

static void digest_to_hex(const unsigned char digest[SHA256_DIGEST_BYTES], char output[SHA256_HEX_BYTES + 1]) {
    static const char hex[] = "0123456789abcdef";

    for (size_t index = 0; index < SHA256_DIGEST_BYTES; ++index) {
        output[index * 2] = hex[digest[index] >> 4U];
        output[(index * 2) + 1] = hex[digest[index] & 0x0fU];
    }
    output[SHA256_HEX_BYTES] = '\0';
}

static struct object_identity identity_from_stat(const struct stat *const metadata) {
    return (struct object_identity){
        .device = metadata->st_dev,
        .inode = metadata->st_ino,
        .mode = metadata->st_mode,
        .links = metadata->st_nlink,
        .uid = metadata->st_uid,
        .gid = metadata->st_gid,
        .size = metadata->st_size,
        .mtime = metadata->st_mtim,
        .ctime = metadata->st_ctim,
    };
}

static bool identities_match(const struct object_identity *const left, const struct object_identity *const right) {
    return left->device == right->device && left->inode == right->inode && left->mode == right->mode &&
           left->links == right->links && left->uid == right->uid && left->gid == right->gid &&
           left->size == right->size && left->mtime.tv_sec == right->mtime.tv_sec &&
           left->mtime.tv_nsec == right->mtime.tv_nsec && left->ctime.tv_sec == right->ctime.tv_sec &&
           left->ctime.tv_nsec == right->ctime.tv_nsec;
}

static bool allowed_filesystem_type(const long filesystem_type) {
    return (unsigned long)filesystem_type == (unsigned long)EXT4_SUPER_MAGIC ||
           (unsigned long)filesystem_type == (unsigned long)XFS_SUPER_MAGIC ||
           (unsigned long)filesystem_type == (unsigned long)BTRFS_SUPER_MAGIC;
}

static enum anchor_reason require_approved_filesystem(const int descriptor) {
    struct statfs metadata;

    if (invoke_fstatfs(descriptor, &metadata) != 0 || !allowed_filesystem_type(metadata.f_type)) {
        return ANCHOR_UNSAFE_FILESYSTEM;
    }
    return ANCHOR_OK;
}

static enum anchor_reason require_absent_xattr(
    const int descriptor,
    const char *const name,
    const enum anchor_reason present_reason
) {
    const ssize_t length = invoke_fgetxattr(descriptor, name, NULL, 0);

    if (length == -1 && errno == ENODATA) {
        return ANCHOR_OK;
    }
    if (length >= 0) {
        return present_reason;
    }
    return ANCHOR_ACL_UNVERIFIABLE;
}

static enum anchor_reason require_acl_free(const int descriptor, const bool directory) {
    enum anchor_reason reason = require_absent_xattr(
        descriptor, "system.posix_acl_access", ANCHOR_ACL_PRESENT
    );

    if (reason != ANCHOR_OK || !directory) {
        return reason;
    }
    return require_absent_xattr(descriptor, "system.posix_acl_default", ANCHOR_ACL_PRESENT);
}

static bool has_only_safe_mode_bits(const mode_t mode) {
    return (mode & (S_IWGRP | S_IWOTH | S_ISUID | S_ISGID | S_ISVTX)) == 0;
}

static enum anchor_reason check_directory_metadata(
    const struct stat *const metadata,
    const uid_t expected_uid,
    const gid_t expected_gid,
    const mode_t expected_mode
) {
    if (!S_ISDIR(metadata->st_mode) || metadata->st_uid != expected_uid ||
        metadata->st_gid != expected_gid || !has_only_safe_mode_bits(metadata->st_mode) ||
        (mode_t)(metadata->st_mode & 07777U) != expected_mode) {
        return ANCHOR_UNSAFE_DIRECTORY;
    }
    return ANCHOR_OK;
}

static enum anchor_reason check_regular_metadata(
    const struct stat *const metadata,
    const uid_t expected_uid,
    const gid_t expected_gid,
    const mode_t expected_mode,
    const size_t maximum_bytes
) {
    if (!S_ISREG(metadata->st_mode) || metadata->st_nlink != 1 || metadata->st_uid != expected_uid ||
        metadata->st_gid != expected_gid || !has_only_safe_mode_bits(metadata->st_mode) ||
        (mode_t)(metadata->st_mode & 07777U) != expected_mode || metadata->st_size <= 0 ||
        (uintmax_t)metadata->st_size > (uintmax_t)maximum_bytes) {
        return ANCHOR_UNSAFE_FILE;
    }
    return ANCHOR_OK;
}

static struct open_how directory_open_how(const bool root) {
    return (struct open_how){
        .flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOATIME | O_NOFOLLOW | O_NONBLOCK,
        .mode = 0,
        .resolve = root ? (RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS)
                        : (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
                           RESOLVE_NO_XDEV),
    };
}

static struct open_how file_open_how(void) {
    return (struct open_how){
        .flags = O_RDONLY | O_CLOEXEC | O_NOATIME | O_NOFOLLOW | O_NONBLOCK,
        .mode = 0,
        .resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV,
    };
}

static enum anchor_reason classify_openat2_failure(void) {
    return (errno == ENOSYS || errno == EINVAL) ? ANCHOR_OPENAT2_UNAVAILABLE : ANCHOR_ROOT_UNREADABLE;
}

static void initialize_bundle_handles(struct bundle_handles *const handles) {
    memset(handles, 0, sizeof(*handles));
    for (size_t index = 0; index < BUNDLE_DIRECTORY_COUNT; ++index) {
        handles->directories[index].descriptor = -1;
    }
    handles->manifest.descriptor = -1;
    handles->bootstrap.descriptor = -1;
    handles->core.descriptor = -1;
}

void gate_e_root_bundle_held_v1_init(struct gate_e_root_bundle_held_v1 *const held) {
    if (held != NULL) {
        initialize_bundle_handles(held);
    }
}

static enum anchor_reason close_bundle_handles(struct bundle_handles *const handles) {
    enum anchor_reason result = ANCHOR_OK;
    int *const file_descriptors[] = {
        &handles->core.descriptor,
        &handles->bootstrap.descriptor,
        &handles->manifest.descriptor,
    };

    for (size_t index = 0; index < sizeof(file_descriptors) / sizeof(file_descriptors[0]); ++index) {
        if (*file_descriptors[index] >= 0 && close(*file_descriptors[index]) != 0) {
            result = ANCHOR_CLOSE_FAILED;
        }
        *file_descriptors[index] = -1;
    }
    for (size_t index = BUNDLE_DIRECTORY_COUNT; index > 0; --index) {
        int *const descriptor = &handles->directories[index - 1].descriptor;
        if (*descriptor >= 0 && close(*descriptor) != 0) {
            result = ANCHOR_CLOSE_FAILED;
        }
        *descriptor = -1;
    }
    initialize_bundle_handles(handles);
    return result;
}

enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_held_v1_close(
    struct gate_e_root_bundle_held_v1 *const held
) {
    if (held == NULL) {
        return ANCHOR_INVALID_ARGUMENT;
    }
    return close_bundle_handles(held);
}

static enum anchor_reason inspect_opened_directory(
    const int descriptor,
    const uid_t expected_uid,
    const gid_t expected_gid,
    const mode_t expected_mode,
    struct object_identity *const identity
) {
    struct stat metadata;
    enum anchor_reason reason;

    if (fstat(descriptor, &metadata) != 0) {
        return ANCHOR_UNSAFE_DIRECTORY;
    }
    reason = check_directory_metadata(&metadata, expected_uid, expected_gid, expected_mode);
    if (reason != ANCHOR_OK) {
        return reason;
    }
    reason = require_approved_filesystem(descriptor);
    if (reason != ANCHOR_OK) {
        return reason;
    }
    reason = require_acl_free(descriptor, true);
    if (reason != ANCHOR_OK) {
        return reason;
    }
    *identity = identity_from_stat(&metadata);
    return ANCHOR_OK;
}

static enum anchor_reason open_root_directory(
    const uid_t expected_uid,
    const gid_t expected_gid,
    struct held_directory *const output
) {
    const struct open_how how = directory_open_how(true);
    int descriptor = -1;
    enum anchor_reason reason;

    descriptor = invoke_openat2(AT_FDCWD, ROOT_BUNDLE_COMPONENTS[0], &how);
    if (descriptor < 0) {
        return classify_openat2_failure();
    }
    reason = inspect_opened_directory(descriptor, expected_uid, expected_gid, 0755, &output->identity);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        return reason;
    }
    output->descriptor = descriptor;
    return ANCHOR_OK;
}

static enum anchor_reason open_child_directory(
    const struct held_directory *const parent,
    const char *const name,
    const uid_t expected_uid,
    const gid_t expected_gid,
    struct held_directory *const output
) {
    const struct open_how how = directory_open_how(false);
    struct stat before;
    struct stat after;
    struct object_identity before_identity;
    struct object_identity after_identity;
    int descriptor = -1;
    enum anchor_reason reason;

    if (fstatat(parent->descriptor, name, &before, AT_SYMLINK_NOFOLLOW) != 0) {
        return ANCHOR_ROOT_UNREADABLE;
    }
    reason = check_directory_metadata(&before, expected_uid, expected_gid, 0755);
    if (reason != ANCHOR_OK) {
        return reason;
    }
    descriptor = invoke_openat2(parent->descriptor, name, &how);
    if (descriptor < 0) {
        return classify_openat2_failure();
    }
    reason = inspect_opened_directory(descriptor, expected_uid, expected_gid, 0755, &output->identity);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        return reason;
    }
    before_identity = identity_from_stat(&before);
    if (!identities_match(&before_identity, &output->identity) ||
        fstatat(parent->descriptor, name, &after, AT_SYMLINK_NOFOLLOW) != 0) {
        (void)close(descriptor);
        return ANCHOR_OBJECT_RACED;
    }
    after_identity = identity_from_stat(&after);
    if (!identities_match(&before_identity, &after_identity)) {
        (void)close(descriptor);
        return ANCHOR_OBJECT_RACED;
    }
    output->descriptor = descriptor;
    return ANCHOR_OK;
}

static enum anchor_reason hash_opened_regular_file(
    const int descriptor,
    const size_t maximum_bytes,
    const bool collect,
    struct held_file *const output,
    unsigned char **const collected_raw
) {
    struct stat before;
    struct stat after;
    struct object_identity before_identity;
    struct object_identity after_identity;
    struct sha256_context digest;
    unsigned char buffer[8192];
    unsigned char *raw = NULL;
    size_t used = 0;
    enum anchor_reason result = ANCHOR_OK;

    if (fstat(descriptor, &before) != 0 || before.st_size <= 0 ||
        (uintmax_t)before.st_size > (uintmax_t)maximum_bytes) {
        return ANCHOR_UNSAFE_FILE;
    }
    if (collect) {
        raw = calloc((size_t)before.st_size + 1U, sizeof(*raw));
        if (raw == NULL) {
            return ANCHOR_FILE_UNREADABLE;
        }
    }
    sha256_init(&digest);
    for (;;) {
        const ssize_t count = read(descriptor, buffer, sizeof(buffer));
        if (count < 0) {
            result = ANCHOR_FILE_UNREADABLE;
            break;
        }
        if (count == 0) {
            break;
        }
        if ((size_t)count > maximum_bytes - used) {
            result = ANCHOR_UNSAFE_FILE;
            break;
        }
        if (collect) {
            memcpy(raw + used, buffer, (size_t)count);
        }
        sha256_update(&digest, buffer, (size_t)count);
        used += (size_t)count;
    }
    before_identity = identity_from_stat(&before);
    if (result == ANCHOR_OK && (used != (size_t)before.st_size || fstat(descriptor, &after) != 0)) {
        result = ANCHOR_OBJECT_RACED;
    }
    if (result == ANCHOR_OK) {
        after_identity = identity_from_stat(&after);
        if (!identities_match(&before_identity, &after_identity)) {
            result = ANCHOR_OBJECT_RACED;
        }
    }
    if (result == ANCHOR_OK) {
        output->identity = before_identity;
        output->byte_length = used;
        sha256_final(&digest, output->digest);
        if (collect) {
            *collected_raw = raw;
            raw = NULL;
        }
    }
    free(raw);
    return result;
}

static enum anchor_reason open_regular_file(
    const struct held_directory *const parent,
    const char *const name,
    const uid_t expected_uid,
    const gid_t expected_gid,
    const mode_t expected_mode,
    const size_t maximum_bytes,
    const bool executable,
    const bool collect,
    struct held_file *const output,
    unsigned char **const collected_raw
) {
    const struct open_how how = file_open_how();
    struct stat before;
    struct stat after;
    struct stat opened;
    struct object_identity before_identity;
    struct object_identity opened_identity;
    struct object_identity after_identity;
    int descriptor = -1;
    enum anchor_reason reason;

    *collected_raw = NULL;
    if (fstatat(parent->descriptor, name, &before, AT_SYMLINK_NOFOLLOW) != 0) {
        return ANCHOR_ROOT_UNREADABLE;
    }
    reason = check_regular_metadata(&before, expected_uid, expected_gid, expected_mode, maximum_bytes);
    if (reason != ANCHOR_OK) {
        return reason;
    }
    descriptor = invoke_openat2(parent->descriptor, name, &how);
    if (descriptor < 0) {
        return classify_openat2_failure();
    }
    if (fstat(descriptor, &opened) != 0) {
        (void)close(descriptor);
        return ANCHOR_UNSAFE_FILE;
    }
    reason = check_regular_metadata(&opened, expected_uid, expected_gid, expected_mode, maximum_bytes);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        return reason;
    }
    reason = require_approved_filesystem(descriptor);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        return reason;
    }
    reason = require_acl_free(descriptor, false);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        return reason;
    }
    if (executable) {
        reason = require_absent_xattr(descriptor, "security.capability", ANCHOR_CAPABILITY_PRESENT);
        if (reason != ANCHOR_OK) {
            (void)close(descriptor);
            return reason;
        }
    }
    before_identity = identity_from_stat(&before);
    opened_identity = identity_from_stat(&opened);
    if (!identities_match(&before_identity, &opened_identity)) {
        (void)close(descriptor);
        return ANCHOR_OBJECT_RACED;
    }
    output->descriptor = descriptor;
    reason = hash_opened_regular_file(descriptor, maximum_bytes, collect, output, collected_raw);
    if (reason != ANCHOR_OK) {
        (void)close(descriptor);
        output->descriptor = -1;
        free(*collected_raw);
        *collected_raw = NULL;
        return reason;
    }
    if (fstatat(parent->descriptor, name, &after, AT_SYMLINK_NOFOLLOW) != 0) {
        (void)close(descriptor);
        output->descriptor = -1;
        free(*collected_raw);
        *collected_raw = NULL;
        return ANCHOR_OBJECT_RACED;
    }
    after_identity = identity_from_stat(&after);
    if (!identities_match(&before_identity, &after_identity) ||
        !identities_match(&opened_identity, &output->identity)) {
        (void)close(descriptor);
        output->descriptor = -1;
        free(*collected_raw);
        *collected_raw = NULL;
        return ANCHOR_OBJECT_RACED;
    }
    return ANCHOR_OK;
}

static bool consume_literal(
    const unsigned char **const cursor,
    const unsigned char *const end,
    const char *const literal
) {
    const size_t length = strlen(literal);

    if ((size_t)(end - *cursor) < length || memcmp(*cursor, literal, length) != 0) {
        return false;
    }
    *cursor += length;
    return true;
}

static bool consume_positive_decimal(
    const unsigned char **const cursor,
    const unsigned char *const end,
    const size_t maximum,
    size_t *const output
) {
    uintmax_t value = 0;
    size_t digits = 0;

    if (*cursor == end || **cursor < (unsigned char)'1' || **cursor > (unsigned char)'9') {
        return false;
    }
    while (*cursor != end && **cursor >= (unsigned char)'0' && **cursor <= (unsigned char)'9') {
        const unsigned int digit = (unsigned int)(**cursor - (unsigned char)'0');
        if (value > (UINTMAX_MAX - digit) / 10U) {
            return false;
        }
        value = (value * 10U) + digit;
        ++*cursor;
        ++digits;
        if (digits > 20) {
            return false;
        }
    }
    if (value == 0 || value > maximum || value > SIZE_MAX) {
        return false;
    }
    *output = (size_t)value;
    return true;
}

static bool consume_sha256(
    const unsigned char **const cursor,
    const unsigned char *const end,
    char output[SHA256_HEX_BYTES + 1]
) {
    bool nonzero = false;

    if ((size_t)(end - *cursor) < SHA256_HEX_BYTES) {
        return false;
    }
    for (size_t index = 0; index < SHA256_HEX_BYTES; ++index) {
        const unsigned char value = (*cursor)[index];
        if (!((value >= (unsigned char)'0' && value <= (unsigned char)'9') ||
              (value >= (unsigned char)'a' && value <= (unsigned char)'f'))) {
            return false;
        }
        output[index] = (char)value;
        nonzero = nonzero || value != (unsigned char)'0';
    }
    if (!nonzero) {
        return false;
    }
    output[SHA256_HEX_BYTES] = '\0';
    *cursor += SHA256_HEX_BYTES;
    return true;
}

static enum anchor_reason parse_manifest(
    const unsigned char *const raw,
    const size_t length,
    struct manifest_spec *const output
) {
    const unsigned char *cursor = raw;
    const unsigned char *const end = raw + length;

    if (!consume_literal(&cursor, end, "{\"bootstrap\":{\"byte_length\":")) {
        return ANCHOR_MANIFEST_INVALID;
    }
    if (!consume_positive_decimal(&cursor, end, MAX_CODE_BYTES, &output->bootstrap_byte_length) ||
        !consume_literal(&cursor, end, ",\"filename\":\"") ||
        !consume_literal(&cursor, end, BOOTSTRAP_NAME) ||
        !consume_literal(&cursor, end, "\",\"sha256\":\"") ||
        !consume_sha256(&cursor, end, output->bootstrap_sha256) ||
        !consume_literal(&cursor, end, "\"},\"core\":{\"byte_length\":")) {
        return ANCHOR_MANIFEST_INVALID;
    }
    if (!consume_positive_decimal(&cursor, end, MAX_CODE_BYTES, &output->core_byte_length) ||
        !consume_literal(&cursor, end, ",\"filename\":\"") ||
        !consume_literal(&cursor, end, CORE_NAME) ||
        !consume_literal(&cursor, end, "\",\"sha256\":\"") ||
        !consume_sha256(&cursor, end, output->core_sha256) ||
        !consume_literal(&cursor, end, "\"},\"schema_version\":\"") ||
        !consume_literal(&cursor, end, MANIFEST_SCHEMA) ||
        !consume_literal(&cursor, end, "\"}\n") || cursor != end) {
        return ANCHOR_MANIFEST_INVALID;
    }
    return ANCHOR_OK;
}

static bool file_matches_manifest(
    const struct held_file *const file,
    const char expected_sha256[SHA256_HEX_BYTES + 1],
    const size_t expected_byte_length
) {
    char observed[SHA256_HEX_BYTES + 1];

    digest_to_hex(file->digest, observed);
    return file->byte_length == expected_byte_length && strcmp(observed, expected_sha256) == 0;
}

static enum anchor_reason recheck_held_directory(
    const struct held_directory *const directory,
    const struct held_directory *const parent,
    const char *const name
) {
    struct stat opened;
    struct stat named;
    struct object_identity opened_identity;
    struct object_identity named_identity;

    if (fstat(directory->descriptor, &opened) != 0) {
        return ANCHOR_OBJECT_RACED;
    }
    opened_identity = identity_from_stat(&opened);
    if (!identities_match(&opened_identity, &directory->identity)) {
        return ANCHOR_OBJECT_RACED;
    }
    if (parent == NULL) {
        return ANCHOR_OK;
    }
    if (fstatat(parent->descriptor, name, &named, AT_SYMLINK_NOFOLLOW) != 0) {
        return ANCHOR_OBJECT_RACED;
    }
    named_identity = identity_from_stat(&named);
    return identities_match(&named_identity, &directory->identity) ? ANCHOR_OK : ANCHOR_OBJECT_RACED;
}

static enum anchor_reason recheck_held_file(
    const struct held_file *const file,
    const struct held_directory *const parent,
    const char *const name
) {
    struct stat opened;
    struct stat named;
    struct object_identity opened_identity;
    struct object_identity named_identity;

    if (fstat(file->descriptor, &opened) != 0 ||
        fstatat(parent->descriptor, name, &named, AT_SYMLINK_NOFOLLOW) != 0) {
        return ANCHOR_OBJECT_RACED;
    }
    opened_identity = identity_from_stat(&opened);
    named_identity = identity_from_stat(&named);
    if (!identities_match(&opened_identity, &file->identity) ||
        !identities_match(&named_identity, &file->identity)) {
        return ANCHOR_OBJECT_RACED;
    }
    return ANCHOR_OK;
}

static bool held_descriptor_is_readonly_cloexec(const int descriptor) {
    int descriptor_flags;
    int status_flags;

    if (descriptor < 3) {
        return false;
    }
    descriptor_flags = fcntl(descriptor, F_GETFD);
    status_flags = fcntl(descriptor, F_GETFL);

    return descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0 &&
           status_flags >= 0 && (status_flags & O_ACCMODE) == O_RDONLY;
}

static bool bundle_handles_are_clear(const struct bundle_handles *const handles) {
    if (handles == NULL) {
        return false;
    }
    for (size_t index = 0; index < BUNDLE_DIRECTORY_COUNT; ++index) {
        if (handles->directories[index].descriptor != -1) {
            return false;
        }
    }
    return handles->manifest.descriptor == -1 && handles->bootstrap.descriptor == -1 &&
           handles->core.descriptor == -1;
}

static bool bundle_handles_are_complete(const struct bundle_handles *const handles) {
    if (handles == NULL) {
        return false;
    }
    for (size_t index = 0; index < BUNDLE_DIRECTORY_COUNT; ++index) {
        if (!held_descriptor_is_readonly_cloexec(handles->directories[index].descriptor)) {
            return false;
        }
    }
    return held_descriptor_is_readonly_cloexec(handles->manifest.descriptor) &&
           held_descriptor_is_readonly_cloexec(handles->bootstrap.descriptor) &&
           held_descriptor_is_readonly_cloexec(handles->core.descriptor);
}

static enum anchor_reason recheck_bundle_handles(const struct bundle_handles *const handles) {
    enum anchor_reason reason;

    if (!bundle_handles_are_complete(handles)) {
        return ANCHOR_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < BUNDLE_DIRECTORY_COUNT; ++index) {
        reason = recheck_held_directory(
            &handles->directories[index],
            index == 0 ? NULL : &handles->directories[index - 1],
            index == 0 ? NULL : ROOT_BUNDLE_COMPONENTS[index]
        );
        if (reason != ANCHOR_OK) {
            return reason;
        }
    }
    reason = recheck_held_file(
        &handles->manifest, &handles->directories[BUNDLE_DIRECTORY_COUNT - 1], MANIFEST_NAME
    );
    if (reason != ANCHOR_OK) {
        return reason;
    }
    reason = recheck_held_file(
        &handles->bootstrap, &handles->directories[BUNDLE_DIRECTORY_COUNT - 1], BOOTSTRAP_NAME
    );
    if (reason != ANCHOR_OK) {
        return reason;
    }
    return recheck_held_file(
        &handles->core, &handles->directories[BUNDLE_DIRECTORY_COUNT - 1], CORE_NAME
    );
}

enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_held_v1_recheck(
    const struct gate_e_root_bundle_held_v1 *const held
) {
    return recheck_bundle_handles(held);
}

static enum anchor_reason acquire_bundle_from_held_prefix_fd(
    const int prefix_fd,
    const uid_t expected_uid,
    const gid_t expected_gid,
    struct bundle_handles *const output
) {
    struct bundle_handles candidate;
    struct manifest_spec manifest_specification;
    unsigned char *manifest_raw = NULL;
    unsigned char *unused_raw = NULL;
    enum anchor_reason reason;

    if (!bundle_handles_are_clear(output)) {
        return ANCHOR_INVALID_ARGUMENT;
    }
    initialize_bundle_handles(output);
    initialize_bundle_handles(&candidate);
    candidate.directories[0].descriptor = fcntl(prefix_fd, F_DUPFD_CLOEXEC, 3);
    if (candidate.directories[0].descriptor < 0) {
        return ANCHOR_ROOT_UNREADABLE;
    }
    reason = inspect_opened_directory(
        candidate.directories[0].descriptor, expected_uid, expected_gid, 0755,
        &candidate.directories[0].identity
    );
    if (reason != ANCHOR_OK) {
        goto finish;
    }
    for (size_t index = 1; index < BUNDLE_DIRECTORY_COUNT; ++index) {
        reason = open_child_directory(
            &candidate.directories[index - 1], ROOT_BUNDLE_COMPONENTS[index], expected_uid, expected_gid,
            &candidate.directories[index]
        );
        if (reason != ANCHOR_OK) {
            goto finish;
        }
    }
    reason = open_regular_file(
        &candidate.directories[BUNDLE_DIRECTORY_COUNT - 1], MANIFEST_NAME, expected_uid, expected_gid,
        0644, MAX_MANIFEST_BYTES, false, true, &candidate.manifest, &manifest_raw
    );
    if (reason != ANCHOR_OK) {
        goto finish;
    }
    reason = parse_manifest(manifest_raw, candidate.manifest.byte_length, &manifest_specification);
    if (reason != ANCHOR_OK) {
        goto finish;
    }
    free(manifest_raw);
    manifest_raw = NULL;
    reason = open_regular_file(
        &candidate.directories[BUNDLE_DIRECTORY_COUNT - 1], BOOTSTRAP_NAME, expected_uid, expected_gid,
        0755, MAX_CODE_BYTES, true, false, &candidate.bootstrap, &unused_raw
    );
    if (reason != ANCHOR_OK) {
        goto finish;
    }
    reason = open_regular_file(
        &candidate.directories[BUNDLE_DIRECTORY_COUNT - 1], CORE_NAME, expected_uid, expected_gid,
        0644, MAX_CODE_BYTES, false, false, &candidate.core, &unused_raw
    );
    if (reason != ANCHOR_OK) {
        goto finish;
    }
    if (!file_matches_manifest(
            &candidate.bootstrap, manifest_specification.bootstrap_sha256,
            manifest_specification.bootstrap_byte_length
        ) ||
        !file_matches_manifest(
            &candidate.core, manifest_specification.core_sha256,
            manifest_specification.core_byte_length
        )) {
        reason = ANCHOR_DIGEST_MISMATCH;
        goto finish;
    }
    reason = recheck_bundle_handles(&candidate);

finish:
    free(manifest_raw);
    free(unused_raw);
    if (reason != ANCHOR_OK) {
        (void)close_bundle_handles(&candidate);
        initialize_bundle_handles(output);
        return reason;
    }
    *output = candidate;
    initialize_bundle_handles(&candidate);
    return ANCHOR_OK;
}

enum gate_e_root_bundle_reason_v1 gate_e_root_bundle_acquire_fixed_v1(
    struct gate_e_root_bundle_held_v1 *const held
) {
    struct held_directory root = {.descriptor = -1};
    enum anchor_reason reason;

    if (!bundle_handles_are_clear(held)) {
        return ANCHOR_INVALID_ARGUMENT;
    }
    initialize_bundle_handles(held);
    if (getuid() != 0 || geteuid() != 0 || getgid() != 0 || getegid() != 0) {
        return ANCHOR_NOT_ROOT;
    }
    reason = open_root_directory(0, 0, &root);
    if (reason == ANCHOR_OK) {
        reason = acquire_bundle_from_held_prefix_fd(root.descriptor, 0, 0, held);
    }
    if (root.descriptor >= 0 && close(root.descriptor) != 0 && reason == ANCHOR_OK) {
        reason = ANCHOR_CLOSE_FAILED;
    }
    if (reason != ANCHOR_OK) {
        (void)close_bundle_handles(held);
    }
    return reason;
}

#if !defined(GATE_E_ROOT_BUNDLE_AUTHENTICATOR_LIBRARY) || \
    defined(GATE_E_ROOT_BUNDLE_AUTHENTICATOR_TESTING)
static const char *reason_code(const enum anchor_reason reason) {
    switch (reason) {
    case ANCHOR_OK:
        return "checked";
    case ANCHOR_NOT_ROOT:
        return "effective-uid-gid-not-root";
    case ANCHOR_OPENAT2_UNAVAILABLE:
        return "openat2-abi-unavailable";
    case ANCHOR_ROOT_UNREADABLE:
        return "fixed-root-bundle-unreadable";
    case ANCHOR_UNSAFE_DIRECTORY:
        return "root-bundle-directory-policy-failed";
    case ANCHOR_UNSAFE_FILESYSTEM:
        return "root-bundle-filesystem-not-approved";
    case ANCHOR_ACL_PRESENT:
        return "root-bundle-posix-acl-present";
    case ANCHOR_ACL_UNVERIFIABLE:
        return "root-bundle-xattr-policy-unverifiable";
    case ANCHOR_CAPABILITY_PRESENT:
        return "root-bundle-bootstrap-capability-present";
    case ANCHOR_UNSAFE_FILE:
        return "root-bundle-file-policy-failed";
    case ANCHOR_FILE_UNREADABLE:
        return "root-bundle-file-unreadable";
    case ANCHOR_OBJECT_RACED:
        return "root-bundle-object-raced";
    case ANCHOR_MANIFEST_INVALID:
        return "root-bundle-manifest-invalid";
    case ANCHOR_DIGEST_MISMATCH:
        return "root-bundle-manifest-digest-mismatch";
    case ANCHOR_CLOSE_FAILED:
        return "root-bundle-held-fd-close-failed";
    case ANCHOR_INVALID_ARGUMENT:
        return "root-bundle-held-handle-invalid";
    }
    return "unknown-root-bundle-preflight-reason";
}

static bool print_report(FILE *const stream, const enum anchor_reason reason) {
    const char *const observation_status = reason == ANCHOR_OK ? "checked" : "not-established";
    const char *const detail = reason == ANCHOR_OK ? "null" : reason_code(reason);
    int written = 0;

    if (reason == ANCHOR_OK) {
        written = fprintf(
            stream,
            "{\"schema_version\":\"riley.rc3-gate-e-native-root-bundle-preflight.v1\","
            "\"status\":\"not-established\",\"object_observation_status\":\"%s\","
            "\"scope\":\"root-bundle-object-observation-only\","
            "\"authority\":\"not-authoritative\",\"installation\":\"not-installed\","
            "\"host_initial_namespace\":\"not-established\","
            "\"pre_python_loader_boundary\":\"not-established\","
            "\"same_object_exec\":\"not-established\","
            "\"interpreter_runtime_closure\":\"not-established\","
            "\"guardian_lease\":\"not-established\","
            "\"execution_authority\":\"not-established\","
            "\"actual_gate_e_producer\":\"not-established\","
            "\"gpu_execution\":\"not-run\",\"docker_execution\":\"not-established\","
            "\"evidence\":\"not-established\",\"qualification_status\":\"not-run\","
            "\"reason_code\":%s}\n",
            observation_status,
            detail
        );
    } else {
        written = fprintf(
            stream,
            "{\"schema_version\":\"riley.rc3-gate-e-native-root-bundle-preflight.v1\","
            "\"status\":\"not-established\",\"object_observation_status\":\"%s\","
            "\"scope\":\"root-bundle-object-observation-only\","
            "\"authority\":\"not-authoritative\",\"installation\":\"not-installed\","
            "\"host_initial_namespace\":\"not-established\","
            "\"pre_python_loader_boundary\":\"not-established\","
            "\"same_object_exec\":\"not-established\","
            "\"interpreter_runtime_closure\":\"not-established\","
            "\"guardian_lease\":\"not-established\","
            "\"execution_authority\":\"not-established\","
            "\"actual_gate_e_producer\":\"not-established\","
            "\"gpu_execution\":\"not-run\",\"docker_execution\":\"not-established\","
            "\"evidence\":\"not-established\",\"qualification_status\":\"not-run\","
            "\"reason_code\":\"%s\"}\n",
            observation_status,
            detail
        );
    }
    return written >= 0 && fflush(stream) == 0;
}

static bool accepted_cli(const int argc, const char *const argv[]) {
    return argc == 2 && argv != NULL && argv[0] != NULL && argv[1] != NULL &&
           strcmp(argv[1], "--authenticate-root-bundle-v1") == 0;
}

static const char *program_name_for_usage(const int argc, const char *const argv[]) {
    if (argc > 0 && argv != NULL && argv[0] != NULL && argv[0][0] != '\0') {
        return argv[0];
    }
    return "gate_e_root_bundle_authenticator";
}
#endif

#ifndef GATE_E_ROOT_BUNDLE_AUTHENTICATOR_LIBRARY
static enum anchor_reason authenticate_fixed_root_bundle_v1(void) {
    struct bundle_handles held;
    enum anchor_reason reason;

    gate_e_root_bundle_held_v1_init(&held);
    reason = gate_e_root_bundle_acquire_fixed_v1(&held);
    if (reason == ANCHOR_OK) {
        reason = gate_e_root_bundle_held_v1_close(&held);
    }
    return reason;
}

int main(const int argc, char *const argv[]) {
    enum anchor_reason reason;

    if (!accepted_cli(argc, (const char *const *)argv)) {
        (void)fprintf(
            stderr, "usage: %s --authenticate-root-bundle-v1\n",
            program_name_for_usage(argc, (const char *const *)argv)
        );
        return 64;
    }
    reason = authenticate_fixed_root_bundle_v1();
    if (!print_report(stdout, reason)) {
        (void)fprintf(stderr, "unable to emit root bundle preflight report\n");
        return 2;
    }
    return reason == ANCHOR_OK ? 0 : 2;
}
#endif
