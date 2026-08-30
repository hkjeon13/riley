/*
 * Linux-only source/audit primitive for a future RC3 Gate E native guardian.
 * It copies bytes only from an already-authenticated held regular FD to a new
 * anonymous sealed memfd. It neither opens a path nor executes, installs,
 * signals, locks, creates a cgroup, contacts a GPU/Docker service, or writes
 * a filesystem/evidence/receipt artifact. A separately reviewed static root
 * guardian must authenticate the source object and decide any FD handoff.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gate_e_sealed_leaf_snapshot.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* Linux UAPI additions from the no-exec memfd ABI; old libc headers lack them. */
#ifndef MFD_NOEXEC_SEAL
#define MFD_NOEXEC_SEAL 0x0008U
#endif

#ifndef F_SEAL_EXEC
#define F_SEAL_EXEC 0x0020
#endif

enum {
    SHA256_BLOCK_BYTES = 64,
    SNAPSHOT_BUFFER_BYTES = 8192,
    INITIAL_MEMFD_SEALS = F_SEAL_EXEC,
    ADDED_MEMFD_SEALS = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE,
    REQUIRED_MEMFD_SEALS = F_SEAL_EXEC | F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE,
};

static const uint64_t SNAPSHOT_INITIALIZED_STATE = UINT64_C(0x52494c45534e4150);

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[SHA256_BLOCK_BYTES];
    size_t block_length;
};

struct object_identity {
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

typedef int (*memfd_create_invoker)(const char *, unsigned int);
typedef ssize_t (*pread_invoker)(int, void *, size_t, off_t);
typedef ssize_t (*pwrite_invoker)(int, const void *, size_t, off_t);
typedef int (*add_seals_invoker)(int, int);

static int linux_memfd_create(const char *const name, const unsigned int flags) {
#ifdef SYS_memfd_create
    return (int)syscall(SYS_memfd_create, name, flags);
#else
    (void)name;
    (void)flags;
    errno = ENOSYS;
    return -1;
#endif
}

static memfd_create_invoker invoke_memfd_create = linux_memfd_create;
static pread_invoker invoke_pread = pread;
static pwrite_invoker invoke_pwrite = pwrite;

static int linux_add_seals(const int descriptor, const int seals) {
    return fcntl(descriptor, F_ADD_SEALS, seals);
}

static add_seals_invoker invoke_add_seals = linux_add_seals;

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
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU,
        0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU,
        0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
        0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU,
        0xbef9a3f7U, 0xc67178f2U,
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
        words[index] = load_be32(block + (index * 4U));
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
    while (length != 0U) {
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

static void sha256_final(struct sha256_context *const context, unsigned char digest[GATE_E_SEALED_LEAF_SHA256_BYTES]) {
    const uint64_t final_bits = context->bit_length + ((uint64_t)context->block_length * UINT64_C(8));
    size_t index = context->block_length;

    context->block[index++] = 0x80U;
    if (index > 56U) {
        while (index < SHA256_BLOCK_BYTES) {
            context->block[index++] = 0U;
        }
        sha256_transform(context, context->block);
        index = 0;
    }
    while (index < 56U) {
        context->block[index++] = 0U;
    }
    for (size_t byte = 0; byte < 8U; ++byte) {
        context->block[63U - byte] = (unsigned char)(final_bits >> (byte * 8U));
    }
    sha256_transform(context, context->block);
    for (size_t word = 0; word < 8U; ++word) {
        store_be32(digest + (word * 4U), context->state[word]);
    }
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

static void clear_snapshot(struct gate_e_sealed_leaf_snapshot *const snapshot) {
    snapshot->descriptor = -1;
    snapshot->byte_length = 0;
    memset(snapshot->sha256, 0, sizeof(snapshot->sha256));
}

void gate_e_sealed_leaf_snapshot_init(struct gate_e_sealed_leaf_snapshot *const snapshot) {
    if (snapshot == NULL) {
        return;
    }
    snapshot->initialized_state = SNAPSHOT_INITIALIZED_STATE;
    clear_snapshot(snapshot);
}

static bool digest_is_nonzero(const unsigned char digest[GATE_E_SEALED_LEAF_SHA256_BYTES]) {
    unsigned char combined = 0U;

    for (size_t index = 0; index < GATE_E_SEALED_LEAF_SHA256_BYTES; ++index) {
        combined |= digest[index];
    }
    return combined != 0U;
}

static bool checked_regular_source(const struct stat *const metadata, const size_t expected_byte_length) {
    return S_ISREG(metadata->st_mode) && metadata->st_nlink == 1 && metadata->st_size > 0 &&
           (uintmax_t)metadata->st_size == (uintmax_t)expected_byte_length;
}

static bool source_status_flags_are_safe(const int status_flags) {
    if ((status_flags & O_ACCMODE) != O_RDONLY || (status_flags & O_NOATIME) == 0 ||
        (status_flags & O_APPEND) != 0) {
        return false;
    }
#ifdef O_DIRECT
    if ((status_flags & O_DIRECT) != 0) {
        return false;
    }
#endif
#ifdef O_PATH
    if ((status_flags & O_PATH) != 0) {
        return false;
    }
#endif
    return true;
}

static enum gate_e_sealed_leaf_reason inspect_source(
    const int descriptor,
    const size_t expected_byte_length,
    struct object_identity *const identity
) {
    struct stat metadata;
    const int descriptor_flags = fcntl(descriptor, F_GETFD);
    const int status_flags = fcntl(descriptor, F_GETFL);

    if (descriptor_flags < 0 || status_flags < 0) {
        return GATE_E_SEALED_LEAF_SOURCE_UNREADABLE;
    }
    if ((descriptor_flags & FD_CLOEXEC) == 0 || !source_status_flags_are_safe(status_flags)) {
        return GATE_E_SEALED_LEAF_SOURCE_UNSAFE;
    }
    if (fstat(descriptor, &metadata) != 0) {
        return GATE_E_SEALED_LEAF_SOURCE_UNREADABLE;
    }
    if (!checked_regular_source(&metadata, expected_byte_length)) {
        return GATE_E_SEALED_LEAF_SOURCE_UNSAFE;
    }
    *identity = identity_from_stat(&metadata);
    return GATE_E_SEALED_LEAF_OK;
}

static bool close_descriptor_once(int *const descriptor) {
    const int value = *descriptor;

    *descriptor = -1;
    return value < 0 || close(value) == 0;
}

static enum gate_e_sealed_leaf_reason pin_source_descriptor(
    const int descriptor,
    int *const pinned_descriptor
) {
    int duplicate_descriptor = fcntl(descriptor, F_DUPFD_CLOEXEC, 3);

    if (duplicate_descriptor < 0) {
        return errno == EBADF ? GATE_E_SEALED_LEAF_SOURCE_UNREADABLE
                              : GATE_E_SEALED_LEAF_SOURCE_PIN_FAILED;
    }
    *pinned_descriptor = duplicate_descriptor;
    return GATE_E_SEALED_LEAF_OK;
}

static bool current_file_size_limit_allows(const size_t expected_byte_length) {
    struct rlimit limit;

    return getrlimit(RLIMIT_FSIZE, &limit) == 0 &&
           (limit.rlim_cur == RLIM_INFINITY ||
            (uintmax_t)limit.rlim_cur >= (uintmax_t)expected_byte_length);
}

static bool checked_private_memfd(
    const int descriptor,
    const size_t expected_byte_length,
    const int expected_seals
) {
    struct stat metadata;
    const int descriptor_flags = fcntl(descriptor, F_GETFD);
    const int status_flags = fcntl(descriptor, F_GETFL);
    const int seals = fcntl(descriptor, F_GET_SEALS);

    return descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0 && status_flags >= 0 &&
           (status_flags & O_ACCMODE) == O_RDWR && (status_flags & O_APPEND) == 0 &&
           seals == expected_seals && fstat(descriptor, &metadata) == 0 && S_ISREG(metadata.st_mode) &&
           (metadata.st_mode & 0111) == 0 && metadata.st_nlink == 0 && metadata.st_size >= 0 &&
           (uintmax_t)metadata.st_size == (uintmax_t)expected_byte_length;
}

static enum gate_e_sealed_leaf_reason create_private_noexec_memfd(int *const descriptor) {
    int raw_descriptor = invoke_memfd_create(
        "riley-gate-e-sealed-leaf", MFD_ALLOW_SEALING | MFD_CLOEXEC | MFD_NOEXEC_SEAL
    );
    int normalized_descriptor;

    if (raw_descriptor < 0) {
        return (errno == ENOSYS || errno == EINVAL || errno == EPERM)
                   ? GATE_E_SEALED_LEAF_MEMFD_UNAVAILABLE
                   : GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
    }
    normalized_descriptor = fcntl(raw_descriptor, F_DUPFD_CLOEXEC, 3);
    if (normalized_descriptor < 0) {
        if (!close_descriptor_once(&raw_descriptor)) {
            return GATE_E_SEALED_LEAF_CLOSE_FAILED;
        }
        return GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
    }
    if (!close_descriptor_once(&raw_descriptor)) {
        (void)close_descriptor_once(&normalized_descriptor);
        return GATE_E_SEALED_LEAF_CLOSE_FAILED;
    }
    *descriptor = normalized_descriptor;
    return GATE_E_SEALED_LEAF_OK;
}

static enum gate_e_sealed_leaf_reason pwrite_all_to_new_memfd(
    const int descriptor,
    const unsigned char *buffer,
    size_t length,
    size_t offset
) {
    while (length != 0U) {
        const ssize_t count = invoke_pwrite(descriptor, buffer, length, (off_t)offset);

        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0 || (size_t)count > length) {
            return GATE_E_SEALED_LEAF_MEMFD_WRITE_FAILED;
        }
        buffer += (size_t)count;
        length -= (size_t)count;
        offset += (size_t)count;
    }
    return GATE_E_SEALED_LEAF_OK;
}

static enum gate_e_sealed_leaf_reason copy_and_hash_source(
    const int source_descriptor,
    const int destination_descriptor,
    const size_t expected_byte_length,
    unsigned char observed_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES]
) {
    struct sha256_context context;
    unsigned char buffer[SNAPSHOT_BUFFER_BYTES];
    size_t offset = 0;

    sha256_init(&context);
    while (offset < expected_byte_length) {
        const size_t remaining = expected_byte_length - offset;
        const size_t request = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        ssize_t count;
        enum gate_e_sealed_leaf_reason reason;

        do {
            count = invoke_pread(source_descriptor, buffer, request, (off_t)offset);
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            return GATE_E_SEALED_LEAF_SOURCE_UNREADABLE;
        }
        if (count == 0 || (size_t)count > remaining) {
            return GATE_E_SEALED_LEAF_SOURCE_RACED;
        }
        reason = pwrite_all_to_new_memfd(
            destination_descriptor, buffer, (size_t)count, offset
        );
        if (reason != GATE_E_SEALED_LEAF_OK) {
            return reason;
        }
        sha256_update(&context, buffer, (size_t)count);
        offset += (size_t)count;
    }
    sha256_final(&context, observed_sha256);
    return GATE_E_SEALED_LEAF_OK;
}

static enum gate_e_sealed_leaf_reason verify_sealed_snapshot(
    const int descriptor,
    const size_t expected_byte_length,
    const unsigned char expected_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES]
) {
    struct stat before;
    struct stat after;
    struct object_identity before_identity;
    struct object_identity after_identity;
    struct sha256_context context;
    unsigned char observed_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES];
    unsigned char buffer[SNAPSHOT_BUFFER_BYTES];
    size_t offset = 0;

    if (!checked_private_memfd(descriptor, expected_byte_length, REQUIRED_MEMFD_SEALS) ||
        fstat(descriptor, &before) != 0) {
        return GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
    }
    before_identity = identity_from_stat(&before);
    sha256_init(&context);
    while (offset < expected_byte_length) {
        const size_t remaining = expected_byte_length - offset;
        const size_t request = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        ssize_t count;

        do {
            count = invoke_pread(descriptor, buffer, request, (off_t)offset);
        } while (count < 0 && errno == EINTR);
        if (count <= 0 || (size_t)count > remaining) {
            return GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
        }
        sha256_update(&context, buffer, (size_t)count);
        offset += (size_t)count;
    }
    sha256_final(&context, observed_sha256);
    if (memcmp(observed_sha256, expected_sha256, GATE_E_SEALED_LEAF_SHA256_BYTES) != 0 ||
        fstat(descriptor, &after) != 0) {
        return GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
    }
    after_identity = identity_from_stat(&after);
    if (!identities_match(&before_identity, &after_identity)) {
        return GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
    }
    return GATE_E_SEALED_LEAF_OK;
}

enum gate_e_sealed_leaf_reason gate_e_snapshot_held_leaf_v1(
    const int source_descriptor,
    const unsigned char expected_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES],
    const size_t expected_byte_length,
    struct gate_e_sealed_leaf_snapshot *const output
) {
    struct object_identity source_before;
    struct object_identity source_after;
    unsigned char expected_digest[GATE_E_SEALED_LEAF_SHA256_BYTES];
    unsigned char observed_sha256[GATE_E_SEALED_LEAF_SHA256_BYTES];
    int pinned_source_descriptor = -1;
    int snapshot_descriptor = -1;
    enum gate_e_sealed_leaf_reason reason;

    if (output == NULL || output->initialized_state != SNAPSHOT_INITIALIZED_STATE ||
        output->descriptor != -1) {
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    if (expected_sha256 == NULL) {
        clear_snapshot(output);
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    memcpy(expected_digest, expected_sha256, sizeof(expected_digest));
    clear_snapshot(output);
    if (source_descriptor < 0 || expected_byte_length == 0U ||
        expected_byte_length > GATE_E_SEALED_LEAF_MAX_BYTES) {
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    if (!digest_is_nonzero(expected_digest)) {
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    reason = pin_source_descriptor(source_descriptor, &pinned_source_descriptor);
    if (reason != GATE_E_SEALED_LEAF_OK) {
        return reason;
    }
    reason = inspect_source(pinned_source_descriptor, expected_byte_length, &source_before);
    if (reason != GATE_E_SEALED_LEAF_OK) {
        goto fail;
    }
    if (!current_file_size_limit_allows(expected_byte_length)) {
        reason = GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
        goto fail;
    }
    reason = create_private_noexec_memfd(&snapshot_descriptor);
    if (reason != GATE_E_SEALED_LEAF_OK) {
        goto fail;
    }
    if (!checked_private_memfd(snapshot_descriptor, 0U, INITIAL_MEMFD_SEALS)) {
        reason = GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
        goto fail;
    }
    reason = copy_and_hash_source(
        pinned_source_descriptor, snapshot_descriptor, expected_byte_length, observed_sha256
    );
    if (reason != GATE_E_SEALED_LEAF_OK) {
        goto fail;
    }
    if (memcmp(observed_sha256, expected_digest, GATE_E_SEALED_LEAF_SHA256_BYTES) != 0) {
        reason = GATE_E_SEALED_LEAF_DIGEST_MISMATCH;
        goto fail;
    }
    reason = inspect_source(pinned_source_descriptor, expected_byte_length, &source_after);
    if (reason != GATE_E_SEALED_LEAF_OK) {
        goto fail;
    }
    if (!identities_match(&source_before, &source_after)) {
        reason = GATE_E_SEALED_LEAF_SOURCE_RACED;
        goto fail;
    }
    if (!close_descriptor_once(&pinned_source_descriptor)) {
        reason = GATE_E_SEALED_LEAF_CLOSE_FAILED;
        goto fail;
    }
    if (!checked_private_memfd(snapshot_descriptor, expected_byte_length, INITIAL_MEMFD_SEALS)) {
        reason = GATE_E_SEALED_LEAF_MEMFD_UNSAFE;
        goto fail;
    }
    if (invoke_add_seals(snapshot_descriptor, ADDED_MEMFD_SEALS) != 0) {
        reason = GATE_E_SEALED_LEAF_MEMFD_SEAL_FAILED;
        goto fail;
    }
    reason = verify_sealed_snapshot(snapshot_descriptor, expected_byte_length, expected_digest);
    if (reason != GATE_E_SEALED_LEAF_OK) {
        goto fail;
    }
    output->descriptor = snapshot_descriptor;
    output->byte_length = expected_byte_length;
    memcpy(output->sha256, expected_digest, sizeof(output->sha256));
    snapshot_descriptor = -1;
    return GATE_E_SEALED_LEAF_OK;

fail: {
    bool cleanup_failed = false;

    if (!close_descriptor_once(&snapshot_descriptor)) {
        cleanup_failed = true;
    }
    if (!close_descriptor_once(&pinned_source_descriptor)) {
        cleanup_failed = true;
    }
    if (cleanup_failed) {
        reason = GATE_E_SEALED_LEAF_CLOSE_FAILED;
    }
    clear_snapshot(output);
    return reason;
}
}

enum gate_e_sealed_leaf_reason gate_e_sealed_leaf_snapshot_close(
    struct gate_e_sealed_leaf_snapshot *const snapshot
) {
    enum gate_e_sealed_leaf_reason reason = GATE_E_SEALED_LEAF_OK;

    if (snapshot == NULL || snapshot->initialized_state != SNAPSHOT_INITIALIZED_STATE) {
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    if (snapshot->descriptor < -1 || (snapshot->descriptor >= 0 && snapshot->descriptor < 3)) {
        clear_snapshot(snapshot);
        return GATE_E_SEALED_LEAF_INVALID_ARGUMENT;
    }
    if (!close_descriptor_once(&snapshot->descriptor)) {
        reason = GATE_E_SEALED_LEAF_CLOSE_FAILED;
    }
    clear_snapshot(snapshot);
    return reason;
}
