/*
 * Linux-only source precursor for a future static guardian. This library
 * joins canonical execution-closure declaration bytes to already-borrowed
 * regular descriptors. It deliberately does not open a path, authenticate
 * provenance, inspect ELF, resolve a loader, create a child, arrange FD 31/32,
 * execute, or perform any PID1/GPU/Docker/evidence/qualification operation.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "gate_e_execution_closure_held_fds_v1.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    SHA256_BLOCK_BYTES = 64,
    HASH_BUFFER_BYTES = 8192,
    FIXED_ROLE_COUNT = 2,
};

static const uint64_t HELD_FDS_INITIALIZED_STATE = UINT64_C(0x52494c4543484644);

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[SHA256_BLOCK_BYTES];
    size_t block_length;
};

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

static void sha256_transform(
    struct sha256_context *const context,
    const unsigned char block[SHA256_BLOCK_BYTES]
) {
    static const uint32_t round_constants[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
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

    for (size_t index = 0U; index < 16U; ++index) {
        words[index] = load_be32(block + (index * 4U));
    }
    for (size_t index = 16U; index < 64U; ++index) {
        const uint32_t s0 = rotate_right(words[index - 15U], 7U) ^
                            rotate_right(words[index - 15U], 18U) ^
                            (words[index - 15U] >> 3U);
        const uint32_t s1 = rotate_right(words[index - 2U], 17U) ^
                            rotate_right(words[index - 2U], 19U) ^
                            (words[index - 2U] >> 10U);

        words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }
    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (size_t index = 0U; index < 64U; ++index) {
        const uint32_t sigma1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                rotate_right(e, 25U);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temporary1 = h + sigma1 + choose + round_constants[index] + words[index];
        const uint32_t sigma0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                rotate_right(a, 22U);
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
    context->bit_length = 0U;
    context->block_length = 0U;
}

static void sha256_update(
    struct sha256_context *const context,
    const unsigned char *data,
    size_t length
) {
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
            context->block_length = 0U;
        }
    }
}

static void sha256_final(
    struct sha256_context *const context,
    unsigned char digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    const uint64_t final_bits = context->bit_length + ((uint64_t)context->block_length * UINT64_C(8));
    size_t index = context->block_length;

    context->block[index++] = 0x80U;
    if (index > 56U) {
        while (index < SHA256_BLOCK_BYTES) {
            context->block[index++] = 0U;
        }
        sha256_transform(context, context->block);
        index = 0U;
    }
    while (index < 56U) {
        context->block[index++] = 0U;
    }
    for (size_t byte = 0U; byte < 8U; ++byte) {
        context->block[63U - byte] = (unsigned char)(final_bits >> (byte * 8U));
    }
    sha256_transform(context, context->block);
    for (size_t word = 0U; word < 8U; ++word) {
        store_be32(digest + (word * 4U), context->state[word]);
    }
}

static bool bytes_are_zero(const unsigned char *const bytes, const size_t length) {
    unsigned char combined = 0U;

    for (size_t index = 0U; index < length; ++index) {
        combined |= bytes[index];
    }
    return combined == 0U;
}

static bool digest_is_nonzero(
    const unsigned char digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    return !bytes_are_zero(digest, GATE_E_EXECUTION_CLOSURE_SHA256_BYTES);
}

static struct gate_e_execution_closure_object_identity_v1 identity_from_stat(
    const struct stat *const metadata
) {
    return (struct gate_e_execution_closure_object_identity_v1){
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

static bool identities_equal(
    const struct gate_e_execution_closure_object_identity_v1 *const left,
    const struct gate_e_execution_closure_object_identity_v1 *const right
) {
    return left->device == right->device && left->inode == right->inode &&
           left->mode == right->mode && left->links == right->links &&
           left->uid == right->uid && left->gid == right->gid &&
           left->size == right->size && left->mtime.tv_sec == right->mtime.tv_sec &&
           left->mtime.tv_nsec == right->mtime.tv_nsec &&
           left->ctime.tv_sec == right->ctime.tv_sec &&
           left->ctime.tv_nsec == right->ctime.tv_nsec;
}

static bool identities_alias(
    const struct gate_e_execution_closure_object_identity_v1 *const left,
    const struct gate_e_execution_closure_object_identity_v1 *const right
) {
    return left->device == right->device && left->inode == right->inode;
}

static bool identity_is_zero(
    const struct gate_e_execution_closure_object_identity_v1 *const identity
) {
    return identity->device == 0 && identity->inode == 0 && identity->mode == 0 &&
           identity->links == 0 && identity->uid == 0 && identity->gid == 0 &&
           identity->size == 0 && identity->mtime.tv_sec == 0 &&
           identity->mtime.tv_nsec == 0L && identity->ctime.tv_sec == 0 &&
           identity->ctime.tv_nsec == 0L;
}

static bool leaf_is_empty(const struct gate_e_execution_closure_leaf *const leaf) {
    return leaf->audit_path_length == 0U && leaf->byte_length == 0U &&
           bytes_are_zero(leaf->audit_path, sizeof(leaf->audit_path)) &&
           bytes_are_zero(leaf->sha256, sizeof(leaf->sha256));
}

static bool leaf_is_well_formed(const struct gate_e_execution_closure_leaf *const leaf) {
    return leaf->audit_path_length != 0U &&
           leaf->audit_path_length <= GATE_E_EXECUTION_CLOSURE_MAX_AUDIT_PATH_BYTES &&
           leaf->audit_path[leaf->audit_path_length] == 0U &&
           leaf->byte_length != 0U &&
           leaf->byte_length <= GATE_E_EXECUTION_CLOSURE_MAX_LEAF_BYTES &&
           digest_is_nonzero(leaf->sha256);
}

static void clear_held_file(struct gate_e_execution_closure_held_file_v1 *const file) {
    file->descriptor = -1;
    memset(&file->declaration, 0, sizeof(file->declaration));
    memset(&file->identity, 0, sizeof(file->identity));
}

static bool held_file_is_empty(const struct gate_e_execution_closure_held_file_v1 *const file) {
    return file->descriptor == -1 && leaf_is_empty(&file->declaration) &&
           identity_is_zero(&file->identity);
}

static bool held_file_is_well_formed(
    const struct gate_e_execution_closure_held_file_v1 *const file
) {
    return file->descriptor >= 3 && leaf_is_well_formed(&file->declaration);
}

void gate_e_execution_closure_held_fds_v1_init(
    struct gate_e_execution_closure_held_fds_v1 *const output
) {
    if (output == NULL) {
        return;
    }
    memset(output, 0, sizeof(*output));
    output->initialized_state = HELD_FDS_INITIALIZED_STATE;
    clear_held_file(&output->dynamic_loader);
    clear_held_file(&output->interpreter);
    for (size_t index = 0U; index < GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES; ++index) {
        clear_held_file(&output->runtime_leaves[index]);
    }
}

static bool output_is_initialized_empty(
    const struct gate_e_execution_closure_held_fds_v1 *const output
) {
    if (output == NULL || output->initialized_state != HELD_FDS_INITIALIZED_STATE ||
        output->runtime_leaf_count != 0U ||
        !bytes_are_zero(output->runtime_closure_sha256, sizeof(output->runtime_closure_sha256)) ||
        !held_file_is_empty(&output->dynamic_loader) ||
        !held_file_is_empty(&output->interpreter)) {
        return false;
    }
    for (size_t index = 0U; index < GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES; ++index) {
        if (!held_file_is_empty(&output->runtime_leaves[index])) {
            return false;
        }
    }
    return true;
}

static bool close_descriptor_once(int *const descriptor) {
    const int value = *descriptor;

    *descriptor = -1;
    return value == -1 || (value >= 3 && close(value) == 0);
}

static enum gate_e_execution_closure_held_fds_reason_v1 close_after_failure(
    int *const descriptor,
    const enum gate_e_execution_closure_held_fds_reason_v1 failure_reason
) {
    return close_descriptor_once(descriptor)
               ? failure_reason
               : GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1;
}

enum gate_e_execution_closure_held_fds_reason_v1
gate_e_execution_closure_held_fds_v1_close(
    struct gate_e_execution_closure_held_fds_v1 *const held
) {
    bool close_failed = false;

    if (held == NULL || held->initialized_state != HELD_FDS_INITIALIZED_STATE) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1;
    }
    if (!close_descriptor_once(&held->dynamic_loader.descriptor)) {
        close_failed = true;
    }
    if (!close_descriptor_once(&held->interpreter.descriptor)) {
        close_failed = true;
    }
    for (size_t index = 0U; index < GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES; ++index) {
        if (!close_descriptor_once(&held->runtime_leaves[index].descriptor)) {
            close_failed = true;
        }
    }
    gate_e_execution_closure_held_fds_v1_init(held);
    return close_failed ? GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1
                        : GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

static const int *borrowed_descriptor_at(
    const struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed,
    const size_t index
) {
    if (index == 0U) {
        return &borrowed->dynamic_loader_descriptor;
    }
    if (index == 1U) {
        return &borrowed->interpreter_descriptor;
    }
    return &borrowed->runtime_leaf_descriptors[index - FIXED_ROLE_COUNT];
}

static const struct gate_e_execution_closure_held_file_v1 *held_file_at(
    const struct gate_e_execution_closure_held_fds_v1 *const held,
    const size_t index
) {
    if (index == 0U) {
        return &held->dynamic_loader;
    }
    if (index == 1U) {
        return &held->interpreter;
    }
    return &held->runtime_leaves[index - FIXED_ROLE_COUNT];
}

static struct gate_e_execution_closure_held_file_v1 *mutable_held_file_at(
    struct gate_e_execution_closure_held_fds_v1 *const held,
    const size_t index
) {
    if (index == 0U) {
        return &held->dynamic_loader;
    }
    if (index == 1U) {
        return &held->interpreter;
    }
    return &held->runtime_leaves[index - FIXED_ROLE_COUNT];
}

static const struct gate_e_execution_closure_leaf *parsed_leaf_at(
    const struct gate_e_execution_closure_manifest *const parsed,
    const size_t index
) {
    if (index == 0U) {
        return &parsed->dynamic_loader;
    }
    if (index == 1U) {
        return &parsed->interpreter;
    }
    return &parsed->runtime_leaves[index - FIXED_ROLE_COUNT];
}

static bool status_flags_are_safe(const int status_flags) {
    if ((status_flags & O_ACCMODE) != O_RDONLY || (status_flags & O_APPEND) != 0) {
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

static enum gate_e_execution_closure_held_fds_reason_v1 inspect_descriptor(
    const int descriptor,
    const uint64_t expected_byte_length,
    struct gate_e_execution_closure_object_identity_v1 *const identity
) {
    struct stat metadata;
    const int descriptor_flags = fcntl(descriptor, F_GETFD);
    const int status_flags = fcntl(descriptor, F_GETFL);

    if (descriptor < 3 || descriptor_flags < 0 || status_flags < 0) {
        return descriptor < 3 ? GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1
                              : GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_UNREADABLE_V1;
    }
    if ((descriptor_flags & FD_CLOEXEC) == 0 || !status_flags_are_safe(status_flags)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1;
    }
    if (fstat(descriptor, &metadata) != 0) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_UNREADABLE_V1;
    }
    if (!S_ISREG(metadata.st_mode) || metadata.st_nlink < 1) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_UNSAFE_V1;
    }
    if (metadata.st_size <= 0 || (uintmax_t)metadata.st_size != (uintmax_t)expected_byte_length) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_LENGTH_MISMATCH_V1;
    }
    *identity = identity_from_stat(&metadata);
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

static enum gate_e_execution_closure_held_fds_reason_v1 hash_descriptor(
    const int descriptor,
    const uint64_t expected_byte_length,
    unsigned char digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    struct sha256_context context;
    unsigned char buffer[HASH_BUFFER_BYTES];
    uint64_t offset = 0U;

    sha256_init(&context);
    while (offset < expected_byte_length) {
        const uint64_t remaining = expected_byte_length - offset;
        const size_t requested = remaining < (uint64_t)sizeof(buffer)
                                     ? (size_t)remaining
                                     : sizeof(buffer);
        const off_t position = (off_t)offset;
        const ssize_t count = pread(descriptor, buffer, requested, position);

        if (position < 0 || (uint64_t)position != offset || count <= 0) {
            return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_UNREADABLE_V1;
        }
        sha256_update(&context, buffer, (size_t)count);
        offset += (uint64_t)count;
    }
    sha256_final(&context, digest);
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

static enum gate_e_execution_closure_held_fds_reason_v1 pin_descriptor(
    const int descriptor,
    int *const duplicate
) {
    const int result = fcntl(descriptor, F_DUPFD_CLOEXEC, 3);

    if (result < 0) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_PIN_FAILED_V1;
    }
    *duplicate = result;
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

static bool candidate_contains_identity(
    const struct gate_e_execution_closure_held_fds_v1 *const candidate,
    const size_t accepted_count,
    const struct gate_e_execution_closure_object_identity_v1 *const identity
) {
    for (size_t index = 0U; index < accepted_count; ++index) {
        if (identities_alias(&held_file_at(candidate, index)->identity, identity)) {
            return true;
        }
    }
    return false;
}

static enum gate_e_execution_closure_held_fds_reason_v1 bind_one_descriptor(
    const int borrowed_descriptor,
    const struct gate_e_execution_closure_leaf *const declaration,
    struct gate_e_execution_closure_held_fds_v1 *const candidate,
    const size_t accepted_count
) {
    struct gate_e_execution_closure_object_identity_v1 before;
    struct gate_e_execution_closure_object_identity_v1 after;
    unsigned char observed_digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
    struct gate_e_execution_closure_held_file_v1 *const destination =
        mutable_held_file_at(candidate, accepted_count);
    enum gate_e_execution_closure_held_fds_reason_v1 result;
    int duplicate = -1;

    result = inspect_descriptor(borrowed_descriptor, declaration->byte_length, &before);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return result;
    }
    result = pin_descriptor(borrowed_descriptor, &duplicate);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return result;
    }
    result = inspect_descriptor(duplicate, declaration->byte_length, &before);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return close_after_failure(&duplicate, result);
    }
    result = hash_descriptor(duplicate, declaration->byte_length, observed_digest);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return close_after_failure(&duplicate, result);
    }
    result = inspect_descriptor(duplicate, declaration->byte_length, &after);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return close_after_failure(&duplicate, result);
    }
    if (!identities_equal(&before, &after)) {
        return close_after_failure(
            &duplicate, GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1
        );
    }
    if (memcmp(observed_digest, declaration->sha256, sizeof(observed_digest)) != 0) {
        return close_after_failure(
            &duplicate, GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1
        );
    }
    if (candidate_contains_identity(candidate, accepted_count, &after)) {
        return close_after_failure(
            &duplicate, GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_ALIAS_V1
        );
    }
    destination->descriptor = duplicate;
    destination->declaration = *declaration;
    destination->identity = after;
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

static bool input_descriptors_are_distinct(
    const struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed,
    const size_t total_count
) {
    for (size_t left = 0U; left < total_count; ++left) {
        const int left_descriptor = *borrowed_descriptor_at(borrowed, left);

        if (left_descriptor < 3) {
            return false;
        }
        for (size_t right = left + 1U; right < total_count; ++right) {
            if (left_descriptor == *borrowed_descriptor_at(borrowed, right)) {
                return false;
            }
        }
    }
    return true;
}

static bool output_binding_is_well_formed(
    const struct gate_e_execution_closure_held_fds_v1 *const held
) {
    size_t total_count;

    if (held == NULL || held->initialized_state != HELD_FDS_INITIALIZED_STATE ||
        held->runtime_leaf_count == 0U ||
        held->runtime_leaf_count > GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES ||
        !digest_is_nonzero(held->runtime_closure_sha256)) {
        return false;
    }
    total_count = FIXED_ROLE_COUNT + held->runtime_leaf_count;
    for (size_t index = 0U; index < total_count; ++index) {
        const struct gate_e_execution_closure_held_file_v1 *const file =
            held_file_at(held, index);

        if (!held_file_is_well_formed(file)) {
            return false;
        }
        for (size_t prior = 0U; prior < index; ++prior) {
            const struct gate_e_execution_closure_held_file_v1 *const previous =
                held_file_at(held, prior);

            if (file->descriptor == previous->descriptor ||
                identities_alias(&file->identity, &previous->identity)) {
                return false;
            }
        }
    }
    for (size_t index = held->runtime_leaf_count;
         index < GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES;
         ++index) {
        if (!held_file_is_empty(&held->runtime_leaves[index])) {
            return false;
        }
    }
    return true;
}

static enum gate_e_execution_closure_held_fds_reason_v1 recheck_one_descriptor(
    const struct gate_e_execution_closure_held_file_v1 *const file
) {
    struct gate_e_execution_closure_object_identity_v1 before;
    struct gate_e_execution_closure_object_identity_v1 after;
    unsigned char observed_digest[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES];
    enum gate_e_execution_closure_held_fds_reason_v1 result;
    int duplicate = -1;

    result = inspect_descriptor(file->descriptor, file->declaration.byte_length, &before);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return result;
    }
    if (!identities_equal(&before, &file->identity)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1;
    }
    result = pin_descriptor(file->descriptor, &duplicate);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        return result;
    }
    result = inspect_descriptor(duplicate, file->declaration.byte_length, &before);
    if (result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        result = hash_descriptor(duplicate, file->declaration.byte_length, observed_digest);
    }
    if (result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        result = inspect_descriptor(duplicate, file->declaration.byte_length, &after);
    }
    if (result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1 &&
        (!identities_equal(&before, &after) || !identities_equal(&after, &file->identity))) {
        result = GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_IDENTITY_DRIFT_V1;
    }
    if (result == GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1 &&
        memcmp(observed_digest, file->declaration.sha256, sizeof(observed_digest)) != 0) {
        result = GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_OBJECT_DIGEST_MISMATCH_V1;
    }
    if (!close_descriptor_once(&duplicate)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1;
    }
    return result;
}

enum gate_e_execution_closure_held_fds_reason_v1
gate_e_execution_closure_held_fds_v1_recheck(
    const struct gate_e_execution_closure_held_fds_v1 *const held
) {
    const size_t total_count =
        held == NULL ? 0U : FIXED_ROLE_COUNT + held->runtime_leaf_count;

    if (!output_binding_is_well_formed(held)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_BINDING_MISMATCH_V1;
    }
    for (size_t index = 0U; index < total_count; ++index) {
        const enum gate_e_execution_closure_held_fds_reason_v1 result =
            recheck_one_descriptor(held_file_at(held, index));

        if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
            return result;
        }
    }
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}

enum gate_e_execution_closure_held_fds_reason_v1
gate_e_bind_execution_closure_held_fds_v1(
    const unsigned char *const raw_manifest,
    const size_t raw_manifest_length,
    const struct gate_e_execution_closure_borrowed_fds_v1 *const borrowed,
    struct gate_e_execution_closure_held_fds_v1 *const output
) {
    struct gate_e_execution_closure_manifest parsed;
    struct gate_e_execution_closure_held_fds_v1 candidate;
    enum gate_e_execution_closure_reason parse_result;
    enum gate_e_execution_closure_held_fds_reason_v1 result;
    const size_t total_count =
        borrowed == NULL ? 0U : FIXED_ROLE_COUNT + borrowed->runtime_leaf_count;

    if (!output_is_initialized_empty(output) || borrowed == NULL ||
        (borrowed->runtime_leaf_count != 0U && borrowed->runtime_leaf_descriptors == NULL)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INVALID_ARGUMENT_V1;
    }
    gate_e_execution_closure_manifest_v1_init(&parsed);
    parse_result = gate_e_parse_execution_closure_manifest_v1(
        raw_manifest, raw_manifest_length, &parsed
    );
    if (parse_result != GATE_E_EXECUTION_CLOSURE_OK) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_MANIFEST_REJECTED_V1;
    }
    if (borrowed->runtime_leaf_count != parsed.runtime_leaf_count) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_RUNTIME_COUNT_MISMATCH_V1;
    }
    if (!input_descriptors_are_distinct(borrowed, total_count)) {
        return GATE_E_EXECUTION_CLOSURE_HELD_FDS_INPUT_DESCRIPTOR_ALIAS_V1;
    }
    gate_e_execution_closure_held_fds_v1_init(&candidate);
    candidate.runtime_leaf_count = parsed.runtime_leaf_count;
    memcpy(
        candidate.runtime_closure_sha256,
        parsed.runtime_closure_sha256,
        sizeof(candidate.runtime_closure_sha256)
    );
    for (size_t index = 0U; index < total_count; ++index) {
        result = bind_one_descriptor(
            *borrowed_descriptor_at(borrowed, index),
            parsed_leaf_at(&parsed, index),
            &candidate,
            index
        );
        if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
            if (gate_e_execution_closure_held_fds_v1_close(&candidate) !=
                GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
                return GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1;
            }
            return result;
        }
    }
    result = gate_e_execution_closure_held_fds_v1_recheck(&candidate);
    if (result != GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
        if (gate_e_execution_closure_held_fds_v1_close(&candidate) !=
            GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1) {
            return GATE_E_EXECUTION_CLOSURE_HELD_FDS_CLOSE_FAILED_V1;
        }
        return result;
    }
    *output = candidate;
    return GATE_E_EXECUTION_CLOSURE_HELD_FDS_OK_V1;
}
