/*
 * Pure C11 parser for the future Gate E execution-closure sidecar.
 *
 * This library accepts only caller-owned bytes in the one canonical v1 JSON
 * form. It has no filesystem, descriptor, process, loader, ELF, privilege,
 * cgroup, GPU, Docker, evidence, receipt, or qualification operation.
 */

#include "gate_e_execution_closure_manifest_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    SHA256_BLOCK_BYTES = 64,
    MAX_INTEGER_LITERAL_DIGITS = 19,
};

static const uint64_t MANIFEST_INITIALIZED_STATE = UINT64_C(0x52494c4559434d31);

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    unsigned char block[SHA256_BLOCK_BYTES];
    size_t block_length;
};

struct parser {
    const unsigned char *cursor;
    const unsigned char *end;
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
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU,
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

    for (size_t index = 0; index < 16U; ++index) {
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
    for (size_t index = 0; index < 64U; ++index) {
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

static void clear_manifest(struct gate_e_execution_closure_manifest *const output) {
    memset(&output->dynamic_loader, 0, sizeof(output->dynamic_loader));
    memset(&output->interpreter, 0, sizeof(output->interpreter));
    output->runtime_leaf_count = 0U;
    memset(output->runtime_leaves, 0, sizeof(output->runtime_leaves));
    memset(output->runtime_closure_sha256, 0, sizeof(output->runtime_closure_sha256));
}

void gate_e_execution_closure_manifest_v1_init(
    struct gate_e_execution_closure_manifest *const output
) {
    if (output == NULL) {
        return;
    }
    memset(output, 0, sizeof(*output));
    output->initialized_state = MANIFEST_INITIALIZED_STATE;
}

static bool take_literal(
    struct parser *const parser,
    const char *const literal,
    const size_t literal_length
) {
    if ((size_t)(parser->end - parser->cursor) < literal_length ||
        memcmp(parser->cursor, literal, literal_length) != 0) {
        return false;
    }
    parser->cursor += literal_length;
    return true;
}

static bool parse_ascii_string(
    struct parser *const parser,
    const unsigned char **const value,
    size_t *const value_length
) {
    const unsigned char *start;

    if (parser->cursor == parser->end || *parser->cursor != (unsigned char)'"') {
        return false;
    }
    ++parser->cursor;
    start = parser->cursor;
    while (parser->cursor != parser->end) {
        const unsigned char byte = *parser->cursor;

        if (byte == (unsigned char)'"') {
            *value = start;
            *value_length = (size_t)(parser->cursor - start);
            ++parser->cursor;
            return true;
        }
        if (byte < 0x20U || byte >= 0x80U || byte == (unsigned char)'\\') {
            return false;
        }
        ++parser->cursor;
    }
    return false;
}

static bool is_ascii_letter(const unsigned char byte) {
    return (byte >= (unsigned char)'a' && byte <= (unsigned char)'z') ||
           (byte >= (unsigned char)'A' && byte <= (unsigned char)'Z');
}

static bool is_ascii_digit(const unsigned char byte) {
    return byte >= (unsigned char)'0' && byte <= (unsigned char)'9';
}

static bool is_component_initial(const unsigned char byte) {
    return is_ascii_letter(byte) || is_ascii_digit(byte) || byte == (unsigned char)'_' ||
           byte == (unsigned char)'+' || byte == (unsigned char)'@' || byte == (unsigned char)'%' ||
           byte == (unsigned char)':' || byte == (unsigned char)'=' || byte == (unsigned char)',' ||
           byte == (unsigned char)'-';
}

static bool is_component_subsequent(const unsigned char byte) {
    return is_component_initial(byte) || byte == (unsigned char)'.';
}

static bool valid_audit_path(const unsigned char *const path, const size_t path_length) {
    bool component_initial = true;

    if (path_length < 2U || path_length > GATE_E_EXECUTION_CLOSURE_MAX_AUDIT_PATH_BYTES ||
        path[0] != (unsigned char)'/' || path[path_length - 1U] == (unsigned char)'/') {
        return false;
    }
    for (size_t index = 1U; index < path_length; ++index) {
        const unsigned char byte = path[index];

        if (byte == (unsigned char)'/') {
            if (component_initial) {
                return false;
            }
            component_initial = true;
        } else if (component_initial) {
            if (!is_component_initial(byte)) {
                return false;
            }
            component_initial = false;
        } else if (!is_component_subsequent(byte)) {
            return false;
        }
    }
    return !component_initial;
}

static bool parse_positive_decimal(struct parser *const parser, uint64_t *const value) {
    size_t digits = 0U;
    uint64_t parsed = 0U;
    bool leading_zero = false;

    if (parser->cursor == parser->end || !is_ascii_digit(*parser->cursor)) {
        return false;
    }
    leading_zero = *parser->cursor == (unsigned char)'0';
    while (parser->cursor != parser->end && is_ascii_digit(*parser->cursor)) {
        const unsigned int digit = (unsigned int)(*parser->cursor - (unsigned char)'0');

        if (digits >= MAX_INTEGER_LITERAL_DIGITS ||
            parsed > (UINT64_MAX - (uint64_t)digit) / UINT64_C(10)) {
            return false;
        }
        parsed = (parsed * UINT64_C(10)) + (uint64_t)digit;
        ++digits;
        ++parser->cursor;
    }
    if (leading_zero && digits != 1U) {
        return false;
    }
    *value = parsed;
    return true;
}

static int lower_hex_value(const unsigned char byte) {
    if (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') {
        return (int)(byte - (unsigned char)'0');
    }
    if (byte >= (unsigned char)'a' && byte <= (unsigned char)'f') {
        return (int)(byte - (unsigned char)'a') + 10;
    }
    return -1;
}

static enum gate_e_execution_closure_reason decode_sha256(
    const unsigned char *const text,
    const size_t text_length,
    unsigned char output[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES]
) {
    unsigned char nonzero = 0U;

    if (text_length != GATE_E_EXECUTION_CLOSURE_SHA256_BYTES * 2U) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_SHA256;
    }
    for (size_t index = 0U; index < GATE_E_EXECUTION_CLOSURE_SHA256_BYTES; ++index) {
        const int high = lower_hex_value(text[index * 2U]);
        const int low = lower_hex_value(text[(index * 2U) + 1U]);

        if (high < 0 || low < 0) {
            return GATE_E_EXECUTION_CLOSURE_INVALID_SHA256;
        }
        output[index] = (unsigned char)(((unsigned int)high << 4U) | (unsigned int)low);
        nonzero |= output[index];
    }
    return nonzero == 0U ? GATE_E_EXECUTION_CLOSURE_ZERO_SHA256 : GATE_E_EXECUTION_CLOSURE_OK;
}

static enum gate_e_execution_closure_reason parse_leaf(
    struct parser *const parser,
    struct gate_e_execution_closure_leaf *const output
) {
    const unsigned char *path;
    const unsigned char *sha256;
    size_t path_length;
    size_t sha256_length;
    uint64_t byte_length;
    enum gate_e_execution_closure_reason result;

    if (!take_literal(parser, "{\"audit_path\":", sizeof("{\"audit_path\":") - 1U) ||
        !parse_ascii_string(parser, &path, &path_length) ||
        !take_literal(parser, ",\"byte_length\":", sizeof(",\"byte_length\":") - 1U) ||
        !parse_positive_decimal(parser, &byte_length) ||
        !take_literal(parser, ",\"sha256\":", sizeof(",\"sha256\":") - 1U) ||
        !parse_ascii_string(parser, &sha256, &sha256_length) ||
        !take_literal(parser, "}", sizeof("}") - 1U)) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
    }
    if (!valid_audit_path(path, path_length)) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_AUDIT_PATH;
    }
    if (byte_length == 0U || byte_length > GATE_E_EXECUTION_CLOSURE_MAX_LEAF_BYTES) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_BYTE_LENGTH;
    }
    result = decode_sha256(sha256, sha256_length, output->sha256);
    if (result != GATE_E_EXECUTION_CLOSURE_OK) {
        return result;
    }
    memcpy(output->audit_path, path, path_length);
    output->audit_path[path_length] = 0U;
    output->audit_path_length = path_length;
    output->byte_length = byte_length;
    return GATE_E_EXECUTION_CLOSURE_OK;
}

static int compare_audit_paths(
    const struct gate_e_execution_closure_leaf *const left,
    const struct gate_e_execution_closure_leaf *const right
) {
    const size_t shared_length = left->audit_path_length < right->audit_path_length
                                     ? left->audit_path_length
                                     : right->audit_path_length;
    const int compared = memcmp(left->audit_path, right->audit_path, shared_length);

    if (compared != 0) {
        return compared;
    }
    if (left->audit_path_length < right->audit_path_length) {
        return -1;
    }
    if (left->audit_path_length > right->audit_path_length) {
        return 1;
    }
    return 0;
}

static enum gate_e_execution_closure_reason parse_manifest(
    const unsigned char *const raw,
    const size_t raw_length,
    struct gate_e_execution_closure_manifest *const output
) {
    struct parser parser = {.cursor = raw, .end = raw + raw_length - 1U};
    const unsigned char *schema_version;
    size_t schema_version_length;
    uint64_t total_bytes;
    struct sha256_context hash;
    enum gate_e_execution_closure_reason result;

    if (!take_literal(&parser, "{\"dynamic_loader\":", sizeof("{\"dynamic_loader\":") - 1U)) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
    }
    result = parse_leaf(&parser, &output->dynamic_loader);
    if (result != GATE_E_EXECUTION_CLOSURE_OK) {
        return result;
    }
    if (!take_literal(&parser, ",\"interpreter\":", sizeof(",\"interpreter\":") - 1U)) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
    }
    result = parse_leaf(&parser, &output->interpreter);
    if (result != GATE_E_EXECUTION_CLOSURE_OK) {
        return result;
    }
    if (compare_audit_paths(&output->dynamic_loader, &output->interpreter) == 0) {
        return GATE_E_EXECUTION_CLOSURE_DUPLICATE_AUDIT_PATH;
    }
    if (!take_literal(&parser, ",\"runtime_leaves\":[", sizeof(",\"runtime_leaves\":[") - 1U)) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
    }
    if (parser.cursor == parser.end || *parser.cursor == (unsigned char)']') {
        return GATE_E_EXECUTION_CLOSURE_INVALID_RUNTIME_LEAVES;
    }
    total_bytes = output->dynamic_loader.byte_length + output->interpreter.byte_length;
    for (;;) {
        struct gate_e_execution_closure_leaf *current;

        if (output->runtime_leaf_count == GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES) {
            return GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAF_BUDGET_EXCEEDED;
        }
        current = &output->runtime_leaves[output->runtime_leaf_count];
        result = parse_leaf(&parser, current);
        if (result != GATE_E_EXECUTION_CLOSURE_OK) {
            return result;
        }
        if ((output->runtime_leaf_count != 0U &&
             compare_audit_paths(
                 &output->runtime_leaves[output->runtime_leaf_count - 1U], current
             ) >= 0) ||
            compare_audit_paths(&output->dynamic_loader, current) == 0 ||
            compare_audit_paths(&output->interpreter, current) == 0) {
            return output->runtime_leaf_count != 0U &&
                           compare_audit_paths(
                               &output->runtime_leaves[output->runtime_leaf_count - 1U], current
                           ) >= 0
                       ? GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAVES_NOT_STRICTLY_SORTED
                       : GATE_E_EXECUTION_CLOSURE_DUPLICATE_AUDIT_PATH;
        }
        if (current->byte_length > GATE_E_EXECUTION_CLOSURE_MAX_CLOSURE_BYTES - total_bytes) {
            return GATE_E_EXECUTION_CLOSURE_CLOSURE_BYTE_BUDGET_EXCEEDED;
        }
        total_bytes += current->byte_length;
        ++output->runtime_leaf_count;
        if (parser.cursor == parser.end) {
            return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
        }
        if (*parser.cursor == (unsigned char)']') {
            ++parser.cursor;
            break;
        }
        if (*parser.cursor != (unsigned char)',') {
            return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
        }
        ++parser.cursor;
        if (output->runtime_leaf_count == GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES) {
            return GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAF_BUDGET_EXCEEDED;
        }
    }
    if (!take_literal(
            &parser,
            ",\"schema_version\":",
            sizeof(",\"schema_version\":") - 1U
        ) ||
        !parse_ascii_string(&parser, &schema_version, &schema_version_length) ||
        !take_literal(&parser, "}", sizeof("}") - 1U) || parser.cursor != parser.end) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON;
    }
    if (schema_version_length != sizeof(GATE_E_EXECUTION_CLOSURE_SCHEMA_VERSION) - 1U ||
        memcmp(
            schema_version,
            GATE_E_EXECUTION_CLOSURE_SCHEMA_VERSION,
            sizeof(GATE_E_EXECUTION_CLOSURE_SCHEMA_VERSION) - 1U
        ) != 0) {
        return GATE_E_EXECUTION_CLOSURE_UNSUPPORTED_SCHEMA_VERSION;
    }
    sha256_init(&hash);
    sha256_update(&hash, raw, raw_length);
    sha256_final(&hash, output->runtime_closure_sha256);
    return GATE_E_EXECUTION_CLOSURE_OK;
}

enum gate_e_execution_closure_reason
gate_e_parse_execution_closure_manifest_v1(
    const unsigned char *const raw,
    const size_t raw_length,
    struct gate_e_execution_closure_manifest *const output
) {
    struct gate_e_execution_closure_manifest candidate;
    enum gate_e_execution_closure_reason result;

    if (output == NULL || output->initialized_state != MANIFEST_INITIALIZED_STATE) {
        return GATE_E_EXECUTION_CLOSURE_INVALID_ARGUMENT;
    }
    if (raw == NULL || raw_length == 0U || raw_length > GATE_E_EXECUTION_CLOSURE_MAX_MANIFEST_BYTES) {
        clear_manifest(output);
        return GATE_E_EXECUTION_CLOSURE_INVALID_MANIFEST_BYTE_LENGTH;
    }
    if (raw[raw_length - 1U] != (unsigned char)'\n' ||
        (raw_length > 1U && raw[raw_length - 2U] == (unsigned char)'\n')) {
        clear_manifest(output);
        return GATE_E_EXECUTION_CLOSURE_INVALID_TERMINAL_NEWLINE;
    }

    gate_e_execution_closure_manifest_v1_init(&candidate);
    result = parse_manifest(raw, raw_length, &candidate);
    if (result != GATE_E_EXECUTION_CLOSURE_OK) {
        clear_manifest(output);
        return result;
    }
    *output = candidate;
    return GATE_E_EXECUTION_CLOSURE_OK;
}
