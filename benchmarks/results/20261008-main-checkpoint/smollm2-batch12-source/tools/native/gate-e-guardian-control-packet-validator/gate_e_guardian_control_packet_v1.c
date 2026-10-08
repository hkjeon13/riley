/*
 * Pure C11 validator for the future Gate E guardian control-packet contract.
 *
 * It accepts only caller-owned bytes and a caller-owned active-session
 * binding. It has no transport, filesystem, descriptor, process, privilege,
 * cgroup, GPU, Docker, evidence, receipt, or qualification operation.
 */

#include "gate_e_guardian_control_packet_v1.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    SHA256_HEX_BYTES = GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES * 2U,
};

static const uint64_t BINDING_INITIALIZED_STATE = UINT64_C(0x52494c4559474342);

struct parser {
    const unsigned char *cursor;
    const unsigned char *end;
};

struct parsed_packet {
    enum gate_e_guardian_control_packet_kind kind;
    uint64_t generation;
    unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char guardian_contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
};

static void clear_binding(struct gate_e_guardian_control_packet_binding *const binding) {
    binding->generation = 0U;
    memset(binding->boot_id, 0, sizeof(binding->boot_id));
    memset(binding->lease_id, 0, sizeof(binding->lease_id));
    memset(binding->nonce, 0, sizeof(binding->nonce));
    memset(
        binding->guardian_contract_sha256,
        0,
        sizeof(binding->guardian_contract_sha256)
    );
}

void gate_e_guardian_control_packet_binding_v1_init(
    struct gate_e_guardian_control_packet_binding *const binding
) {
    if (binding == NULL) {
        return;
    }
    memset(binding, 0, sizeof(*binding));
    binding->initialized_state = BINDING_INITIALIZED_STATE;
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

static bool is_ascii_digit(const unsigned char byte) {
    return byte >= (unsigned char)'0' && byte <= (unsigned char)'9';
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

static bool nonzero_digest(
    const unsigned char digest[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES]
) {
    unsigned char nonzero = 0U;

    for (size_t index = 0U; index < GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES; ++index) {
        nonzero |= digest[index];
    }
    return nonzero != 0U;
}

static bool valid_expected_binding(
    const struct gate_e_guardian_control_packet_binding *const expected_binding
) {
    return expected_binding != NULL &&
           expected_binding->initialized_state == BINDING_INITIALIZED_STATE &&
           expected_binding->generation != 0U &&
           nonzero_digest(expected_binding->boot_id) &&
           nonzero_digest(expected_binding->lease_id) &&
           nonzero_digest(expected_binding->nonce) &&
           nonzero_digest(expected_binding->guardian_contract_sha256);
}

enum gate_e_guardian_control_packet_reason
gate_e_guardian_control_packet_binding_v1_set(
    struct gate_e_guardian_control_packet_binding *const binding,
    const uint64_t generation,
    const unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const unsigned char guardian_contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES]
) {
    struct gate_e_guardian_control_packet_binding candidate;

    if (binding == NULL || binding->initialized_state != BINDING_INITIALIZED_STATE) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_ARGUMENT;
    }
    if (generation == 0U || boot_id == NULL || lease_id == NULL || nonce == NULL ||
        guardian_contract_sha256 == NULL || !nonzero_digest(boot_id) ||
        !nonzero_digest(lease_id) || !nonzero_digest(nonce) ||
        !nonzero_digest(guardian_contract_sha256)) {
        clear_binding(binding);
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING;
    }
    gate_e_guardian_control_packet_binding_v1_init(&candidate);
    candidate.generation = generation;
    memcpy(candidate.boot_id, boot_id, sizeof(candidate.boot_id));
    memcpy(candidate.lease_id, lease_id, sizeof(candidate.lease_id));
    memcpy(candidate.nonce, nonce, sizeof(candidate.nonce));
    memcpy(
        candidate.guardian_contract_sha256,
        guardian_contract_sha256,
        sizeof(candidate.guardian_contract_sha256)
    );
    *binding = candidate;
    return GATE_E_GUARDIAN_CONTROL_PACKET_OK;
}

static enum gate_e_guardian_control_packet_reason parse_sha256(
    struct parser *const parser,
    unsigned char output[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES]
) {
    unsigned char nonzero = 0U;

    if ((size_t)(parser->end - parser->cursor) < (size_t)SHA256_HEX_BYTES + 1U) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256;
    }
    for (size_t index = 0U; index < GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES; ++index) {
        const int high = lower_hex_value(parser->cursor[index * 2U]);
        const int low = lower_hex_value(parser->cursor[(index * 2U) + 1U]);

        if (high < 0 || low < 0) {
            return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256;
        }
        output[index] = (unsigned char)(((unsigned int)high << 4U) | (unsigned int)low);
        nonzero |= output[index];
    }
    if (parser->cursor[SHA256_HEX_BYTES] != (unsigned char)'"') {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256;
    }
    parser->cursor += (size_t)SHA256_HEX_BYTES + 1U;
    return nonzero == 0U ? GATE_E_GUARDIAN_CONTROL_PACKET_ZERO_SHA256
                         : GATE_E_GUARDIAN_CONTROL_PACKET_OK;
}

static enum gate_e_guardian_control_packet_reason parse_positive_generation(
    struct parser *const parser,
    uint64_t *const output
) {
    uint64_t value = 0U;
    bool leading_zero;

    if (parser->cursor == parser->end || !is_ascii_digit(*parser->cursor)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION;
    }
    leading_zero = *parser->cursor == (unsigned char)'0';
    while (parser->cursor != parser->end && is_ascii_digit(*parser->cursor)) {
        const unsigned int digit = (unsigned int)(*parser->cursor - (unsigned char)'0');

        if (value > (UINT64_MAX - (uint64_t)digit) / UINT64_C(10)) {
            return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION;
        }
        value = (value * UINT64_C(10)) + (uint64_t)digit;
        ++parser->cursor;
    }
    if (leading_zero || value == 0U) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION;
    }
    *output = value;
    return GATE_E_GUARDIAN_CONTROL_PACKET_OK;
}

static enum gate_e_guardian_control_packet_reason parse_kind(
    struct parser *const parser,
    enum gate_e_guardian_control_packet_kind *const output
) {
    if (take_literal(parser, "ready", sizeof("ready") - 1U)) {
        *output = GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY;
        return GATE_E_GUARDIAN_CONTROL_PACKET_OK;
    }
    if (take_literal(
            parser,
            "no_action_complete",
            sizeof("no_action_complete") - 1U
        )) {
        *output = GATE_E_GUARDIAN_CONTROL_PACKET_KIND_NO_ACTION_COMPLETE;
        return GATE_E_GUARDIAN_CONTROL_PACKET_OK;
    }
    return GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_KIND;
}

static enum gate_e_guardian_control_packet_reason parse_packet(
    const unsigned char *const raw,
    const size_t raw_length,
    struct parsed_packet *const output
) {
    struct parser parser = {.cursor = raw, .end = raw + raw_length};
    enum gate_e_guardian_control_packet_reason result;

    if (!take_literal(&parser, "{\"boot_id\":\"", sizeof("{\"boot_id\":\"") - 1U)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_sha256(&parser, output->boot_id);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(
            &parser,
            ",\"contract_sha256\":\"",
            sizeof(",\"contract_sha256\":\"") - 1U
        )) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_sha256(&parser, output->guardian_contract_sha256);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(&parser, ",\"generation\":", sizeof(",\"generation\":") - 1U)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_positive_generation(&parser, &output->generation);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(&parser, ",\"kind\":\"", sizeof(",\"kind\":\"") - 1U)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_kind(&parser, &output->kind);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(&parser, "\",\"lease_id\":\"", sizeof("\",\"lease_id\":\"") - 1U)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_sha256(&parser, output->lease_id);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(&parser, ",\"nonce\":\"", sizeof(",\"nonce\":\"") - 1U)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    result = parse_sha256(&parser, output->nonce);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (!take_literal(
            &parser,
            ",\"schema_version\":\"",
            sizeof(",\"schema_version\":\"") - 1U
        )) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    if (!take_literal(
            &parser,
            GATE_E_GUARDIAN_CONTROL_PACKET_SCHEMA_VERSION,
            sizeof(GATE_E_GUARDIAN_CONTROL_PACKET_SCHEMA_VERSION) - 1U
        )) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_SCHEMA_VERSION;
    }
    if (!take_literal(&parser, "\"}", sizeof("\"}") - 1U) || parser.cursor != parser.end) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON;
    }
    return GATE_E_GUARDIAN_CONTROL_PACKET_OK;
}

static bool equal_binding(
    const struct parsed_packet *const packet,
    const struct gate_e_guardian_control_packet_binding *const expected_binding
) {
    return packet->generation == expected_binding->generation &&
           memcmp(packet->boot_id, expected_binding->boot_id, sizeof(packet->boot_id)) == 0 &&
           memcmp(packet->lease_id, expected_binding->lease_id, sizeof(packet->lease_id)) == 0 &&
           memcmp(packet->nonce, expected_binding->nonce, sizeof(packet->nonce)) == 0 &&
           memcmp(
               packet->guardian_contract_sha256,
               expected_binding->guardian_contract_sha256,
               sizeof(packet->guardian_contract_sha256)
           ) == 0;
}

enum gate_e_guardian_control_packet_reason
gate_e_validate_guardian_control_packet_v1(
    const unsigned char *const raw,
    const size_t raw_length,
    const enum gate_e_guardian_control_packet_kind expected_kind,
    const struct gate_e_guardian_control_packet_binding *const expected_binding
) {
    struct parsed_packet candidate = {0};
    enum gate_e_guardian_control_packet_reason result;

    if (raw == NULL || raw_length == 0U || raw_length > GATE_E_GUARDIAN_CONTROL_PACKET_MAX_BYTES) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_PACKET_BYTE_LENGTH;
    }
    if (expected_kind != GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY &&
        expected_kind != GATE_E_GUARDIAN_CONTROL_PACKET_KIND_NO_ACTION_COMPLETE) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_ARGUMENT;
    }
    if (!valid_expected_binding(expected_binding)) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING;
    }
    result = parse_packet(raw, raw_length, &candidate);
    if (result != GATE_E_GUARDIAN_CONTROL_PACKET_OK) {
        return result;
    }
    if (candidate.kind != expected_kind) {
        return GATE_E_GUARDIAN_CONTROL_PACKET_UNEXPECTED_KIND;
    }
    return equal_binding(&candidate, expected_binding)
               ? GATE_E_GUARDIAN_CONTROL_PACKET_OK
               : GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH;
}
