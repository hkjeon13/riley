#include "gate_e_guardian_control_packet_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static const unsigned char valid_ready_packet[] =
    "{\"boot_id\":\"0756b868aec39c81b1029e69a47ab516f073928e44cc662156525db06a78ea90\","
    "\"contract_sha256\":\"7f458091b87e519e52e2fb07b767cc984667c820a77d9dc5578390f39690e35a\","
    "\"generation\":7,\"kind\":\"ready\","
    "\"lease_id\":\"a5ca22b9a0490b2c5bc023e0d7f467a24c9f3711932c94f525f080572bd6ab12\","
    "\"nonce\":\"78377b525757b494427f89014f97d79928f3938d14eb51e20fb5dec9834eb304\","
    "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\"}";

static const unsigned char valid_complete_packet[] =
    "{\"boot_id\":\"0756b868aec39c81b1029e69a47ab516f073928e44cc662156525db06a78ea90\","
    "\"contract_sha256\":\"7f458091b87e519e52e2fb07b767cc984667c820a77d9dc5578390f39690e35a\","
    "\"generation\":7,\"kind\":\"no_action_complete\","
    "\"lease_id\":\"a5ca22b9a0490b2c5bc023e0d7f467a24c9f3711932c94f525f080572bd6ab12\","
    "\"nonce\":\"78377b525757b494427f89014f97d79928f3938d14eb51e20fb5dec9834eb304\","
    "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\"}";

static const char fixture_boot_id[] =
    "0756b868aec39c81b1029e69a47ab516f073928e44cc662156525db06a78ea90";
static const char fixture_lease_id[] =
    "a5ca22b9a0490b2c5bc023e0d7f467a24c9f3711932c94f525f080572bd6ab12";
static const char fixture_nonce[] =
    "78377b525757b494427f89014f97d79928f3938d14eb51e20fb5dec9834eb304";
static const char fixture_contract[] =
    "7f458091b87e519e52e2fb07b767cc984667c820a77d9dc5578390f39690e35a";

static void assert_zero_bytes(const void *const value, const size_t length) {
    const unsigned char *const bytes = value;

    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_cleared_binding(
    const struct gate_e_guardian_control_packet_binding *const binding
) {
    assert(binding->generation == 0U);
    assert_zero_bytes(binding->boot_id, sizeof(binding->boot_id));
    assert_zero_bytes(binding->lease_id, sizeof(binding->lease_id));
    assert_zero_bytes(binding->nonce, sizeof(binding->nonce));
    assert_zero_bytes(
        binding->guardian_contract_sha256,
        sizeof(binding->guardian_contract_sha256)
    );
}

static unsigned int fixture_hex_value(const char byte) {
    if (byte >= '0' && byte <= '9') {
        return (unsigned int)(byte - '0');
    }
    assert(byte >= 'a' && byte <= 'f');
    return (unsigned int)(byte - 'a') + 10U;
}

static void decode_fixture_digest(
    unsigned char destination[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    const char text[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES * 2U + 1U]
) {
    assert(strlen(text) == GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES * 2U);
    for (size_t index = 0U; index < GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES; ++index) {
        destination[index] = (unsigned char)(
            (fixture_hex_value(text[index * 2U]) << 4U) |
            fixture_hex_value(text[(index * 2U) + 1U])
        );
    }
}

static void fixture_digests(
    unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES],
    unsigned char contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES]
) {
    decode_fixture_digest(boot_id, fixture_boot_id);
    decode_fixture_digest(lease_id, fixture_lease_id);
    decode_fixture_digest(nonce, fixture_nonce);
    decode_fixture_digest(contract_sha256, fixture_contract);
}

static struct gate_e_guardian_control_packet_binding make_binding(const uint64_t generation) {
    struct gate_e_guardian_control_packet_binding binding;
    unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];

    fixture_digests(boot_id, lease_id, nonce, contract_sha256);
    gate_e_guardian_control_packet_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            generation,
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    return binding;
}

static unsigned char *find_fragment(
    unsigned char *const buffer,
    const size_t buffer_length,
    const char *const fragment
) {
    const size_t fragment_length = strlen(fragment);

    assert(fragment_length != 0U);
    assert(buffer_length >= fragment_length);
    for (size_t index = 0U; index <= buffer_length - fragment_length; ++index) {
        if (memcmp(buffer + index, fragment, fragment_length) == 0) {
            return buffer + index;
        }
    }
    return NULL;
}

static size_t replace_fragment(
    unsigned char *const buffer,
    const size_t buffer_length,
    const size_t capacity,
    const char *const old_fragment,
    const char *const new_fragment
) {
    const size_t old_length = strlen(old_fragment);
    const size_t new_length = strlen(new_fragment);
    unsigned char *const location = find_fragment(buffer, buffer_length, old_fragment);
    size_t offset;
    size_t new_total;

    assert(location != NULL);
    offset = (size_t)(location - buffer);
    new_total = buffer_length - old_length + new_length;
    assert(new_total <= capacity);
    memmove(
        location + new_length,
        location + old_length,
        buffer_length - offset - old_length
    );
    memcpy(location, new_fragment, new_length);
    return new_total;
}

static void assert_rejected(
    const unsigned char *const raw,
    const size_t raw_length,
    const enum gate_e_guardian_control_packet_kind expected_kind,
    const struct gate_e_guardian_control_packet_binding *const binding,
    const enum gate_e_guardian_control_packet_reason expected_reason
) {
    assert(
        gate_e_validate_guardian_control_packet_v1(raw, raw_length, expected_kind, binding) ==
        expected_reason
    );
}

static void assert_not_accepted(
    const unsigned char *const raw,
    const size_t raw_length,
    const enum gate_e_guardian_control_packet_kind expected_kind,
    const struct gate_e_guardian_control_packet_binding *const binding
) {
    assert(
        gate_e_validate_guardian_control_packet_v1(raw, raw_length, expected_kind, binding) !=
        GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
}

static void test_valid_packets_do_not_mutate_input(void) {
    struct gate_e_guardian_control_packet_binding binding = make_binding(UINT64_C(7));
    unsigned char ready_copy[sizeof(valid_ready_packet)];
    unsigned char complete_copy[sizeof(valid_complete_packet)];

    memcpy(ready_copy, valid_ready_packet, sizeof(ready_copy));
    memcpy(complete_copy, valid_complete_packet, sizeof(complete_copy));
    assert(
        gate_e_validate_guardian_control_packet_v1(
            ready_copy,
            sizeof(ready_copy) - 1U,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
            &binding
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    assert(memcmp(ready_copy, valid_ready_packet, sizeof(ready_copy)) == 0);
    assert(
        gate_e_validate_guardian_control_packet_v1(
            complete_copy,
            sizeof(complete_copy) - 1U,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_NO_ACTION_COMPLETE,
            &binding
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    assert(memcmp(complete_copy, valid_complete_packet, sizeof(complete_copy)) == 0);
}

static void test_uint64_generation_and_numeric_rejections(void) {
    unsigned char raw[GATE_E_GUARDIAN_CONTROL_PACKET_MAX_BYTES + 1U];
    size_t raw_length = sizeof(valid_ready_packet) - 1U;
    struct gate_e_guardian_control_packet_binding binding;

    memcpy(raw, valid_ready_packet, raw_length);
    raw_length = replace_fragment(
        raw,
        raw_length,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":18446744073709551615"
    );
    binding = make_binding(UINT64_MAX);
    assert(
        gate_e_validate_guardian_control_packet_v1(
            raw,
            raw_length,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
            &binding
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":0"
    );
    binding = make_binding(UINT64_C(7));
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":01"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":true"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":1.0"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":1e0"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":18446744073709551616"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":-1"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );

    memcpy(raw, valid_ready_packet, sizeof(valid_ready_packet) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_ready_packet) - 1U,
        sizeof(raw),
        "\"generation\":7",
        "\"generation\":+1"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_GENERATION
    );
}

static void test_canonical_shape_sha_schema_and_kind_rejections(void) {
    unsigned char raw[GATE_E_GUARDIAN_CONTROL_PACKET_MAX_BYTES + 1U];
    const size_t fixture_length = sizeof(valid_ready_packet) - 1U;
    size_t raw_length;
    unsigned char *location;
    struct gate_e_guardian_control_packet_binding binding = make_binding(UINT64_C(7));
    static const unsigned char reordered_full_packet[] =
        "{\"contract_sha256\":\"7f458091b87e519e52e2fb07b767cc984667c820a77d9dc5578390f39690e35a\","
        "\"boot_id\":\"0756b868aec39c81b1029e69a47ab516f073928e44cc662156525db06a78ea90\","
        "\"generation\":7,\"kind\":\"ready\","
        "\"lease_id\":\"a5ca22b9a0490b2c5bc023e0d7f467a24c9f3711932c94f525f080572bd6ab12\","
        "\"nonce\":\"78377b525757b494427f89014f97d79928f3938d14eb51e20fb5dec9834eb304\","
        "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\"}";

    memcpy(raw, valid_ready_packet, fixture_length);
    raw[fixture_length] = (unsigned char)'\n';
    assert_rejected(
        raw,
        fixture_length + 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw[0] = (unsigned char)' ';
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "{\"boot_id\":\"",
        "{\"boot_id\": \""
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );
    assert_rejected(
        reordered_full_packet,
        sizeof(reordered_full_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\"}",
        "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\",\"boot_id\":\"0756b868aec39c81b1029e69a47ab516f073928e44cc662156525db06a78ea90\"}"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\"}",
        "\"schema_version\":\"riley.rc3-gate-e-guardian-control.v1\",\"extra\":0}"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    location = find_fragment(raw, fixture_length, "\"boot_id\"");
    assert(location != NULL);
    location[6] = (unsigned char)'x';
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    location = find_fragment(raw, fixture_length, fixture_boot_id);
    assert(location != NULL);
    location[0] = (unsigned char)'A';
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "\"kind\":\"ready\"",
        "\"kind\":\"re\\u0061dy\""
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_KIND
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw[1] = 0U;
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    location = find_fragment(raw, fixture_length, fixture_boot_id);
    assert(location != NULL);
    memset(location, '0', GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES * 2U);
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_ZERO_SHA256
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    location = find_fragment(raw, fixture_length, fixture_boot_id);
    assert(location != NULL);
    location[0] = 0x80U;
    assert_rejected(
        raw,
        fixture_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_SHA256
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "riley.rc3-gate-e-guardian-control.v1",
        "riley.rc3-gate-e-guardian-control.v2"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_SCHEMA_VERSION
    );

    memcpy(raw, valid_ready_packet, fixture_length);
    raw_length = replace_fragment(
        raw,
        fixture_length,
        sizeof(raw),
        "\"kind\":\"ready\"",
        "\"kind\":\"READY\""
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_UNSUPPORTED_KIND
    );
}

static void test_expected_binding_and_kind_mismatches(void) {
    struct gate_e_guardian_control_packet_binding binding = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding stale_generation = make_binding(UINT64_C(8));
    struct gate_e_guardian_control_packet_binding changed_boot_id = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding changed_lease_id = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding changed_nonce = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding changed_contract = make_binding(UINT64_C(7));

    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_NO_ACTION_COMPLETE,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_UNEXPECTED_KIND
    );
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &stale_generation,
        GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH
    );
    changed_boot_id.boot_id[0] ^= 0x01U;
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &changed_boot_id,
        GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH
    );
    changed_lease_id.lease_id[0] ^= 0x01U;
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &changed_lease_id,
        GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH
    );
    changed_nonce.nonce[0] ^= 0x01U;
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &changed_nonce,
        GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH
    );
    changed_contract.guardian_contract_sha256[0] ^= 0x01U;
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &changed_contract,
        GATE_E_GUARDIAN_CONTROL_PACKET_BINDING_MISMATCH
    );
}

static void test_binding_set_snapshots_cross_aliased_inputs(void) {
    struct gate_e_guardian_control_packet_binding original = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding binding = make_binding(UINT64_C(7));

    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            binding.lease_id,
            binding.boot_id,
            binding.nonce,
            binding.guardian_contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    assert(memcmp(binding.boot_id, original.lease_id, sizeof(binding.boot_id)) == 0);
    assert(memcmp(binding.lease_id, original.boot_id, sizeof(binding.lease_id)) == 0);
    assert(memcmp(binding.nonce, original.nonce, sizeof(binding.nonce)) == 0);
    assert(
        memcmp(
            binding.guardian_contract_sha256,
            original.guardian_contract_sha256,
            sizeof(binding.guardian_contract_sha256)
        ) == 0
    );
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            binding.lease_id,
            binding.boot_id,
            binding.nonce,
            binding.guardian_contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    assert(
        gate_e_validate_guardian_control_packet_v1(
            valid_ready_packet,
            sizeof(valid_ready_packet) - 1U,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
            &binding
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
}

static void test_binding_setup_and_bounds_fail_closed(void) {
    struct gate_e_guardian_control_packet_binding binding = make_binding(UINT64_C(7));
    struct gate_e_guardian_control_packet_binding invalid_binding;
    struct gate_e_guardian_control_packet_binding valid_binding;
    unsigned char boot_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char lease_id[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char nonce[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char contract_sha256[GATE_E_GUARDIAN_CONTROL_PACKET_SHA256_BYTES];
    unsigned char too_large[GATE_E_GUARDIAN_CONTROL_PACKET_MAX_BYTES + 1U];

    fixture_digests(boot_id, lease_id, nonce, contract_sha256);
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            0U,
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING
    );

    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );
    assert(
        gate_e_validate_guardian_control_packet_v1(
            valid_ready_packet,
            sizeof(valid_ready_packet) - 1U,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
            &binding
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );

    memset(boot_id, 0, sizeof(boot_id));
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    fixture_digests(boot_id, lease_id, nonce, contract_sha256);
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_OK
    );

    gate_e_guardian_control_packet_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &binding,
            UINT64_C(7),
            NULL,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    memset(&invalid_binding, 0, sizeof(invalid_binding));
    assert(
        gate_e_guardian_control_packet_binding_v1_set(
            &invalid_binding,
            UINT64_C(7),
            boot_id,
            lease_id,
            nonce,
            contract_sha256
        ) == GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_ARGUMENT
    );
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &invalid_binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_EXPECTED_BINDING
    );
    valid_binding = make_binding(UINT64_C(7));
    assert_rejected(
        valid_ready_packet,
        sizeof(valid_ready_packet) - 1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_INVALID,
        &valid_binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_ARGUMENT
    );
    assert_rejected(
        valid_ready_packet,
        0U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &valid_binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_PACKET_BYTE_LENGTH
    );
    assert_rejected(
        NULL,
        1U,
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &valid_binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_PACKET_BYTE_LENGTH
    );
    for (size_t length = 1U; length < sizeof(valid_ready_packet) - 1U; ++length) {
        assert_not_accepted(
            valid_ready_packet,
            length,
            GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
            &valid_binding
        );
    }
    memset(too_large, 'x', sizeof(too_large));
    assert_rejected(
        too_large,
        sizeof(too_large),
        GATE_E_GUARDIAN_CONTROL_PACKET_KIND_READY,
        &valid_binding,
        GATE_E_GUARDIAN_CONTROL_PACKET_INVALID_PACKET_BYTE_LENGTH
    );
}

int main(void) {
    test_valid_packets_do_not_mutate_input();
    test_uint64_generation_and_numeric_rejections();
    test_canonical_shape_sha_schema_and_kind_rejections();
    test_expected_binding_and_kind_mismatches();
    test_binding_set_snapshots_cross_aliased_inputs();
    test_binding_setup_and_bounds_fail_closed();
    return 0;
}
