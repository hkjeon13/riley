#include "gate_e_execution_closure_manifest_v1.c"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const unsigned char valid_manifest[] =
    "{\"dynamic_loader\":{\"audit_path\":\"/lib64/ld-linux-x86-64.so.2\",\"byte_length\":210968,"
    "\"sha256\":\"5535c54aeb6ecbda2a12cea4d81e6ea582dee37356601ee87ddeee3844eca042\"},"
    "\"interpreter\":{\"audit_path\":\"/usr/bin/python3.10\",\"byte_length\":5917224,"
    "\"sha256\":\"5317a27786b53351485f427ffc031a740f27f00f1729ab5800e2be756037ed83\"},"
    "\"runtime_leaves\":[{\"audit_path\":\"/lib/x86_64-linux-gnu/libc.so.6\",\"byte_length\":2022344,"
    "\"sha256\":\"16c8c6eb85e05438f5d6c60ff9869072a3a3b1618aa1481ac7a0cb049f06f51d\"},"
    "{\"audit_path\":\"/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0\",\"byte_length\":5776912,"
    "\"sha256\":\"14505fb3b83499d2ddb00549acb361e368d4cbfeb69c6753a1bbf5711a4e78ee\"}],"
    "\"schema_version\":\"riley.rc3-gate-e-execution-closure-manifest.v1\"}\n";

static const unsigned char expected_manifest_sha256[GATE_E_EXECUTION_CLOSURE_SHA256_BYTES] = {
    0xf1U, 0xe8U, 0x5fU, 0xe0U, 0xfeU, 0x83U, 0x10U, 0x36U,
    0x07U, 0xc2U, 0xd2U, 0x37U, 0x46U, 0xc2U, 0x57U, 0x57U,
    0x19U, 0x70U, 0x38U, 0xc9U, 0x97U, 0x14U, 0x89U, 0xd2U,
    0xb8U, 0x66U, 0x05U, 0xe3U, 0x6fU, 0x8eU, 0x69U, 0x81U,
};

static const char digest_one[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
static const char digest_two[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
static const char digest_three[] =
    "3333333333333333333333333333333333333333333333333333333333333333";

static void assert_zero_bytes(const void *const value, const size_t length) {
    const unsigned char *const bytes = value;

    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_empty(const struct gate_e_execution_closure_manifest *const manifest) {
    assert(manifest->runtime_leaf_count == 0U);
    assert_zero_bytes(&manifest->dynamic_loader, sizeof(manifest->dynamic_loader));
    assert_zero_bytes(&manifest->interpreter, sizeof(manifest->interpreter));
    assert_zero_bytes(manifest->runtime_leaves, sizeof(manifest->runtime_leaves));
    assert_zero_bytes(
        manifest->runtime_closure_sha256,
        sizeof(manifest->runtime_closure_sha256)
    );
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
    const enum gate_e_execution_closure_reason expected
) {
    struct gate_e_execution_closure_manifest output;

    gate_e_execution_closure_manifest_v1_init(&output);
    assert(gate_e_parse_execution_closure_manifest_v1(raw, raw_length, &output) == expected);
    assert_empty(&output);
}

static size_t append_runtime_leaf(
    unsigned char *const destination,
    const size_t capacity,
    const size_t position,
    const unsigned long long value,
    const size_t index
) {
    const int written = snprintf(
        (char *)destination + position,
        capacity - position,
        "%s{\"audit_path\":\"/runtime/lib%03zu.so\",\"byte_length\":%llu,\"sha256\":\"%s\"}",
        index == 0U ? "" : ",",
        index,
        value,
        digest_three
    );

    assert(written >= 0);
    assert((size_t)written < capacity - position);
    return position + (size_t)written;
}

static size_t make_manifest(
    unsigned char *const destination,
    const size_t capacity,
    const size_t runtime_count,
    const uint64_t runtime_byte_length
) {
    int written;
    size_t position;

    written = snprintf(
        (char *)destination,
        capacity,
        "{\"dynamic_loader\":{\"audit_path\":\"/loader\",\"byte_length\":1,\"sha256\":\"%s\"},"
        "\"interpreter\":{\"audit_path\":\"/python\",\"byte_length\":1,\"sha256\":\"%s\"},"
        "\"runtime_leaves\":[",
        digest_one,
        digest_two
    );
    assert(written >= 0);
    assert((size_t)written < capacity);
    position = (size_t)written;
    for (size_t index = 0U; index < runtime_count; ++index) {
        position = append_runtime_leaf(
            destination,
            capacity,
            position,
            (unsigned long long)runtime_byte_length,
            index
        );
    }
    written = snprintf(
        (char *)destination + position,
        capacity - position,
        "],\"schema_version\":\"riley.rc3-gate-e-execution-closure-manifest.v1\"}\n"
    );
    assert(written >= 0);
    assert((size_t)written < capacity - position);
    return position + (size_t)written;
}

static void test_valid_canonical_manifest_and_raw_digest(void) {
    struct gate_e_execution_closure_manifest output;
    unsigned char input_copy[sizeof(valid_manifest)];

    memcpy(input_copy, valid_manifest, sizeof(valid_manifest));
    gate_e_execution_closure_manifest_v1_init(&output);
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            input_copy,
            sizeof(input_copy) - 1U,
            &output
        ) == GATE_E_EXECUTION_CLOSURE_OK
    );
    assert(memcmp(input_copy, valid_manifest, sizeof(valid_manifest)) == 0);
    assert(output.runtime_leaf_count == 2U);
    assert(strcmp((const char *)output.dynamic_loader.audit_path, "/lib64/ld-linux-x86-64.so.2") == 0);
    assert(output.dynamic_loader.byte_length == UINT64_C(210968));
    assert(strcmp((const char *)output.interpreter.audit_path, "/usr/bin/python3.10") == 0);
    assert(output.interpreter.byte_length == UINT64_C(5917224));
    assert(strcmp((const char *)output.runtime_leaves[0].audit_path, "/lib/x86_64-linux-gnu/libc.so.6") == 0);
    assert(
        strcmp(
            (const char *)output.runtime_leaves[1].audit_path,
            "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0"
        ) == 0
    );
    assert(
        memcmp(
            output.runtime_closure_sha256,
            expected_manifest_sha256,
            sizeof(expected_manifest_sha256)
        ) == 0
    );
}

static void test_terminal_newline_canonical_shape_and_output_reset(void) {
    unsigned char raw[sizeof(valid_manifest) + 64U];
    const size_t raw_length = sizeof(valid_manifest) - 1U;
    struct gate_e_execution_closure_manifest output;
    static const unsigned char duplicate_key[] =
        "{\"dynamic_loader\":null,\"dynamic_loader\":null}\n";
    static const unsigned char reordered_key[] = "{\"interpreter\":null}\n";

    memcpy(raw, valid_manifest, raw_length);
    raw[raw_length - 1U] = (unsigned char)' ';
    assert_rejected(
        raw,
        raw_length,
        GATE_E_EXECUTION_CLOSURE_INVALID_TERMINAL_NEWLINE
    );
    memcpy(raw, valid_manifest, raw_length);
    raw[raw_length] = (unsigned char)'\n';
    assert_rejected(
        raw,
        raw_length + 1U,
        GATE_E_EXECUTION_CLOSURE_INVALID_TERMINAL_NEWLINE
    );
    memcpy(raw, valid_manifest, raw_length);
    raw[1] = (unsigned char)' ';
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);
    assert_rejected(
        duplicate_key,
        sizeof(duplicate_key) - 1U,
        GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON
    );
    assert_rejected(
        reordered_key,
        sizeof(reordered_key) - 1U,
        GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON
    );

    gate_e_execution_closure_manifest_v1_init(&output);
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            valid_manifest,
            raw_length,
            &output
        ) == GATE_E_EXECUTION_CLOSURE_OK
    );
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            raw,
            raw_length,
            &output
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON
    );
    assert_empty(&output);
}

static void test_numbers_sha_and_paths_fail_closed(void) {
    unsigned char raw[sizeof(valid_manifest) + 96U];
    const size_t raw_length = sizeof(valid_manifest) - 1U;
    unsigned char *location;
    static const unsigned char oversized_leaf[] =
        "{\"dynamic_loader\":{\"audit_path\":\"/a\",\"byte_length\":536870913,"
        "\"sha256\":\"1111111111111111111111111111111111111111111111111111111111111111\"}}\n";
    static const unsigned char giant_integer_leaf[] =
        "{\"dynamic_loader\":{\"audit_path\":\"/a\",\"byte_length\":99999999999999999999,"
        "\"sha256\":\"1111111111111111111111111111111111111111111111111111111111\"}}\n";

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "210968");
    assert(location != NULL);
    memcpy(location, "010968", sizeof("010968") - 1U);
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "210968");
    assert(location != NULL);
    memcpy(location, "210.68", sizeof("210.68") - 1U);
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "210968");
    assert(location != NULL);
    memcpy(location, "1e0000", sizeof("1e0000") - 1U);
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);
    assert_rejected(
        oversized_leaf,
        sizeof(oversized_leaf) - 1U,
        GATE_E_EXECUTION_CLOSURE_INVALID_BYTE_LENGTH
    );
    assert_rejected(
        giant_integer_leaf,
        sizeof(giant_integer_leaf) - 1U,
        GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON
    );

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "\"/usr/bin/python3.10\"");
    assert(location != NULL);
    location[2] = (unsigned char)'.';
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_AUDIT_PATH);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "\"/usr/bin/python3.10\"");
    assert(location != NULL);
    location[2] = (unsigned char)'\\';
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(raw, raw_length, "\"/usr/bin/python3.10\"");
    assert(location != NULL);
    location[2] = 0x80U;
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_CANONICAL_JSON);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(
        raw,
        raw_length,
        "5535c54aeb6ecbda2a12cea4d81e6ea582dee37356601ee87ddeee3844eca042"
    );
    assert(location != NULL);
    location[0] = (unsigned char)'A';
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_INVALID_SHA256);

    memcpy(raw, valid_manifest, raw_length);
    location = find_fragment(
        raw,
        raw_length,
        "5535c54aeb6ecbda2a12cea4d81e6ea582dee37356601ee87ddeee3844eca042"
    );
    assert(location != NULL);
    memset(location, '0', GATE_E_EXECUTION_CLOSURE_SHA256_BYTES * 2U);
    assert_rejected(raw, raw_length, GATE_E_EXECUTION_CLOSURE_ZERO_SHA256);
}

static void test_runtime_sort_uniqueness_and_budgets(void) {
    unsigned char raw[GATE_E_EXECUTION_CLOSURE_MAX_MANIFEST_BYTES + 1U];
    size_t raw_length;
    unsigned char *location;

    memcpy(raw, valid_manifest, sizeof(valid_manifest) - 1U);
    location = find_fragment(raw, sizeof(valid_manifest) - 1U, "\"/lib/x86_64-linux-gnu/libc.so.6\"");
    assert(location != NULL);
    location[2] = (unsigned char)'z';
    assert_rejected(
        raw,
        sizeof(valid_manifest) - 1U,
        GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAVES_NOT_STRICTLY_SORTED
    );

    memcpy(raw, valid_manifest, sizeof(valid_manifest) - 1U);
    raw_length = replace_fragment(
        raw,
        sizeof(valid_manifest) - 1U,
        sizeof(raw),
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/lib64/ld-linux-x86-64.so.2"
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_EXECUTION_CLOSURE_DUPLICATE_AUDIT_PATH
    );

    raw_length = make_manifest(raw, sizeof(raw), 0U, UINT64_C(1));
    assert_rejected(
        raw,
        raw_length,
        GATE_E_EXECUTION_CLOSURE_INVALID_RUNTIME_LEAVES
    );

    raw_length = make_manifest(
        raw,
        sizeof(raw),
        GATE_E_EXECUTION_CLOSURE_MAX_RUNTIME_LEAVES + 1U,
        UINT64_C(1)
    );
    assert(raw_length < GATE_E_EXECUTION_CLOSURE_MAX_MANIFEST_BYTES);
    assert_rejected(
        raw,
        raw_length,
        GATE_E_EXECUTION_CLOSURE_RUNTIME_LEAF_BUDGET_EXCEEDED
    );

    raw_length = make_manifest(
        raw,
        sizeof(raw),
        4U,
        GATE_E_EXECUTION_CLOSURE_MAX_LEAF_BYTES
    );
    assert_rejected(
        raw,
        raw_length,
        GATE_E_EXECUTION_CLOSURE_CLOSURE_BYTE_BUDGET_EXCEEDED
    );
}

static void test_bounds_and_initialization_fail_closed(void) {
    unsigned char too_large[GATE_E_EXECUTION_CLOSURE_MAX_MANIFEST_BYTES + 1U];
    struct gate_e_execution_closure_manifest output;
    struct gate_e_execution_closure_manifest uninitialized;

    memset(too_large, 'x', sizeof(too_large));
    gate_e_execution_closure_manifest_v1_init(&output);
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            too_large,
            sizeof(too_large),
            &output
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_MANIFEST_BYTE_LENGTH
    );
    assert_empty(&output);
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            valid_manifest,
            sizeof(valid_manifest) - 7U,
            &output
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_TERMINAL_NEWLINE
    );
    assert_empty(&output);
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            NULL,
            1U,
            &output
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_MANIFEST_BYTE_LENGTH
    );
    assert_empty(&output);
    memset(&uninitialized, 0, sizeof(uninitialized));
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            valid_manifest,
            sizeof(valid_manifest) - 1U,
            &uninitialized
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_ARGUMENT
    );
    assert(
        gate_e_parse_execution_closure_manifest_v1(
            valid_manifest,
            sizeof(valid_manifest) - 1U,
            NULL
        ) == GATE_E_EXECUTION_CLOSURE_INVALID_ARGUMENT
    );
}

int main(void) {
    test_valid_canonical_manifest_and_raw_digest();
    test_terminal_newline_canonical_shape_and_output_reset();
    test_numbers_sha_and_paths_fail_closed();
    test_runtime_sort_uniqueness_and_budgets();
    test_bounds_and_initialization_fail_closed();
    return 0;
}
