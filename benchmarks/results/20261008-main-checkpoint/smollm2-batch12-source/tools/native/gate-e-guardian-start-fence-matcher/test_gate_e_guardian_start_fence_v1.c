#include "gate_e_guardian_start_fence_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void fill_boot_id(
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES],
    const unsigned char first_byte,
    const unsigned char last_byte
) {
    memset(boot_id, 0, GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES);
    boot_id[0] = first_byte;
    boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES - 1U] = last_byte;
}

static struct gate_e_guardian_start_fence_candidate make_candidate(
    const unsigned char first_byte,
    const unsigned char last_byte,
    const uint64_t generation
) {
    struct gate_e_guardian_start_fence_candidate candidate = {0};

    fill_boot_id(candidate.boot_id, first_byte, last_byte);
    candidate.generation = generation;
    return candidate;
}

static void assert_zero_bytes(const void *const value, const size_t length) {
    const unsigned char *const bytes = value;

    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_cleared_binding(
    const struct gate_e_guardian_start_fence_binding *const binding
) {
    assert(binding->mode == GATE_E_GUARDIAN_START_FENCE_MODE_INVALID);
    assert_zero_bytes(binding->boot_id, sizeof(binding->boot_id));
    assert(binding->highest_generation == 0U);
}

static void assert_reason(
    const struct gate_e_guardian_start_fence_binding *const binding,
    const struct gate_e_guardian_start_fence_candidate *const candidate,
    const enum gate_e_guardian_start_fence_reason expected
) {
    assert(gate_e_match_guardian_start_fence_v1(binding, candidate) == expected);
}

static void test_valid_paths_are_repeatable_and_nonmutating(void) {
    struct gate_e_guardian_start_fence_binding binding;
    const unsigned char empty_boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES] = {0};
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    struct gate_e_guardian_start_fence_candidate candidate;
    struct gate_e_guardian_start_fence_binding binding_before;
    struct gate_e_guardian_start_fence_candidate candidate_before;

    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED,
            empty_boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate((unsigned char)0x11U, (unsigned char)0x12U, UINT64_C(1));
    binding_before = binding;
    candidate_before = candidate;
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_OK);
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_OK);
    assert(memcmp(&binding, &binding_before, sizeof(binding)) == 0);
    assert(memcmp(&candidate, &candidate_before, sizeof(candidate)) == 0);

    fill_boot_id(boot_id, (unsigned char)0x21U, (unsigned char)0x22U);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            UINT64_C(7)
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate((unsigned char)0x21U, (unsigned char)0x22U, UINT64_C(8));
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_OK);
}

static void test_boot_and_generation_replay_fail_closed(void) {
    struct gate_e_guardian_start_fence_binding binding;
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    struct gate_e_guardian_start_fence_candidate candidate;

    fill_boot_id(boot_id, (unsigned char)0x31U, (unsigned char)0x32U);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            UINT64_C(7)
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate((unsigned char)0x31U, (unsigned char)0x32U, UINT64_C(7));
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_GENERATION_REPLAYED);
    candidate.generation = UINT64_C(6);
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_GENERATION_REPLAYED);
    candidate = make_candidate((unsigned char)0x41U, (unsigned char)0x42U, UINT64_MAX);
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_DURABLE_RECOVERY_REQUIRED);
}

static void test_fixed_width_generation_boundaries(void) {
    struct gate_e_guardian_start_fence_binding binding;
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    struct gate_e_guardian_start_fence_candidate candidate;

    fill_boot_id(boot_id, (unsigned char)0x51U, (unsigned char)0x52U);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate((unsigned char)0x51U, (unsigned char)0x52U, UINT64_MAX);
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_OK);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            UINT64_MAX
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_GENERATION_REPLAYED);
}

static void test_invalid_inputs_clear_a_valid_binding(void) {
    struct gate_e_guardian_start_fence_binding binding;
    struct gate_e_guardian_start_fence_binding uninitialized = {0};
    unsigned char boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES];
    const unsigned char empty_boot_id[GATE_E_GUARDIAN_START_FENCE_BOOT_ID_BYTES] = {0};
    struct gate_e_guardian_start_fence_candidate candidate;

    fill_boot_id(boot_id, (unsigned char)0x61U, (unsigned char)0x62U);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            UINT64_C(1)
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE
    );
    assert_cleared_binding(&binding);
    candidate = make_candidate((unsigned char)0x61U, (unsigned char)0x62U, UINT64_C(2));
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE);

    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED,
            empty_boot_id,
            UINT64_C(1)
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE
    );
    assert_cleared_binding(&binding);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            empty_boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE
    );
    assert_cleared_binding(&binding);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            (enum gate_e_guardian_start_fence_mode)99,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE
    );
    assert_cleared_binding(&binding);
    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            NULL,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE
    );
    assert_cleared_binding(&binding);

    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &uninitialized,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_ARGUMENT
    );
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            NULL,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_INVALID_ARGUMENT
    );
    assert_reason(&uninitialized, &candidate, GATE_E_GUARDIAN_START_FENCE_INVALID_EXPECTED_FENCE);
    gate_e_guardian_start_fence_binding_v1_init(NULL);

    gate_e_guardian_start_fence_binding_v1_init(&binding);
    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_FENCED,
            boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate(0U, 0U, UINT64_C(1));
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_INVALID_CANDIDATE);
    candidate = make_candidate((unsigned char)0x61U, (unsigned char)0x62U, 0U);
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_INVALID_CANDIDATE);
    assert_reason(&binding, NULL, GATE_E_GUARDIAN_START_FENCE_INVALID_CANDIDATE);

    assert(
        gate_e_guardian_start_fence_binding_v1_set(
            &binding,
            GATE_E_GUARDIAN_START_FENCE_MODE_UNFENCED,
            empty_boot_id,
            0U
        ) == GATE_E_GUARDIAN_START_FENCE_OK
    );
    candidate = make_candidate((unsigned char)0x71U, (unsigned char)0x72U, UINT64_C(1));
    assert_reason(&binding, &candidate, GATE_E_GUARDIAN_START_FENCE_OK);
}

int main(void) {
    test_valid_paths_are_repeatable_and_nonmutating();
    test_boot_and_generation_replay_fail_closed();
    test_fixed_width_generation_boundaries();
    test_invalid_inputs_clear_a_valid_binding();
    return 0;
}
