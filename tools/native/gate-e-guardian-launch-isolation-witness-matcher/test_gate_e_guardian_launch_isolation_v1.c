#include "gate_e_guardian_launch_isolation_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void fill_token(
    unsigned char token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES],
    const unsigned char first_byte,
    const unsigned char last_byte
) {
    memset(token, 0, GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES);
    token[0] = first_byte;
    token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES - 1U] = last_byte;
}

static void make_expected_tokens(
    unsigned char bootstrap[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES],
    unsigned char core[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES]
) {
    fill_token(bootstrap, (unsigned char)0x31U, (unsigned char)0x32U);
    fill_token(core, (unsigned char)0x41U, (unsigned char)0x42U);
}

static struct gate_e_guardian_launch_isolation_binding make_binding(void) {
    struct gate_e_guardian_launch_isolation_binding binding;
    unsigned char bootstrap[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    unsigned char core[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];

    make_expected_tokens(bootstrap, core);
    gate_e_guardian_launch_isolation_binding_v1_init(&binding);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&binding, bootstrap, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK
    );
    return binding;
}

static struct gate_e_guardian_launch_isolation_report make_report(void) {
    struct gate_e_guardian_launch_isolation_report report = {0};
    unsigned char bootstrap[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    unsigned char core[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];

    make_expected_tokens(bootstrap, core);
    report.argv = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_FUTURE_BOOTSTRAP_V1;
    report.environment = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_EMPTY;
    report.bootstrap_inherited_fd_mask = GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MASK;
    report.worker_inherited_fd_mask = GATE_E_GUARDIAN_LAUNCH_ISOLATION_STDIO_FD_MASK;
    report.bootstrap_inherited_fd_count = GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT;
    report.worker_inherited_fd_count = GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT;
    report.bootstrap_highest_inherited_fd = GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_MAX_FD;
    report.worker_highest_inherited_fd = GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_MAX_FD;
    report.bootstrap_fd_number = GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD;
    memcpy(report.bootstrap_fd_token, bootstrap, sizeof(report.bootstrap_fd_token));
    report.bootstrap_fd_is_sealed = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    report.bootstrap_fd_carries_lease = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    report.bootstrap_fd_carries_cgroup_control = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    report.core_fd_number = GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD;
    memcpy(report.core_fd_token, core, sizeof(report.core_fd_token));
    report.core_fd_is_sealed = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    report.core_fd_consumed_before_worker = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    report.core_fd_inherited_by_worker = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    report.lease_fd_inherited = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    report.cgroup_control_fd_inherited = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    report.no_new_privs = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    report.capabilities = GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_EMPTY;
    return report;
}

static void assert_zero_bytes(const void *const value, const size_t length) {
    const unsigned char *const bytes = value;

    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_cleared_binding(
    const struct gate_e_guardian_launch_isolation_binding *const binding
) {
    assert_zero_bytes(
        binding->expected_bootstrap_held_fd_token,
        sizeof(binding->expected_bootstrap_held_fd_token)
    );
    assert_zero_bytes(
        binding->expected_core_held_fd_token,
        sizeof(binding->expected_core_held_fd_token)
    );
}

static void assert_reason(
    const struct gate_e_guardian_launch_isolation_binding *const binding,
    const struct gate_e_guardian_launch_isolation_report *const report,
    const enum gate_e_guardian_launch_isolation_reason expected
) {
    assert(gate_e_match_guardian_launch_isolation_v1(binding, report) == expected);
}

static void test_valid_profile_is_repeatable_and_allows_equal_tokens(void) {
    const struct gate_e_guardian_launch_isolation_binding binding = make_binding();
    const struct gate_e_guardian_launch_isolation_binding binding_before = binding;
    const struct gate_e_guardian_launch_isolation_report report = make_report();
    const struct gate_e_guardian_launch_isolation_report report_before = report;
    struct gate_e_guardian_launch_isolation_binding equal_token_binding;
    struct gate_e_guardian_launch_isolation_report equal_token_report;
    unsigned char token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];

    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK);
    assert(memcmp(&binding, &binding_before, sizeof(binding)) == 0);
    assert(memcmp(&report, &report_before, sizeof(report)) == 0);

    fill_token(token, (unsigned char)0x51U, (unsigned char)0x52U);
    gate_e_guardian_launch_isolation_binding_v1_init(&equal_token_binding);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(
            &equal_token_binding,
            token,
            token
        ) == GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK
    );
    equal_token_report = make_report();
    memcpy(
        equal_token_report.bootstrap_fd_token,
        token,
        sizeof(equal_token_report.bootstrap_fd_token)
    );
    memcpy(equal_token_report.core_fd_token, token, sizeof(equal_token_report.core_fd_token));
    assert_reason(
        &equal_token_binding,
        &equal_token_report,
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK
    );
}

static void test_profile_shape_and_token_mismatches_fail_closed(void) {
    const struct gate_e_guardian_launch_isolation_binding binding = make_binding();
    struct gate_e_guardian_launch_isolation_report report;

    report = make_report();
    report.argv = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGV_CLAIM);
    report = make_report();
    report.argv = (enum gate_e_guardian_launch_isolation_argv_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGV_CLAIM);
    report = make_report();
    report.argv = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ARGV_OTHER;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_UNEXPECTED_ARGV);
    report = make_report();
    report.environment = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ENVIRONMENT_CLAIM);
    report = make_report();
    report.environment = (enum gate_e_guardian_launch_isolation_environment_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ENVIRONMENT_CLAIM);
    report = make_report();
    report.environment = GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_ENVIRONMENT_NOT_EMPTY);
    report = make_report();
    /* A normalized set containing extra FD 64 has six entries. */
    ++report.bootstrap_inherited_fd_count;
    report.bootstrap_highest_inherited_fd = UINT32_C(64);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_COUNT_MISMATCH);
    report = make_report();
    ++report.worker_inherited_fd_count;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_COUNT_MISMATCH);
    report = make_report();
    report.bootstrap_highest_inherited_fd = UINT32_C(64);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_MAXIMUM_MISMATCH);
    report = make_report();
    report.worker_highest_inherited_fd = UINT32_MAX;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_MAXIMUM_MISMATCH);
    report = make_report();
    report.bootstrap_inherited_fd_mask &=
        ~(UINT64_C(1) << GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_SET_MISMATCH);
    report = make_report();
    report.bootstrap_inherited_fd_mask |= UINT64_C(1) << 33;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_SET_MISMATCH);
    report = make_report();
    report.worker_inherited_fd_mask |= UINT64_C(1) << GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_WORKER_FD_SET_MISMATCH);
    report = make_report();
    report.bootstrap_fd_number = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NUMBER_MISMATCH);
    report = make_report();
    report.bootstrap_fd_number = GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NUMBER_MISMATCH);
    report = make_report();
    report.core_fd_number = GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NUMBER_MISMATCH);
    report = make_report();
    report.core_fd_number = UINT32_MAX;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NUMBER_MISMATCH);
    report = make_report();
    report.bootstrap_fd_token[0] ^= 0x01U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_TOKEN_MISMATCH);
    report = make_report();
    report.bootstrap_fd_token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES - 1U] ^= 0x01U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_TOKEN_MISMATCH);
    report = make_report();
    report.core_fd_token[0] ^= 0x01U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_TOKEN_MISMATCH);
    report = make_report();
    report.core_fd_token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES - 1U] ^= 0x01U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_TOKEN_MISMATCH);
}

static void test_truth_and_capability_claims_fail_closed(void) {
    const struct gate_e_guardian_launch_isolation_binding binding = make_binding();
    struct gate_e_guardian_launch_isolation_report report;

    report = make_report();
    report.bootstrap_fd_is_sealed = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_NOT_SEALED);
    report = make_report();
    report.bootstrap_fd_carries_lease = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_LEASE);
    report = make_report();
    report.bootstrap_fd_carries_cgroup_control = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_BOOTSTRAP_FD_CARRIES_CGROUP_CONTROL
    );
    report = make_report();
    report.core_fd_is_sealed = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_SEALED);
    report = make_report();
    report.core_fd_consumed_before_worker = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_NOT_CONSUMED_BEFORE_WORKER
    );
    report = make_report();
    report.core_fd_inherited_by_worker = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CORE_FD_INHERITED_BY_WORKER);
    report = make_report();
    report.lease_fd_inherited = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_LEASE_FD_INHERITED);
    report = make_report();
    report.cgroup_control_fd_inherited = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_YES;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_CGROUP_CONTROL_FD_INHERITED
    );
    report = make_report();
    report.no_new_privs = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_NO;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_NO_NEW_PRIVS_NOT_SET);
    report = make_report();
    report.no_new_privs = GATE_E_GUARDIAN_LAUNCH_ISOLATION_TRUTH_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.bootstrap_fd_is_sealed = (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.bootstrap_fd_carries_lease = (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.bootstrap_fd_carries_cgroup_control =
        (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.core_fd_is_sealed = (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.core_fd_consumed_before_worker =
        (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.core_fd_inherited_by_worker =
        (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.lease_fd_inherited = (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.cgroup_control_fd_inherited =
        (enum gate_e_guardian_launch_isolation_truth_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_TRUTH_CLAIM);
    report = make_report();
    report.capabilities = GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_CAPABILITY_CLAIM);
    report = make_report();
    report.capabilities = (enum gate_e_guardian_launch_isolation_capability_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_CAPABILITY_CLAIM);
    report = make_report();
    report.capabilities = GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_CAPABILITIES_NOT_EMPTY);
}

static void test_binding_and_null_fail_closed(void) {
    struct gate_e_guardian_launch_isolation_binding binding = make_binding();
    struct gate_e_guardian_launch_isolation_binding uninitialized = {0};
    struct gate_e_guardian_launch_isolation_binding valid_binding;
    struct gate_e_guardian_launch_isolation_report report = make_report();
    unsigned char bootstrap[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    unsigned char core[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES];
    const unsigned char zero_token[GATE_E_GUARDIAN_LAUNCH_ISOLATION_TOKEN_BYTES] = {0};

    make_expected_tokens(bootstrap, core);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&binding, zero_token, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&binding, bootstrap, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK
    );
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_OK);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&binding, bootstrap, zero_token) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    gate_e_guardian_launch_isolation_binding_v1_init(&binding);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&binding, NULL, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(&uninitialized, bootstrap, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT
    );
    assert(
        gate_e_guardian_launch_isolation_binding_v1_set(NULL, bootstrap, core) ==
        GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT
    );
    assert_reason(&uninitialized, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING);
    valid_binding = make_binding();
    assert_reason(&valid_binding, NULL, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_ARGUMENT);
    gate_e_guardian_launch_isolation_binding_v1_init(NULL);

    binding = make_binding();
    memset(
        binding.expected_bootstrap_held_fd_token,
        0,
        sizeof(binding.expected_bootstrap_held_fd_token)
    );
    assert_reason(&binding, &report, GATE_E_GUARDIAN_LAUNCH_ISOLATION_INVALID_EXPECTED_BINDING);
}

int main(void) {
    test_valid_profile_is_repeatable_and_allows_equal_tokens();
    test_profile_shape_and_token_mismatches_fail_closed();
    test_truth_and_capability_claims_fail_closed();
    test_binding_and_null_fail_closed();
    return 0;
}
