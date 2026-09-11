#include "gate_e_guardian_drain_witness_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void fill_token(
    unsigned char token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES],
    const unsigned char first_byte
) {
    memset(token, 0, GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES);
    token[0] = first_byte;
}

static struct gate_e_guardian_drain_witness_process_identity make_controller(void) {
    struct gate_e_guardian_drain_witness_process_identity controller = {0};

    controller.pid = UINT32_C(1);
    controller.starttime_ticks = UINT64_C(10100);
    controller.uid = 0U;
    controller.gid = 0U;
    fill_token(controller.pidfd_token, (unsigned char)0x31U);
    return controller;
}

static struct gate_e_guardian_drain_witness_cgroup_claim make_cgroup_claim(void) {
    struct gate_e_guardian_drain_witness_cgroup_claim claim = {0};

    claim.identity.st_dev = UINT64_C(1001);
    claim.identity.st_ino = UINT64_C(1002);
    claim.non_delegated = true;
    fill_token(claim.identity.held_fd_token, (unsigned char)0x32U);
    return claim;
}

static void make_worker_terminal_token(
    unsigned char token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES]
) {
    fill_token(token, (unsigned char)0x33U);
}

static struct gate_e_guardian_drain_witness_report make_report(void) {
    struct gate_e_guardian_drain_witness_report report = {0};

    report.controller = make_controller();
    report.cgroup = make_cgroup_claim();
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_EMPTY;
    report.terminal_pidfd_token_count = UINT32_C(1);
    make_worker_terminal_token(report.terminal_pidfd_token);
    return report;
}

static struct gate_e_guardian_drain_witness_binding make_binding(void) {
    struct gate_e_guardian_drain_witness_binding binding;
    const struct gate_e_guardian_drain_witness_process_identity controller = make_controller();
    const struct gate_e_guardian_drain_witness_cgroup_claim cgroup = make_cgroup_claim();
    unsigned char worker_terminal_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];

    make_worker_terminal_token(worker_terminal_token);
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_OK
    );
    return binding;
}

static void assert_zero_bytes(const void *const value, const size_t length) {
    const unsigned char *const bytes = value;

    for (size_t index = 0U; index < length; ++index) {
        assert(bytes[index] == 0U);
    }
}

static void assert_cleared_binding(
    const struct gate_e_guardian_drain_witness_binding *const binding
) {
    assert_zero_bytes(
        &binding->expected_pid1_controller,
        sizeof(binding->expected_pid1_controller)
    );
    assert_zero_bytes(
        binding->registered_worker_terminal_pidfd_token,
        sizeof(binding->registered_worker_terminal_pidfd_token)
    );
    assert_zero_bytes(&binding->held_cgroup, sizeof(binding->held_cgroup));
}

static void assert_process_equal(
    const struct gate_e_guardian_drain_witness_process_identity *const left,
    const struct gate_e_guardian_drain_witness_process_identity *const right
) {
    assert(left->pid == right->pid);
    assert(left->starttime_ticks == right->starttime_ticks);
    assert(left->uid == right->uid);
    assert(left->gid == right->gid);
    assert(memcmp(left->pidfd_token, right->pidfd_token, sizeof(left->pidfd_token)) == 0);
}

static void assert_cgroup_equal(
    const struct gate_e_guardian_drain_witness_cgroup_claim *const left,
    const struct gate_e_guardian_drain_witness_cgroup_claim *const right
) {
    assert(left->identity.st_dev == right->identity.st_dev);
    assert(left->identity.st_ino == right->identity.st_ino);
    assert(
        memcmp(
            left->identity.held_fd_token,
            right->identity.held_fd_token,
            sizeof(left->identity.held_fd_token)
        ) == 0
    );
    assert(left->non_delegated == right->non_delegated);
}

static void assert_reason(
    const struct gate_e_guardian_drain_witness_binding *const binding,
    const struct gate_e_guardian_drain_witness_report *const report,
    const enum gate_e_guardian_drain_witness_reason expected
) {
    assert(gate_e_match_guardian_drain_witness_v1(binding, report) == expected);
}

static void test_valid_report_is_repeatable_and_nonmutating(void) {
    const struct gate_e_guardian_drain_witness_binding binding = make_binding();
    const struct gate_e_guardian_drain_witness_binding binding_before = binding;
    const struct gate_e_guardian_drain_witness_report report = make_report();
    const struct gate_e_guardian_drain_witness_report report_before = report;

    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_OK);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_OK);
    assert_process_equal(
        &binding.expected_pid1_controller,
        &binding_before.expected_pid1_controller
    );
    assert(
        memcmp(
            binding.registered_worker_terminal_pidfd_token,
            binding_before.registered_worker_terminal_pidfd_token,
            sizeof(binding.registered_worker_terminal_pidfd_token)
        ) == 0
    );
    assert(binding.held_cgroup.st_dev == binding_before.held_cgroup.st_dev);
    assert(binding.held_cgroup.st_ino == binding_before.held_cgroup.st_ino);
    assert(
        memcmp(
            binding.held_cgroup.held_fd_token,
            binding_before.held_cgroup.held_fd_token,
            sizeof(binding.held_cgroup.held_fd_token)
        ) == 0
    );
    assert_process_equal(&report.controller, &report_before.controller);
    assert_cgroup_equal(&report.cgroup, &report_before.cgroup);
    assert(report.population == report_before.population);
    assert(report.terminal_pidfd_token_count == report_before.terminal_pidfd_token_count);
    assert(
        memcmp(
            report.terminal_pidfd_token,
            report_before.terminal_pidfd_token,
            sizeof(report.terminal_pidfd_token)
        ) == 0
    );
}

static void test_fixed_width_boundary_values_are_accepted(void) {
    struct gate_e_guardian_drain_witness_process_identity controller = make_controller();
    struct gate_e_guardian_drain_witness_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_drain_witness_binding binding;
    struct gate_e_guardian_drain_witness_report report;
    unsigned char worker_terminal_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];

    controller.starttime_ticks = UINT64_MAX;
    cgroup.identity.st_dev = UINT64_MAX;
    cgroup.identity.st_ino = UINT64_MAX;
    make_worker_terminal_token(worker_terminal_token);
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_OK
    );
    report = make_report();
    report.controller = controller;
    report.cgroup = cgroup;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_OK);
}

static void test_each_claim_mismatch_fails_closed(void) {
    const struct gate_e_guardian_drain_witness_binding binding = make_binding();
    struct gate_e_guardian_drain_witness_report report;

    report = make_report();
    report.controller.pid = UINT32_C(2);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PID_MISMATCH);
    report = make_report();
    ++report.controller.starttime_ticks;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_STARTTIME_TICKS_MISMATCH
    );
    report = make_report();
    report.controller.pidfd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    report.controller.pidfd_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES - 1U] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    report.controller.uid = UINT32_C(1);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_UID_MISMATCH);
    report = make_report();
    report.controller.gid = UINT32_C(1);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_GID_MISMATCH);

    report = make_report();
    report.cgroup.non_delegated = false;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
    report = make_report();
    ++report.cgroup.identity.st_dev;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_DEV_MISMATCH);
    report = make_report();
    ++report.cgroup.identity.st_ino;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_INO_MISMATCH);
    report = make_report();
    report.cgroup.identity.held_fd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH
    );
    report = make_report();
    report.cgroup.identity.held_fd_token[
        GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH
    );

    report = make_report();
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_STILL_POPULATED);
    report = make_report();
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_POPULATION_CLAIM);
    report = make_report();
    report.population = (enum gate_e_guardian_drain_witness_population_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_POPULATION_CLAIM);
    report = make_report();
    report.terminal_pidfd_token_count = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_TERMINAL_TOKEN_COUNT);
    report = make_report();
    report.terminal_pidfd_token_count = UINT32_C(2);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_TERMINAL_TOKEN_COUNT);
    report = make_report();
    report.terminal_pidfd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    report.terminal_pidfd_token[
        GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH
    );
}

static void test_invalid_reports_and_precedence_fail_closed(void) {
    const struct gate_e_guardian_drain_witness_binding binding = make_binding();
    struct gate_e_guardian_drain_witness_report report;

    report = make_report();
    report.controller.pid = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.controller.pid = GATE_E_GUARDIAN_DRAIN_WITNESS_MAX_PID + UINT32_C(1);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.controller.starttime_ticks = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    memset(report.controller.pidfd_token, 0, sizeof(report.controller.pidfd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.cgroup.identity.st_dev = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    report.cgroup.identity.st_ino = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    memset(report.cgroup.identity.held_fd_token, 0, sizeof(report.cgroup.identity.held_fd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_SUPPLIED_CGROUP);

    report = make_report();
    report.controller.pid = UINT32_C(2);
    report.cgroup.non_delegated = false;
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT;
    report.terminal_pidfd_token_count = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CONTROLLER_PID_MISMATCH);
    report = make_report();
    report.cgroup.non_delegated = false;
    report.cgroup.identity.st_dev = 0U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_DRAIN_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
    report = make_report();
    ++report.cgroup.identity.st_dev;
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_ST_DEV_MISMATCH);
    report = make_report();
    report.population = GATE_E_GUARDIAN_DRAIN_WITNESS_POPULATION_PRESENT;
    report.terminal_pidfd_token_count = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_CGROUP_STILL_POPULATED);
}

static void test_binding_setter_clears_and_can_be_reused(void) {
    struct gate_e_guardian_drain_witness_process_identity controller = make_controller();
    struct gate_e_guardian_drain_witness_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_drain_witness_binding binding = make_binding();
    struct gate_e_guardian_drain_witness_binding uninitialized;
    struct gate_e_guardian_drain_witness_binding valid_binding;
    struct gate_e_guardian_drain_witness_report report = make_report();
    unsigned char worker_terminal_token[GATE_E_GUARDIAN_DRAIN_WITNESS_TOKEN_BYTES];

    make_worker_terminal_token(worker_terminal_token);
    controller.pid = UINT32_C(2);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING);
    controller = make_controller();
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_OK
    );
    assert_reason(&binding, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_OK);

    controller.uid = UINT32_C(1);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    controller = make_controller();
    controller.gid = UINT32_C(1);
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    controller = make_controller();
    controller.starttime_ticks = 0U;
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    controller = make_controller();
    memset(controller.pidfd_token, 0, sizeof(controller.pidfd_token));
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    controller = make_controller();
    memset(worker_terminal_token, 0, sizeof(worker_terminal_token));
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    make_worker_terminal_token(worker_terminal_token);
    cgroup = make_cgroup_claim();
    cgroup.non_delegated = false;
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    cgroup.identity.st_dev = 0U;
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    cgroup.identity.st_ino = 0U;
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    memset(cgroup.identity.held_fd_token, 0, sizeof(cgroup.identity.held_fd_token));
    gate_e_guardian_drain_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &binding,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    memset(&uninitialized, 0, sizeof(uninitialized));
    controller = make_controller();
    cgroup = make_cgroup_claim();
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &uninitialized,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT
    );
    assert_reason(&uninitialized, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            NULL,
            &controller,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT
    );
    assert_reason(NULL, &report, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING);
    valid_binding = make_binding();
    assert_reason(&valid_binding, NULL, GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_ARGUMENT);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &valid_binding,
            NULL,
            worker_terminal_token,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_drain_witness_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &valid_binding,
            &controller,
            NULL,
            &cgroup
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_drain_witness_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_drain_witness_binding_v1_set(
            &valid_binding,
            &controller,
            worker_terminal_token,
            NULL
        ) == GATE_E_GUARDIAN_DRAIN_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
}

int main(void) {
    test_valid_report_is_repeatable_and_nonmutating();
    test_fixed_width_boundary_values_are_accepted();
    test_each_claim_mismatch_fails_closed();
    test_invalid_reports_and_precedence_fail_closed();
    test_binding_setter_clears_and_can_be_reused();
    return 0;
}
