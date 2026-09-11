#include "gate_e_guardian_controller_restart_witness_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void fill_token(
    unsigned char token[GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES],
    const unsigned char first_byte
) {
    memset(token, 0, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES);
    token[0] = first_byte;
}

static struct gate_e_guardian_controller_restart_witness_process_identity make_identity(
    const uint32_t pid,
    const uint64_t starttime_ticks,
    const uint32_t uid,
    const uint32_t gid,
    const unsigned char token_first_byte
) {
    struct gate_e_guardian_controller_restart_witness_process_identity identity = {0};

    identity.pid = pid;
    identity.starttime_ticks = starttime_ticks;
    identity.uid = uid;
    identity.gid = gid;
    fill_token(identity.pidfd_token, token_first_byte);
    return identity;
}

static struct gate_e_guardian_controller_restart_witness_process_identity make_guardian(void) {
    return make_identity(UINT32_C(101), UINT64_C(10100), 0U, 0U, (unsigned char)0x51U);
}

static struct gate_e_guardian_controller_restart_witness_process_identity make_warden(void) {
    return make_identity(UINT32_C(102), UINT64_C(10200), 0U, 0U, (unsigned char)0x52U);
}

static struct gate_e_guardian_controller_restart_witness_process_identity make_worker(void) {
    return make_identity(
        UINT32_C(201),
        UINT64_C(20100),
        UINT32_C(65532),
        UINT32_C(65532),
        (unsigned char)0x53U
    );
}

static struct gate_e_guardian_controller_restart_witness_process_identity make_new_controller(void) {
    return make_identity(UINT32_C(1), UINT64_C(30100), 0U, 0U, (unsigned char)0x54U);
}

static struct gate_e_guardian_controller_restart_witness_cgroup_claim make_cgroup_claim(void) {
    struct gate_e_guardian_controller_restart_witness_cgroup_claim claim = {0};

    claim.identity.st_dev = UINT64_C(1001);
    claim.identity.st_ino = UINT64_C(1002);
    claim.non_delegated = true;
    fill_token(claim.identity.held_fd_token, (unsigned char)0x55U);
    return claim;
}

static struct gate_e_guardian_controller_restart_witness_report make_report(void) {
    struct gate_e_guardian_controller_restart_witness_report report = {0};
    const struct gate_e_guardian_controller_restart_witness_process_identity worker = make_worker();

    report.new_pid1_controller = make_new_controller();
    report.cgroup = make_cgroup_claim();
    report.population = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_EMPTY;
    report.terminal_pidfd_token_count = UINT32_C(1);
    memcpy(report.terminal_pidfd_token, worker.pidfd_token, sizeof(report.terminal_pidfd_token));
    return report;
}

static struct gate_e_guardian_controller_restart_witness_binding make_binding(void) {
    struct gate_e_guardian_controller_restart_witness_binding binding;
    const struct gate_e_guardian_controller_restart_witness_process_identity guardian = make_guardian();
    const struct gate_e_guardian_controller_restart_witness_process_identity warden = make_warden();
    const struct gate_e_guardian_controller_restart_witness_process_identity worker = make_worker();
    const struct gate_e_guardian_controller_restart_witness_cgroup_claim cgroup = make_cgroup_claim();

    gate_e_guardian_controller_restart_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
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
    const struct gate_e_guardian_controller_restart_witness_binding *const binding
) {
    assert_zero_bytes(&binding->registered_guardian, sizeof(binding->registered_guardian));
    assert_zero_bytes(&binding->registered_warden, sizeof(binding->registered_warden));
    assert_zero_bytes(&binding->registered_worker, sizeof(binding->registered_worker));
    assert_zero_bytes(&binding->held_cgroup, sizeof(binding->held_cgroup));
}

static void assert_process_equal(
    const struct gate_e_guardian_controller_restart_witness_process_identity *const left,
    const struct gate_e_guardian_controller_restart_witness_process_identity *const right
) {
    assert(left->pid == right->pid);
    assert(left->starttime_ticks == right->starttime_ticks);
    assert(left->uid == right->uid);
    assert(left->gid == right->gid);
    assert(memcmp(left->pidfd_token, right->pidfd_token, sizeof(left->pidfd_token)) == 0);
}

static void assert_cgroup_equal(
    const struct gate_e_guardian_controller_restart_witness_cgroup_claim *const left,
    const struct gate_e_guardian_controller_restart_witness_cgroup_claim *const right
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
    const struct gate_e_guardian_controller_restart_witness_binding *const binding,
    const struct gate_e_guardian_controller_restart_witness_report *const report,
    const enum gate_e_guardian_controller_restart_witness_reason expected
) {
    assert(gate_e_match_guardian_controller_restart_witness_v1(binding, report) == expected);
}

static void test_valid_report_is_repeatable_and_nonmutating(void) {
    const struct gate_e_guardian_controller_restart_witness_binding binding = make_binding();
    const struct gate_e_guardian_controller_restart_witness_binding binding_before = binding;
    const struct gate_e_guardian_controller_restart_witness_report report = make_report();
    const struct gate_e_guardian_controller_restart_witness_report report_before = report;

    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK);
    assert_process_equal(&binding.registered_guardian, &binding_before.registered_guardian);
    assert_process_equal(&binding.registered_warden, &binding_before.registered_warden);
    assert_process_equal(&binding.registered_worker, &binding_before.registered_worker);
    assert(binding.held_cgroup.st_dev == binding_before.held_cgroup.st_dev);
    assert(binding.held_cgroup.st_ino == binding_before.held_cgroup.st_ino);
    assert(
        memcmp(
            binding.held_cgroup.held_fd_token,
            binding_before.held_cgroup.held_fd_token,
            sizeof(binding.held_cgroup.held_fd_token)
        ) == 0
    );
    assert_process_equal(&report.new_pid1_controller, &report_before.new_pid1_controller);
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

static void test_distinct_restarted_pid1_identity_is_accepted(void) {
    const struct gate_e_guardian_controller_restart_witness_binding binding = make_binding();
    struct gate_e_guardian_controller_restart_witness_report report = make_report();

    /* The contract compares against the registered guardian/warden/worker,
     * not a previous controller identity.  A fresh valid PID 1 identity is
     * therefore valid when it remains distinct from all three actors. */
    ++report.new_pid1_controller.starttime_ticks;
    report.new_pid1_controller.pidfd_token[
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK);
}

static void test_fixed_width_boundary_values_are_accepted(void) {
    struct gate_e_guardian_controller_restart_witness_process_identity guardian = make_guardian();
    struct gate_e_guardian_controller_restart_witness_process_identity warden = make_warden();
    struct gate_e_guardian_controller_restart_witness_process_identity worker = make_worker();
    struct gate_e_guardian_controller_restart_witness_process_identity controller = make_new_controller();
    struct gate_e_guardian_controller_restart_witness_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_controller_restart_witness_binding binding;
    struct gate_e_guardian_controller_restart_witness_report report;

    guardian.pid = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_MAX_PID;
    guardian.starttime_ticks = UINT64_MAX;
    warden.pid = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_MAX_PID - UINT32_C(1);
    warden.starttime_ticks = UINT64_MAX - UINT64_C(1);
    worker.pid = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_MAX_PID - UINT32_C(2);
    worker.starttime_ticks = UINT64_MAX - UINT64_C(2);
    worker.uid = UINT32_MAX;
    worker.gid = UINT32_MAX;
    controller.starttime_ticks = UINT64_MAX - UINT64_C(3);
    cgroup.identity.st_dev = UINT64_MAX;
    cgroup.identity.st_ino = UINT64_MAX;
    gate_e_guardian_controller_restart_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
    );
    report = make_report();
    report.new_pid1_controller = controller;
    report.cgroup = cgroup;
    memcpy(report.terminal_pidfd_token, worker.pidfd_token, sizeof(report.terminal_pidfd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK);
}

static void test_invalid_controller_and_distinctness_fail_closed(void) {
    const struct gate_e_guardian_controller_restart_witness_binding binding = make_binding();
    struct gate_e_guardian_controller_restart_witness_report report;
    struct gate_e_guardian_controller_restart_witness_binding special_binding;
    struct gate_e_guardian_controller_restart_witness_process_identity guardian;
    struct gate_e_guardian_controller_restart_witness_process_identity warden;
    struct gate_e_guardian_controller_restart_witness_process_identity worker;
    struct gate_e_guardian_controller_restart_witness_cgroup_claim cgroup;

    report = make_report();
    report.new_pid1_controller.pid = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.new_pid1_controller.pid = UINT32_C(2);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.new_pid1_controller.starttime_ticks = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    memset(report.new_pid1_controller.pidfd_token, 0, sizeof(report.new_pid1_controller.pidfd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.new_pid1_controller.uid = UINT32_C(1);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.new_pid1_controller.gid = UINT32_C(1);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);

    guardian = make_new_controller();
    warden = make_warden();
    worker = make_worker();
    cgroup = make_cgroup_claim();
    gate_e_guardian_controller_restart_witness_binding_v1_init(&special_binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &special_binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
    );
    report = make_report();
    report.new_pid1_controller = guardian;
    assert_reason(
        &special_binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CONTROLLER_IS_GUARDIAN
    );
    guardian = make_guardian();
    warden = make_new_controller();
    gate_e_guardian_controller_restart_witness_binding_v1_init(&special_binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &special_binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
    );
    report = make_report();
    report.new_pid1_controller = warden;
    assert_reason(
        &special_binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CONTROLLER_IS_WARDEN
    );
}

static void test_cgroup_population_and_terminal_token_fail_closed(void) {
    const struct gate_e_guardian_controller_restart_witness_binding binding = make_binding();
    struct gate_e_guardian_controller_restart_witness_report report;

    report = make_report();
    report.cgroup.non_delegated = false;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
    report = make_report();
    report.cgroup.identity.st_dev = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    report.cgroup.identity.st_ino = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    memset(report.cgroup.identity.held_fd_token, 0, sizeof(report.cgroup.identity.held_fd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    ++report.cgroup.identity.st_dev;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_ST_DEV_MISMATCH);
    report = make_report();
    ++report.cgroup.identity.st_ino;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_ST_INO_MISMATCH);
    report = make_report();
    report.cgroup.identity.held_fd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH
    );
    report = make_report();
    report.cgroup.identity.held_fd_token[
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_HELD_FD_TOKEN_MISMATCH
    );

    report = make_report();
    report.population = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_STILL_POPULATED);
    report = make_report();
    report.population = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_INVALID;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_POPULATION_CLAIM);
    report = make_report();
    report.population = (enum gate_e_guardian_controller_restart_witness_population_claim)99;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_POPULATION_CLAIM);
    report = make_report();
    report.terminal_pidfd_token_count = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_TERMINAL_TOKEN_COUNT);
    report = make_report();
    report.terminal_pidfd_token_count = UINT32_C(2);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_TERMINAL_TOKEN_COUNT);
    report = make_report();
    report.terminal_pidfd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    report.terminal_pidfd_token[
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_TERMINAL_PIDFD_TOKEN_MISMATCH
    );

    report = make_report();
    report.new_pid1_controller.pid = UINT32_C(2);
    report.cgroup.non_delegated = false;
    report.population = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_REPORTED_CONTROLLER);
    report = make_report();
    report.cgroup.non_delegated = false;
    report.cgroup.identity.st_dev = 0U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
    report = make_report();
    report.population = GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_POPULATION_PRESENT;
    report.terminal_pidfd_token_count = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_CGROUP_STILL_POPULATED);
}

static void test_binding_setter_clears_and_can_be_reused(void) {
    struct gate_e_guardian_controller_restart_witness_process_identity guardian = make_guardian();
    struct gate_e_guardian_controller_restart_witness_process_identity warden = make_warden();
    struct gate_e_guardian_controller_restart_witness_process_identity worker = make_worker();
    struct gate_e_guardian_controller_restart_witness_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_controller_restart_witness_binding binding = make_binding();
    struct gate_e_guardian_controller_restart_witness_binding uninitialized;
    struct gate_e_guardian_controller_restart_witness_binding valid_binding;
    struct gate_e_guardian_controller_restart_witness_report report = make_report();

    guardian.uid = UINT32_C(1);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING);
    guardian = make_guardian();
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK
    );
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_OK);

    worker.gid = 0U;
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    worker = make_worker();
    warden = guardian;
    gate_e_guardian_controller_restart_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    warden = make_warden();
    cgroup.non_delegated = false;
    gate_e_guardian_controller_restart_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    memset(cgroup.identity.held_fd_token, 0, sizeof(cgroup.identity.held_fd_token));
    gate_e_guardian_controller_restart_witness_binding_v1_init(&binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &binding,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    memset(&uninitialized, 0, sizeof(uninitialized));
    cgroup = make_cgroup_claim();
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &uninitialized,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_ARGUMENT
    );
    assert_reason(&uninitialized, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            NULL,
            &guardian,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_ARGUMENT
    );
    assert_reason(NULL, &report, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING);
    valid_binding = make_binding();
    assert_reason(&valid_binding, NULL, GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_ARGUMENT);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &valid_binding,
            NULL,
            &warden,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_controller_restart_witness_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &valid_binding,
            &guardian,
            NULL,
            &worker,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_controller_restart_witness_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &valid_binding,
            &guardian,
            &warden,
            NULL,
            &cgroup
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_controller_restart_witness_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_controller_restart_witness_binding_v1_set(
            &valid_binding,
            &guardian,
            &warden,
            &worker,
            NULL
        ) == GATE_E_GUARDIAN_CONTROLLER_RESTART_WITNESS_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
}

int main(void) {
    test_valid_report_is_repeatable_and_nonmutating();
    test_distinct_restarted_pid1_identity_is_accepted();
    test_fixed_width_boundary_values_are_accepted();
    test_invalid_controller_and_distinctness_fail_closed();
    test_cgroup_population_and_terminal_token_fail_closed();
    test_binding_setter_clears_and_can_be_reused();
    return 0;
}
