#include "gate_e_guardian_control_envelope_v1.c"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static void fill_token(
    unsigned char token[GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES],
    const unsigned char first_byte
) {
    memset(token, 0, GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES);
    token[0] = first_byte;
}

static struct gate_e_guardian_process_identity make_worker(void) {
    struct gate_e_guardian_process_identity worker = {0};

    worker.pid = UINT32_C(201);
    worker.starttime_ticks = UINT64_C(20100);
    worker.uid = UINT32_C(65532);
    worker.gid = UINT32_C(65532);
    fill_token(worker.pidfd_token, (unsigned char)0x11U);
    return worker;
}

static struct gate_e_guardian_cgroup_claim make_cgroup_claim(void) {
    struct gate_e_guardian_cgroup_claim claim = {0};

    claim.identity.st_dev = UINT64_C(1001);
    claim.identity.st_ino = UINT64_C(1002);
    claim.non_delegated = true;
    fill_token(claim.identity.held_fd_token, (unsigned char)0x22U);
    return claim;
}

static struct gate_e_guardian_control_envelope_report make_report(void) {
    struct gate_e_guardian_control_envelope_report report = {0};

    report.credentials = make_worker();
    report.cgroup = make_cgroup_claim();
    report.ancillary = GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_EMPTY;
    return report;
}

static struct gate_e_guardian_control_envelope_binding make_binding(void) {
    struct gate_e_guardian_control_envelope_binding binding;
    const struct gate_e_guardian_process_identity worker = make_worker();
    const struct gate_e_guardian_cgroup_claim cgroup = make_cgroup_claim();

    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK
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
    const struct gate_e_guardian_control_envelope_binding *const binding
) {
    assert_zero_bytes(&binding->registered_worker, sizeof(binding->registered_worker));
    assert_zero_bytes(&binding->held_cgroup, sizeof(binding->held_cgroup));
}

static void assert_process_equal(
    const struct gate_e_guardian_process_identity *const left,
    const struct gate_e_guardian_process_identity *const right
) {
    assert(left->pid == right->pid);
    assert(left->starttime_ticks == right->starttime_ticks);
    assert(left->uid == right->uid);
    assert(left->gid == right->gid);
    assert(memcmp(left->pidfd_token, right->pidfd_token, sizeof(left->pidfd_token)) == 0);
}

static void assert_cgroup_equal(
    const struct gate_e_guardian_cgroup_claim *const left,
    const struct gate_e_guardian_cgroup_claim *const right
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
    const struct gate_e_guardian_control_envelope_binding *const binding,
    const struct gate_e_guardian_control_envelope_report *const report,
    const enum gate_e_guardian_control_envelope_reason expected
) {
    assert(gate_e_match_guardian_control_envelope_v1(binding, report) == expected);
}

static void test_valid_report_is_repeatable_and_nonmutating(void) {
    const struct gate_e_guardian_control_envelope_binding binding = make_binding();
    const struct gate_e_guardian_control_envelope_binding binding_before = binding;
    const struct gate_e_guardian_control_envelope_report report = make_report();
    const struct gate_e_guardian_control_envelope_report report_before = report;

    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK);
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK);
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
    assert_process_equal(&report.credentials, &report_before.credentials);
    assert_cgroup_equal(&report.cgroup, &report_before.cgroup);
    assert(report.ancillary == report_before.ancillary);
}

static void test_fixed_width_boundary_values_are_accepted(void) {
    struct gate_e_guardian_process_identity worker = make_worker();
    struct gate_e_guardian_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_control_envelope_binding binding;
    struct gate_e_guardian_control_envelope_report report;

    worker.pid = GATE_E_GUARDIAN_CONTROL_ENVELOPE_MAX_PID;
    worker.starttime_ticks = UINT64_MAX;
    worker.uid = UINT32_MAX;
    worker.gid = UINT32_MAX;
    cgroup.identity.st_dev = UINT64_MAX;
    cgroup.identity.st_ino = UINT64_MAX;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK
    );
    report = make_report();
    report.credentials = worker;
    report.cgroup = cgroup;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK);
}

static void test_each_credential_and_cgroup_mismatch_fails_closed(void) {
    const struct gate_e_guardian_control_envelope_binding binding = make_binding();
    struct gate_e_guardian_control_envelope_report report;

    report = make_report();
    ++report.credentials.pid;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PID_MISMATCH
    );
    report = make_report();
    ++report.credentials.starttime_ticks;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_STARTTIME_TICKS_MISMATCH
    );
    report = make_report();
    report.credentials.pidfd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    report.credentials.pidfd_token[
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PIDFD_TOKEN_MISMATCH
    );
    report = make_report();
    ++report.credentials.uid;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_UID_MISMATCH
    );
    report = make_report();
    ++report.credentials.gid;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_GID_MISMATCH
    );
    report = make_report();
    report.cgroup.non_delegated = false;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
    report = make_report();
    ++report.cgroup.identity.st_dev;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_DEV_MISMATCH
    );
    report = make_report();
    ++report.cgroup.identity.st_ino;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_INO_MISMATCH
    );
    report = make_report();
    report.cgroup.identity.held_fd_token[0] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_HELD_FD_TOKEN_MISMATCH
    );
    report = make_report();
    report.cgroup.identity.held_fd_token[
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_TOKEN_BYTES - 1U
    ] ^= 0x01U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_HELD_FD_TOKEN_MISMATCH
    );
    report = make_report();
    report.ancillary = GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_PRESENT;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_FDS_PRESENT
    );
    report = make_report();
    report.ancillary = GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_INVALID;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ANCILLARY_CLAIM
    );
    report = make_report();
    report.ancillary = (enum gate_e_guardian_control_envelope_ancillary_claim)99;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ANCILLARY_CLAIM
    );
}

static void test_invalid_report_values_and_precedence_fail_closed(void) {
    const struct gate_e_guardian_control_envelope_binding binding = make_binding();
    struct gate_e_guardian_control_envelope_report report;

    report = make_report();
    report.credentials.pid = 0U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS
    );
    report = make_report();
    report.credentials.pid = GATE_E_GUARDIAN_CONTROL_ENVELOPE_MAX_PID + UINT32_C(1);
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS
    );
    report = make_report();
    report.credentials.starttime_ticks = 0U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS
    );
    report = make_report();
    memset(report.credentials.pidfd_token, 0, sizeof(report.credentials.pidfd_token));
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_REPORTED_CREDENTIALS
    );
    report = make_report();
    report.credentials.uid = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_UID_MISMATCH);
    report = make_report();
    report.credentials.gid = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_GID_MISMATCH);
    report = make_report();
    report.cgroup.identity.st_dev = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    report.cgroup.identity.st_ino = 0U;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_SUPPLIED_CGROUP);
    report = make_report();
    memset(report.cgroup.identity.held_fd_token, 0, sizeof(report.cgroup.identity.held_fd_token));
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_SUPPLIED_CGROUP);

    report = make_report();
    ++report.credentials.pid;
    report.cgroup.non_delegated = false;
    report.ancillary = GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_PRESENT;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PID_MISMATCH
    );
    report = make_report();
    ++report.cgroup.identity.st_dev;
    report.ancillary = GATE_E_GUARDIAN_CONTROL_ENVELOPE_ANCILLARY_PRESENT;
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_CGROUP_ST_DEV_MISMATCH);
    report = make_report();
    ++report.credentials.pid;
    ++report.credentials.gid;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_CREDENTIAL_PID_MISMATCH
    );
    report = make_report();
    report.cgroup.non_delegated = false;
    report.cgroup.identity.st_dev = 0U;
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_SUPPLIED_CGROUP_NOT_NON_DELEGATED
    );
}

static void test_binding_setter_clears_and_can_be_reused(void) {
    struct gate_e_guardian_process_identity worker = make_worker();
    struct gate_e_guardian_cgroup_claim cgroup = make_cgroup_claim();
    struct gate_e_guardian_control_envelope_binding binding = make_binding();
    struct gate_e_guardian_control_envelope_binding uninitialized;
    struct gate_e_guardian_control_envelope_binding valid_binding;
    struct gate_e_guardian_control_envelope_report report = make_report();

    worker.uid = 0U;
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    assert_reason(
        &binding,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    worker = make_worker();
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK
    );
    assert_reason(&binding, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_OK);

    worker.pid = 0U;
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    worker = make_worker();
    worker.pid = GATE_E_GUARDIAN_CONTROL_ENVELOPE_MAX_PID + UINT32_C(1);
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    worker = make_worker();
    worker.starttime_ticks = 0U;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    worker = make_worker();
    memset(worker.pidfd_token, 0, sizeof(worker.pidfd_token));
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    worker = make_worker();
    worker.gid = 0U;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    worker = make_worker();
    cgroup = make_cgroup_claim();
    cgroup.non_delegated = false;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    cgroup.identity.st_dev = 0U;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    cgroup.identity.st_ino = 0U;
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);
    cgroup = make_cgroup_claim();
    memset(cgroup.identity.held_fd_token, 0, sizeof(cgroup.identity.held_fd_token));
    gate_e_guardian_control_envelope_binding_v1_init(&binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&binding, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&binding);

    memset(&uninitialized, 0, sizeof(uninitialized));
    worker = make_worker();
    cgroup = make_cgroup_claim();
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&uninitialized, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT
    );
    assert_reason(
        &uninitialized,
        &report,
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(NULL, &worker, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT
    );
    assert_reason(NULL, &report, GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING);
    valid_binding = make_binding();
    assert_reason(&valid_binding, NULL, GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_ARGUMENT);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&valid_binding, NULL, &cgroup) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
    gate_e_guardian_control_envelope_binding_v1_init(&valid_binding);
    assert(
        gate_e_guardian_control_envelope_binding_v1_set(&valid_binding, &worker, NULL) ==
        GATE_E_GUARDIAN_CONTROL_ENVELOPE_INVALID_EXPECTED_BINDING
    );
    assert_cleared_binding(&valid_binding);
}

int main(void) {
    test_valid_report_is_repeatable_and_nonmutating();
    test_fixed_width_boundary_values_are_accepted();
    test_each_credential_and_cgroup_mismatch_fails_closed();
    test_invalid_report_values_and_precedence_fail_closed();
    test_binding_setter_clears_and_can_be_reused();
    return 0;
}
