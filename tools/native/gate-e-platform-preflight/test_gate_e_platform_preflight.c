#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GATE_E_PLATFORM_PREFLIGHT_LIBRARY
#include "gate_e_platform_preflight.c"

#include <assert.h>
#include <limits.h>
#include <sys/stat.h>

static int forced_enosys_calls = 0;

static int forced_enosys_openat2(
    const int directory_fd,
    const char *const path,
    const struct open_how *const how
) {
    (void)directory_fd;
    (void)path;
    (void)how;
    ++forced_enosys_calls;
    errno = ENOSYS;
    return -1;
}

static void write_exact_file(const char *const path, const char *const contents) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    const size_t expected = strlen(contents);
    ssize_t written = 0;

    assert(fd >= 0);
    written = write(fd, contents, expected);
    assert(written == (ssize_t)expected);
    assert(close(fd) == 0);
}

static void test_snapshot_contract(void) {
    struct platform_snapshot snapshot = {
        .root_identity = true,
        .openat2_available = true,
        .initial_user_namespace = true,
        .initial_mount_namespace = true,
        .initial_cgroup_namespace = true,
        .full_initial_uid_map = true,
        .full_initial_gid_map = true,
        .pid_one_systemd = true,
        .cgroup_v2 = true,
        .cgroup_root_safe = true,
    };

    assert(evaluate_snapshot(&snapshot) == PREFLIGHT_OK);
    snapshot.root_identity = false;
    assert(evaluate_snapshot(&snapshot) == PREFLIGHT_NOT_ROOT);
    snapshot.root_identity = true;
    snapshot.initial_user_namespace = false;
    assert(evaluate_snapshot(&snapshot) == PREFLIGHT_INITIAL_USER_NAMESPACE_MISMATCH);
    snapshot.initial_user_namespace = true;
    snapshot.openat2_available = false;
    assert(evaluate_snapshot(&snapshot) == PREFLIGHT_OPENAT2_UNAVAILABLE);
}

static void test_full_initial_map_contract(void) {
    assert(is_full_initial_map("         0          0 4294967295\n"));
    assert(!is_full_initial_map("0 1000 1\n"));
    assert(!is_full_initial_map("0 0 4294967294\n"));
    assert(!is_full_initial_map("0 0 4294967295 extra\n"));
}

static void test_exact_cli_contract(void) {
    const char *const accepted[] = {"tool", "--observe-linux-platform-v1"};
    const char *const unknown[] = {"tool", "--anchor=/tmp/unsafe"};
    const char *const missing[] = {"tool"};

    assert(accepted_cli(2, accepted));
    assert(!accepted_cli(2, unknown));
    assert(!accepted_cli(1, missing));
}

static void test_openat2_does_not_fallback(void) {
    const openat2_invoker saved = invoke_openat2;
    const int root_fd = open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);

    assert(root_fd >= 0);
    forced_enosys_calls = 0;
    invoke_openat2 = forced_enosys_openat2;
    assert(open_readonly_beneath(root_fd, "proc/1/comm") == -1);
    assert(errno == ENOSYS);
    assert(forced_enosys_calls == 1);
    invoke_openat2 = saved;
    assert(close(root_fd) == 0);
}

static void test_no_follow_and_held_fd_fixture(void) {
    char template[] = "/tmp/riley-gate-e-platform-preflight.XXXXXX";
    char safe_directory[PATH_MAX] = {0};
    char original_path[PATH_MAX] = {0};
    char replacement_path[PATH_MAX] = {0};
    char final_link_path[PATH_MAX] = {0};
    char ancestor_link_path[PATH_MAX] = {0};
    char observed[8] = {0};
    const char *const root = mkdtemp(template);
    int root_fd = -1;
    int held_fd = -1;

    assert(root != NULL);
    root_fd = open(root, O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    assert(root_fd >= 0);
    assert(snprintf(safe_directory, sizeof(safe_directory), "%s/safe", root) > 0);
    assert(mkdir(safe_directory, 0700) == 0);
    assert(snprintf(original_path, sizeof(original_path), "%s/safe/original", root) > 0);
    assert(snprintf(replacement_path, sizeof(replacement_path), "%s/safe/replacement", root) > 0);
    assert(snprintf(final_link_path, sizeof(final_link_path), "%s/final-link", root) > 0);
    assert(snprintf(ancestor_link_path, sizeof(ancestor_link_path), "%s/ancestor-link", root) > 0);
    write_exact_file(original_path, "old");
    write_exact_file(replacement_path, "new");
    assert(symlink("safe/original", final_link_path) == 0);
    assert(symlink("safe", ancestor_link_path) == 0);
    assert(open_readonly_beneath(root_fd, "final-link") == -1);
    assert(open_readonly_beneath(root_fd, "ancestor-link/original") == -1);
    assert(open_readonly_beneath(root_fd, "../etc/passwd") == -1);

    held_fd = open_readonly_beneath(root_fd, "safe/original");
    assert(held_fd >= 0);
    assert(rename(replacement_path, original_path) == 0);
    assert(read(held_fd, observed, sizeof(observed)) == 3);
    assert(memcmp(observed, "old", 3) == 0);
    assert(close(held_fd) == 0);
    assert(close(root_fd) == 0);
    assert(unlink(final_link_path) == 0);
    assert(unlink(ancestor_link_path) == 0);
    assert(unlink(original_path) == 0);
    assert(rmdir(safe_directory) == 0);
    assert(rmdir(root) == 0);
}

int main(void) {
    test_snapshot_contract();
    test_full_initial_map_contract();
    test_exact_cli_contract();
    test_openat2_does_not_fallback();
    test_no_follow_and_held_fd_fixture();
    (void)puts("gate-e platform preflight tests passed");
    return 0;
}
