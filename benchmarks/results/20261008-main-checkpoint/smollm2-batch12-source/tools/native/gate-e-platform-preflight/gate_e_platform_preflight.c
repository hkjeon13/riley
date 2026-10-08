/*
 * Linux-only, no-action platform observation for the future RC3 Gate E
 * guardian.  This file intentionally lives outside the Cargo workspace: it
 * is a review aid, not a Riley runtime, release artifact, or privileged
 * launcher.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/magic.h>
#include <linux/openat2.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

enum {
    GATE_E_PLATFORM_BUFFER_BYTES = 512,
};

enum preflight_reason {
    PREFLIGHT_OK,
    PREFLIGHT_NOT_ROOT,
    PREFLIGHT_OPENAT2_UNAVAILABLE,
    PREFLIGHT_REQUIRED_PATH_UNREADABLE,
    PREFLIGHT_INITIAL_NAMESPACE_UNREADABLE,
    PREFLIGHT_INITIAL_USER_NAMESPACE_MISMATCH,
    PREFLIGHT_INITIAL_MOUNT_NAMESPACE_MISMATCH,
    PREFLIGHT_INITIAL_CGROUP_NAMESPACE_MISMATCH,
    PREFLIGHT_UID_MAP_MISMATCH,
    PREFLIGHT_GID_MAP_MISMATCH,
    PREFLIGHT_PID_ONE_NOT_SYSTEMD,
    PREFLIGHT_CGROUP2_UNAVAILABLE,
    PREFLIGHT_CGROUP_ROOT_UNSAFE,
};

struct platform_snapshot {
    bool root_identity;
    bool openat2_available;
    bool initial_user_namespace;
    bool initial_mount_namespace;
    bool initial_cgroup_namespace;
    bool full_initial_uid_map;
    bool full_initial_gid_map;
    bool pid_one_systemd;
    bool cgroup_v2;
    bool cgroup_root_safe;
};

typedef int (*openat2_invoker)(int, const char *, const struct open_how *);

static int linux_openat2(int directory_fd, const char *path, const struct open_how *how) {
#ifdef SYS_openat2
    return (int)syscall(SYS_openat2, directory_fd, path, how, sizeof(*how));
#else
    (void)directory_fd;
    (void)path;
    (void)how;
    errno = ENOSYS;
    return -1;
#endif
}

static openat2_invoker invoke_openat2 = linux_openat2;

#ifndef GATE_E_PLATFORM_PREFLIGHT_LIBRARY
static const char *reason_code(const enum preflight_reason reason) {
    switch (reason) {
    case PREFLIGHT_OK:
        return "checked";
    case PREFLIGHT_NOT_ROOT:
        return "effective-uid-gid-not-root";
    case PREFLIGHT_OPENAT2_UNAVAILABLE:
        return "openat2-abi-unavailable";
    case PREFLIGHT_REQUIRED_PATH_UNREADABLE:
        return "fixed-platform-path-unreadable";
    case PREFLIGHT_INITIAL_NAMESPACE_UNREADABLE:
        return "initial-namespace-unreadable";
    case PREFLIGHT_INITIAL_USER_NAMESPACE_MISMATCH:
        return "initial-user-namespace-mismatch";
    case PREFLIGHT_INITIAL_MOUNT_NAMESPACE_MISMATCH:
        return "initial-mount-namespace-mismatch";
    case PREFLIGHT_INITIAL_CGROUP_NAMESPACE_MISMATCH:
        return "initial-cgroup-namespace-mismatch";
    case PREFLIGHT_UID_MAP_MISMATCH:
        return "full-initial-uid-map-required";
    case PREFLIGHT_GID_MAP_MISMATCH:
        return "full-initial-gid-map-required";
    case PREFLIGHT_PID_ONE_NOT_SYSTEMD:
        return "pid-one-is-not-systemd";
    case PREFLIGHT_CGROUP2_UNAVAILABLE:
        return "cgroup-v2-unavailable";
    case PREFLIGHT_CGROUP_ROOT_UNSAFE:
        return "cgroup-root-owner-or-mode-unsafe";
    }
    return "unknown-preflight-reason";
}
#endif

static bool valid_relative_path(const char *path) {
    const char *component = path;

    if (path == NULL || path[0] == '\0' || path[0] == '/') {
        return false;
    }
    for (const char *cursor = path;; ++cursor) {
        if (*cursor != '/' && *cursor != '\0') {
            continue;
        }
        const size_t length = (size_t)(cursor - component);
        if (length == 0 || (length == 1 && component[0] == '.') ||
            (length == 2 && component[0] == '.' && component[1] == '.')) {
            return false;
        }
        if (*cursor == '\0') {
            return true;
        }
        component = cursor + 1;
    }
}

static int open_readonly_beneath(const int directory_fd, const char *path) {
    const struct open_how how = {
        .flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW,
        .mode = 0,
        .resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS,
    };

    if (!valid_relative_path(path)) {
        errno = EXDEV;
        return -1;
    }
    return invoke_openat2(directory_fd, path, &how);
}

#ifndef GATE_E_PLATFORM_PREFLIGHT_LIBRARY
static enum preflight_reason read_relative_file(
    const int root_fd,
    const char *path,
    char *buffer,
    const size_t capacity,
    size_t *const length
) {
    int fd = -1;
    size_t used = 0;

    if (capacity < 2) {
        return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
    }
    fd = open_readonly_beneath(root_fd, path);
    if (fd < 0) {
        return (errno == ENOSYS || errno == EINVAL) ? PREFLIGHT_OPENAT2_UNAVAILABLE
                                                     : PREFLIGHT_REQUIRED_PATH_UNREADABLE;
    }
    for (;;) {
        const ssize_t count = read(fd, buffer + used, capacity - used - 1);
        if (count < 0) {
            (void)close(fd);
            return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        }
        if (count == 0) {
            break;
        }
        used += (size_t)count;
        if (used == capacity - 1) {
            (void)close(fd);
            return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        }
    }
    if (close(fd) != 0) {
        return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
    }
    buffer[used] = '\0';
    *length = used;
    return PREFLIGHT_OK;
}
#endif

static bool is_full_initial_map(const char *value) {
    const char *cursor = value;
    unsigned long long fields[3] = {0, 0, 0};

    for (size_t index = 0; index < 3; ++index) {
        char *end = NULL;
        errno = 0;
        while (isspace((unsigned char)*cursor) != 0) {
            ++cursor;
        }
        if (*cursor == '\0' || isdigit((unsigned char)*cursor) == 0) {
            return false;
        }
        fields[index] = strtoull(cursor, &end, 10);
        if (errno != 0 || end == cursor) {
            return false;
        }
        cursor = end;
    }
    while (isspace((unsigned char)*cursor) != 0) {
        ++cursor;
    }
    return *cursor == '\0' && fields[0] == 0 && fields[1] == 0 &&
           fields[2] == UINT32_MAX;
}

#ifndef GATE_E_PLATFORM_PREFLIGHT_LIBRARY
static bool format_pid_path(
    char *const destination,
    const size_t capacity,
    const char *const pattern,
    const pid_t pid
) {
    const int rendered = snprintf(destination, capacity, pattern, (long)pid);

    return rendered >= 0 && (size_t)rendered < capacity;
}

static enum preflight_reason namespace_matches(
    const char *self_path,
    const char *initial_path,
    bool *const matches
) {
    struct stat self_metadata;
    struct stat initial_metadata;

    if (stat(self_path, &self_metadata) != 0 || stat(initial_path, &initial_metadata) != 0) {
        return PREFLIGHT_INITIAL_NAMESPACE_UNREADABLE;
    }
    *matches = self_metadata.st_dev == initial_metadata.st_dev &&
               self_metadata.st_ino == initial_metadata.st_ino;
    return PREFLIGHT_OK;
}

static enum preflight_reason observe_live_platform(struct platform_snapshot *const snapshot) {
    char buffer[GATE_E_PLATFORM_BUFFER_BYTES] = {0};
    char relative_path[96] = {0};
    char self_namespace[96] = {0};
    const pid_t pid = getpid();
    int root_fd = -1;
    size_t length = 0;
    enum preflight_reason reason = PREFLIGHT_OK;

    memset(snapshot, 0, sizeof(*snapshot));
    snapshot->root_identity = getuid() == 0 && geteuid() == 0 && getgid() == 0 && getegid() == 0;
    if (!snapshot->root_identity) {
        return PREFLIGHT_NOT_ROOT;
    }
    root_fd = open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (root_fd < 0) {
        return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
    }
    if (!format_pid_path(relative_path, sizeof(relative_path), "proc/%ld/uid_map", pid)) {
        reason = PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        goto finish;
    }
    reason = read_relative_file(root_fd, relative_path, buffer, sizeof(buffer), &length);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    snapshot->full_initial_uid_map = is_full_initial_map(buffer);
    if (!format_pid_path(relative_path, sizeof(relative_path), "proc/%ld/gid_map", pid)) {
        reason = PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        goto finish;
    }
    reason = read_relative_file(root_fd, relative_path, buffer, sizeof(buffer), &length);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    snapshot->full_initial_gid_map = is_full_initial_map(buffer);

    if (!format_pid_path(self_namespace, sizeof(self_namespace), "/proc/%ld/ns/user", pid)) {
        reason = PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        goto finish;
    }
    reason = namespace_matches(self_namespace, "/proc/1/ns/user", &snapshot->initial_user_namespace);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    if (!format_pid_path(self_namespace, sizeof(self_namespace), "/proc/%ld/ns/mnt", pid)) {
        reason = PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        goto finish;
    }
    reason = namespace_matches(self_namespace, "/proc/1/ns/mnt", &snapshot->initial_mount_namespace);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    if (!format_pid_path(self_namespace, sizeof(self_namespace), "/proc/%ld/ns/cgroup", pid)) {
        reason = PREFLIGHT_REQUIRED_PATH_UNREADABLE;
        goto finish;
    }
    reason = namespace_matches(self_namespace, "/proc/1/ns/cgroup", &snapshot->initial_cgroup_namespace);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }

    reason = read_relative_file(root_fd, "proc/1/comm", buffer, sizeof(buffer), &length);
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    snapshot->pid_one_systemd = strcmp(buffer, "systemd\n") == 0;
    reason = read_relative_file(
        root_fd, "sys/fs/cgroup/cgroup.controllers", buffer, sizeof(buffer), &length
    );
    if (reason != PREFLIGHT_OK) {
        goto finish;
    }
    snapshot->cgroup_v2 = length != 0;
    {
        struct stat cgroup_metadata;
        struct statfs cgroup_filesystem;
        if (stat("/sys/fs/cgroup", &cgroup_metadata) != 0 ||
            statfs("/sys/fs/cgroup", &cgroup_filesystem) != 0) {
            reason = PREFLIGHT_CGROUP2_UNAVAILABLE;
            goto finish;
        }
        snapshot->cgroup_v2 = snapshot->cgroup_v2 &&
                               (unsigned long)cgroup_filesystem.f_type ==
                                   (unsigned long)CGROUP2_SUPER_MAGIC;
        snapshot->cgroup_root_safe = S_ISDIR(cgroup_metadata.st_mode) &&
                                      cgroup_metadata.st_uid == 0 &&
                                      (cgroup_metadata.st_mode & 0022U) == 0;
    }
    snapshot->openat2_available = true;

finish:
    if (close(root_fd) != 0 && reason == PREFLIGHT_OK) {
        return PREFLIGHT_REQUIRED_PATH_UNREADABLE;
    }
    return reason;
}
#endif

static enum preflight_reason evaluate_snapshot(const struct platform_snapshot *const snapshot) {
    if (!snapshot->root_identity) {
        return PREFLIGHT_NOT_ROOT;
    }
    if (!snapshot->openat2_available) {
        return PREFLIGHT_OPENAT2_UNAVAILABLE;
    }
    if (!snapshot->initial_user_namespace) {
        return PREFLIGHT_INITIAL_USER_NAMESPACE_MISMATCH;
    }
    if (!snapshot->initial_mount_namespace) {
        return PREFLIGHT_INITIAL_MOUNT_NAMESPACE_MISMATCH;
    }
    if (!snapshot->initial_cgroup_namespace) {
        return PREFLIGHT_INITIAL_CGROUP_NAMESPACE_MISMATCH;
    }
    if (!snapshot->full_initial_uid_map) {
        return PREFLIGHT_UID_MAP_MISMATCH;
    }
    if (!snapshot->full_initial_gid_map) {
        return PREFLIGHT_GID_MAP_MISMATCH;
    }
    if (!snapshot->pid_one_systemd) {
        return PREFLIGHT_PID_ONE_NOT_SYSTEMD;
    }
    if (!snapshot->cgroup_v2) {
        return PREFLIGHT_CGROUP2_UNAVAILABLE;
    }
    if (!snapshot->cgroup_root_safe) {
        return PREFLIGHT_CGROUP_ROOT_UNSAFE;
    }
    return PREFLIGHT_OK;
}

#ifndef GATE_E_PLATFORM_PREFLIGHT_LIBRARY
static bool print_report(const enum preflight_reason reason) {
    const char *const status = reason == PREFLIGHT_OK ? "checked" : "not-established";
    const char *const detail = reason == PREFLIGHT_OK ? "null" : reason_code(reason);
    int written = 0;

    if (reason == PREFLIGHT_OK) {
        written = printf(
            "{\"schema_version\":\"riley.rc3-gate-e-native-platform-preflight.v1\","
            "\"status\":\"%s\",\"scope\":\"platform-observation-only\","
            "\"authority\":\"not-authoritative\",\"installation\":\"not-installed\","
            "\"execution_authority\":\"not-established\","
            "\"actual_gate_e_producer\":\"not-established\","
            "\"qualification_status\":\"not-run\",\"reason_code\":%s}\n",
            status,
            detail
        );
    } else {
        written = printf(
            "{\"schema_version\":\"riley.rc3-gate-e-native-platform-preflight.v1\","
            "\"status\":\"%s\",\"scope\":\"platform-observation-only\","
            "\"authority\":\"not-authoritative\",\"installation\":\"not-installed\","
            "\"execution_authority\":\"not-established\","
            "\"actual_gate_e_producer\":\"not-established\","
            "\"qualification_status\":\"not-run\",\"reason_code\":\"%s\"}\n",
            status,
            detail
        );
    }
    return written >= 0 && fflush(stdout) == 0;
}
#endif

static bool accepted_cli(const int argc, const char *const argv[]) {
    return argc == 2 && strcmp(argv[1], "--observe-linux-platform-v1") == 0;
}

#ifndef GATE_E_PLATFORM_PREFLIGHT_LIBRARY
int main(const int argc, char *const argv[]) {
    struct platform_snapshot snapshot;
    enum preflight_reason reason;

    if (!accepted_cli(argc, (const char *const *)argv)) {
        (void)fprintf(stderr, "usage: %s --observe-linux-platform-v1\n", argv[0]);
        return 64;
    }
    reason = observe_live_platform(&snapshot);
    if (reason == PREFLIGHT_OK) {
        reason = evaluate_snapshot(&snapshot);
    }
    if (!print_report(reason)) {
        (void)fprintf(stderr, "unable to emit platform preflight report\n");
        return 2;
    }
    return reason == PREFLIGHT_OK ? 0 : 2;
}
#endif
