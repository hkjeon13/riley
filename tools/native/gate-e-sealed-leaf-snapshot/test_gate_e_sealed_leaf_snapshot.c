#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "gate_e_sealed_leaf_snapshot.c"

#include <assert.h>
#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>

struct fixture {
    char root[PATH_MAX];
    char leaf[PATH_MAX];
    int descriptor;
    const char *contents;
    size_t byte_length;
    unsigned char sha256[GATE_E_SEALED_LEAF_SHA256_BYTES];
};

static int forced_memfd_calls = 0;
static int raw_source_descriptor_to_reuse = -1;
static int reused_pipe_descriptors[2] = {-1, -1};
static bool reuse_raw_source_descriptor_on_next_pread = false;
static int first_source_pread_descriptor = -1;

static int forced_memfd_enosys(const char *const name, const unsigned int flags) {
    (void)name;
    (void)flags;
    ++forced_memfd_calls;
    errno = ENOSYS;
    return -1;
}

static ssize_t forced_pread_eio(const int descriptor, void *const buffer, const size_t size, const off_t offset) {
    (void)descriptor;
    (void)buffer;
    (void)size;
    (void)offset;
    errno = EIO;
    return -1;
}

static ssize_t close_and_reuse_raw_source_on_first_pread(
    const int descriptor,
    void *const buffer,
    const size_t size,
    const off_t offset
) {
    if (reuse_raw_source_descriptor_on_next_pread) {
        reuse_raw_source_descriptor_on_next_pread = false;
        first_source_pread_descriptor = descriptor;
        assert(descriptor != raw_source_descriptor_to_reuse);
        assert(close(raw_source_descriptor_to_reuse) == 0);
        assert(pipe2(reused_pipe_descriptors, O_CLOEXEC) == 0);
        assert(reused_pipe_descriptors[0] == raw_source_descriptor_to_reuse);
    }
    return pread(descriptor, buffer, size, offset);
}

static ssize_t forced_pwrite_eio(
    const int descriptor,
    const void *const buffer,
    const size_t size,
    const off_t offset
) {
    (void)descriptor;
    (void)buffer;
    (void)size;
    (void)offset;
    errno = EIO;
    return -1;
}

static int forced_add_seals_eperm(const int descriptor, const int seals) {
    (void)descriptor;
    (void)seals;
    errno = EPERM;
    return -1;
}

static void write_exact_file(const char *const path, const char *const contents) {
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    const size_t length = strlen(contents);
    size_t written = 0;

    assert(descriptor >= 0);
    while (written < length) {
        const ssize_t count = write(descriptor, contents + written, length - written);
        assert(count > 0);
        written += (size_t)count;
    }
    assert(fsync(descriptor) == 0);
    assert(close(descriptor) == 0);
    assert(chmod(path, 0600) == 0);
}

static void sha256_text(const char *const text, unsigned char output[GATE_E_SEALED_LEAF_SHA256_BYTES]) {
    struct sha256_context context;

    sha256_init(&context);
    sha256_update(&context, (const unsigned char *)text, strlen(text));
    sha256_final(&context, output);
}

static void test_sha256_matches_standard_vectors(void) {
    static const unsigned char empty_digest[GATE_E_SEALED_LEAF_SHA256_BYTES] = {
        0xe3U, 0xb0U, 0xc4U, 0x42U, 0x98U, 0xfcU, 0x1cU, 0x14U,
        0x9aU, 0xfbU, 0xf4U, 0xc8U, 0x99U, 0x6fU, 0xb9U, 0x24U,
        0x27U, 0xaeU, 0x41U, 0xe4U, 0x64U, 0x9bU, 0x93U, 0x4cU,
        0xa4U, 0x95U, 0x99U, 0x1bU, 0x78U, 0x52U, 0xb8U, 0x55U,
    };
    static const unsigned char abc_digest[GATE_E_SEALED_LEAF_SHA256_BYTES] = {
        0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU,
        0x41U, 0x41U, 0x40U, 0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U,
        0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U, 0x7aU, 0x9cU,
        0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU,
    };
    unsigned char observed[GATE_E_SEALED_LEAF_SHA256_BYTES];

    sha256_text("", observed);
    assert(memcmp(observed, empty_digest, sizeof(observed)) == 0);
    sha256_text("abc", observed);
    assert(memcmp(observed, abc_digest, sizeof(observed)) == 0);
}

static struct fixture make_fixture(void) {
    static const char contents[] = "sealed leaf snapshot fixture\n";
    struct fixture fixture = {.descriptor = -1, .contents = contents, .byte_length = strlen(contents)};
    char template[] = "/tmp/riley-sealed-leaf-snapshot.XXXXXX";
    char *const root = mkdtemp(template);

    assert(root != NULL);
    assert(strlen(root) < sizeof(fixture.root));
    memcpy(fixture.root, root, strlen(root) + 1U);
    assert(chmod(fixture.root, 0700) == 0);
    assert(snprintf(fixture.leaf, sizeof(fixture.leaf), "%s/leaf.py", fixture.root) > 0);
    write_exact_file(fixture.leaf, fixture.contents);
    fixture.descriptor = open(fixture.leaf, O_RDONLY | O_CLOEXEC | O_NOATIME);
    assert(fixture.descriptor >= 0);
    sha256_text(fixture.contents, fixture.sha256);
    return fixture;
}

static void destroy_fixture(struct fixture *const fixture) {
    if (fixture->descriptor >= 0) {
        assert(close(fixture->descriptor) == 0);
        fixture->descriptor = -1;
    }
    (void)unlink(fixture->leaf);
    (void)rmdir(fixture->root);
}

static size_t open_descriptor_count(void) {
    DIR *const directory = opendir("/proc/self/fd");
    struct dirent *entry;
    size_t count = 0;

    assert(directory != NULL);
    while ((entry = readdir(directory)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            ++count;
        }
    }
    assert(closedir(directory) == 0);
    return count;
}

static void assert_empty_snapshot(const struct gate_e_sealed_leaf_snapshot *const snapshot) {
    static const unsigned char zero_digest[GATE_E_SEALED_LEAF_SHA256_BYTES] = {0};

    assert(snapshot->descriptor == -1);
    assert(snapshot->byte_length == 0U);
    assert(memcmp(snapshot->sha256, zero_digest, sizeof(zero_digest)) == 0);
}

static void test_success_seals_anonymous_copy_and_preserves_source_offset(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot;
    struct stat metadata;
    char observed[128] = {0};
    void *writable_mapping;
    int mutator;
    const size_t descriptors_before = open_descriptor_count();
    const size_t source_offset = 3;

    gate_e_sealed_leaf_snapshot_init(&snapshot);
    assert(lseek(fixture.descriptor, (off_t)source_offset, SEEK_SET) == (off_t)source_offset);
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_OK);
    assert(lseek(fixture.descriptor, 0, SEEK_CUR) == (off_t)source_offset);
    assert(snapshot.descriptor >= 0);
    assert(snapshot.descriptor >= 3);
    assert(open_descriptor_count() == descriptors_before + 1U);
    assert(snapshot.byte_length == fixture.byte_length);
    assert(memcmp(snapshot.sha256, fixture.sha256, sizeof(snapshot.sha256)) == 0);
    assert((fcntl(snapshot.descriptor, F_GETFD) & FD_CLOEXEC) != 0);
    assert(fcntl(snapshot.descriptor, F_GET_SEALS) == REQUIRED_MEMFD_SEALS);
    assert(fstat(snapshot.descriptor, &metadata) == 0);
    assert(S_ISREG(metadata.st_mode));
    assert(metadata.st_nlink == 0);
    assert((size_t)metadata.st_size == fixture.byte_length);
    assert((metadata.st_mode & 0111) == 0);
    errno = 0;
    assert(fchmod(snapshot.descriptor, 0700) == -1);
    assert(errno == EPERM);
    assert(pread(snapshot.descriptor, observed, fixture.byte_length, 0) == (ssize_t)fixture.byte_length);
    assert(memcmp(observed, fixture.contents, fixture.byte_length) == 0);
    errno = 0;
    assert(pwrite(snapshot.descriptor, "x", 1, 0) == -1);
    assert(errno == EPERM);
    errno = 0;
    writable_mapping = mmap(
        NULL, fixture.byte_length, PROT_READ | PROT_WRITE, MAP_SHARED, snapshot.descriptor, 0
    );
    assert(writable_mapping == MAP_FAILED);
    assert(errno == EPERM);
    errno = 0;
    assert(ftruncate(snapshot.descriptor, 0) == -1);
    assert(errno == EPERM);
    mutator = open(fixture.leaf, O_WRONLY | O_CLOEXEC | O_NOATIME);
    assert(mutator >= 0);
    assert(pwrite(mutator, "X", 1, 0) == 1);
    assert(fsync(mutator) == 0);
    assert(close(mutator) == 0);
    memset(observed, 0, sizeof(observed));
    assert(pread(snapshot.descriptor, observed, fixture.byte_length, 0) == (ssize_t)fixture.byte_length);
    assert(memcmp(observed, fixture.contents, fixture.byte_length) == 0);
    errno = 0;
    assert(fcntl(snapshot.descriptor, F_ADD_SEALS, F_SEAL_WRITE) == -1);
    assert(errno == EPERM);
    assert(gate_e_sealed_leaf_snapshot_close(&snapshot) == GATE_E_SEALED_LEAF_OK);
    assert_empty_snapshot(&snapshot);
    assert(open_descriptor_count() == descriptors_before);
    destroy_fixture(&fixture);
}

static void test_pins_source_before_the_caller_fd_number_is_reused(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot;
    const pread_invoker saved_pread = invoke_pread;
    char observed[128] = {0};

    gate_e_sealed_leaf_snapshot_init(&snapshot);
    raw_source_descriptor_to_reuse = fixture.descriptor;
    reused_pipe_descriptors[0] = -1;
    reused_pipe_descriptors[1] = -1;
    first_source_pread_descriptor = -1;
    reuse_raw_source_descriptor_on_next_pread = true;
    invoke_pread = close_and_reuse_raw_source_on_first_pread;
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_OK);
    assert(first_source_pread_descriptor >= 3);
    assert(first_source_pread_descriptor != raw_source_descriptor_to_reuse);
    fixture.descriptor = -1;
    assert(reused_pipe_descriptors[0] == raw_source_descriptor_to_reuse);
    assert(close(reused_pipe_descriptors[0]) == 0);
    assert(close(reused_pipe_descriptors[1]) == 0);
    reused_pipe_descriptors[0] = -1;
    reused_pipe_descriptors[1] = -1;
    assert(pread(snapshot.descriptor, observed, fixture.byte_length, 0) == (ssize_t)fixture.byte_length);
    assert(memcmp(observed, fixture.contents, fixture.byte_length) == 0);
    assert(gate_e_sealed_leaf_snapshot_close(&snapshot) == GATE_E_SEALED_LEAF_OK);
    invoke_pread = saved_pread;
    raw_source_descriptor_to_reuse = -1;
    destroy_fixture(&fixture);
}

static void test_alias_digest_and_initialization_contract(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot = {.descriptor = -1};

    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_INVALID_ARGUMENT);
    gate_e_sealed_leaf_snapshot_init(&snapshot);
    memcpy(snapshot.sha256, fixture.sha256, sizeof(snapshot.sha256));
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, snapshot.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_OK);
    assert(memcmp(snapshot.sha256, fixture.sha256, sizeof(snapshot.sha256)) == 0);
    assert(gate_e_sealed_leaf_snapshot_close(&snapshot) == GATE_E_SEALED_LEAF_OK);
    destroy_fixture(&fixture);
}

static void test_invalid_arguments_clear_a_valid_output_object(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot;
    static const unsigned char zero_digest[GATE_E_SEALED_LEAF_SHA256_BYTES] = {0};

    gate_e_sealed_leaf_snapshot_init(&snapshot);
    snapshot.byte_length = 17U;
    memset(snapshot.sha256, 0x5a, sizeof(snapshot.sha256));
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, zero_digest, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_INVALID_ARGUMENT);
    assert_empty_snapshot(&snapshot);

    snapshot.byte_length = 23U;
    memset(snapshot.sha256, 0xa5, sizeof(snapshot.sha256));
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, NULL, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_INVALID_ARGUMENT);
    assert_empty_snapshot(&snapshot);

    snapshot.byte_length = 29U;
    memset(snapshot.sha256, 0x3c, sizeof(snapshot.sha256));
    assert(gate_e_snapshot_held_leaf_v1(
               -1, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_INVALID_ARGUMENT);
    assert_empty_snapshot(&snapshot);
    destroy_fixture(&fixture);
}

static void test_rejects_untrusted_descriptor_and_input_drift(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot;
    unsigned char altered_digest[GATE_E_SEALED_LEAF_SHA256_BYTES];
    const int no_atime_descriptor = open(fixture.leaf, O_RDONLY | O_CLOEXEC);
    int pipe_descriptors[2] = {-1, -1};

    assert(no_atime_descriptor >= 0);
    gate_e_sealed_leaf_snapshot_init(&snapshot);
    assert(gate_e_snapshot_held_leaf_v1(
               no_atime_descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_SOURCE_UNSAFE);
    assert(close(no_atime_descriptor) == 0);

    memcpy(altered_digest, fixture.sha256, sizeof(altered_digest));
    altered_digest[0] ^= 0x01U;
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, altered_digest, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_DIGEST_MISMATCH);
    assert_empty_snapshot(&snapshot);
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length + 1U, &snapshot
           ) == GATE_E_SEALED_LEAF_SOURCE_UNSAFE);
    assert(pipe2(pipe_descriptors, O_CLOEXEC) == 0);
    assert(gate_e_snapshot_held_leaf_v1(
               pipe_descriptors[0], fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_SOURCE_UNSAFE);
    assert(close(pipe_descriptors[0]) == 0);
    assert(close(pipe_descriptors[1]) == 0);
    destroy_fixture(&fixture);
}

static void test_rejects_low_rlimit_fsize_before_any_memfd_write(void) {
    struct fixture fixture = make_fixture();
    const pid_t child = fork();
    int status;

    assert(child >= 0);
    if (child == 0) {
        struct gate_e_sealed_leaf_snapshot snapshot;
        struct rlimit limit;

        limit.rlim_cur = 0;
        limit.rlim_max = 0;
        gate_e_sealed_leaf_snapshot_init(&snapshot);
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0 ||
            gate_e_snapshot_held_leaf_v1(
                fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
            ) != GATE_E_SEALED_LEAF_MEMFD_UNSAFE ||
            snapshot.descriptor != -1) {
            _exit(1);
        }
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
    destroy_fixture(&fixture);
}

static void test_output_never_uses_standard_descriptors(void) {
    struct fixture fixture = make_fixture();
    const pid_t child = fork();
    int status;

    assert(child >= 0);
    if (child == 0) {
        struct gate_e_sealed_leaf_snapshot snapshot;

        (void)close(0);
        (void)close(1);
        (void)close(2);
        gate_e_sealed_leaf_snapshot_init(&snapshot);
        if (gate_e_snapshot_held_leaf_v1(
                fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
            ) != GATE_E_SEALED_LEAF_OK ||
            snapshot.descriptor < 3 ||
            gate_e_sealed_leaf_snapshot_close(&snapshot) != GATE_E_SEALED_LEAF_OK) {
            _exit(1);
        }
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
    destroy_fixture(&fixture);
}

static void test_close_never_interprets_a_corrupt_standard_descriptor_as_output(void) {
    const pid_t child = fork();
    int status;

    assert(child >= 0);
    if (child == 0) {
        struct gate_e_sealed_leaf_snapshot snapshot;

        gate_e_sealed_leaf_snapshot_init(&snapshot);
        snapshot.descriptor = 0;
        if (gate_e_sealed_leaf_snapshot_close(&snapshot) != GATE_E_SEALED_LEAF_INVALID_ARGUMENT ||
            fcntl(0, F_GETFD) < 0 || snapshot.descriptor != -1) {
            _exit(1);
        }
        _exit(0);
    }
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
}

static void test_failure_hooks_do_not_publish_or_leak_memfds(void) {
    struct fixture fixture = make_fixture();
    struct gate_e_sealed_leaf_snapshot snapshot;
    const memfd_create_invoker saved_memfd = invoke_memfd_create;
    const pread_invoker saved_pread = invoke_pread;
    const pwrite_invoker saved_pwrite = invoke_pwrite;
    const add_seals_invoker saved_add_seals = invoke_add_seals;
    size_t before;

    gate_e_sealed_leaf_snapshot_init(&snapshot);
    forced_memfd_calls = 0;
    invoke_memfd_create = forced_memfd_enosys;
    before = open_descriptor_count();
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_MEMFD_UNAVAILABLE);
    assert(forced_memfd_calls == 1);
    assert_empty_snapshot(&snapshot);
    assert(open_descriptor_count() == before);
    invoke_memfd_create = saved_memfd;

    invoke_pread = forced_pread_eio;
    before = open_descriptor_count();
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_SOURCE_UNREADABLE);
    assert_empty_snapshot(&snapshot);
    assert(open_descriptor_count() == before);
    invoke_pread = saved_pread;

    invoke_pwrite = forced_pwrite_eio;
    before = open_descriptor_count();
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_MEMFD_WRITE_FAILED);
    assert_empty_snapshot(&snapshot);
    assert(open_descriptor_count() == before);
    invoke_pwrite = saved_pwrite;

    invoke_add_seals = forced_add_seals_eperm;
    before = open_descriptor_count();
    assert(gate_e_snapshot_held_leaf_v1(
               fixture.descriptor, fixture.sha256, fixture.byte_length, &snapshot
           ) == GATE_E_SEALED_LEAF_MEMFD_SEAL_FAILED);
    assert_empty_snapshot(&snapshot);
    assert(open_descriptor_count() == before);
    invoke_add_seals = saved_add_seals;
    destroy_fixture(&fixture);
}

static bool identifier_character(const unsigned char value) {
    return (value >= (unsigned char)'a' && value <= (unsigned char)'z') ||
           (value >= (unsigned char)'A' && value <= (unsigned char)'Z') ||
           (value >= (unsigned char)'0' && value <= (unsigned char)'9') || value == (unsigned char)'_';
}

static bool source_has_identifier(const char *const source, const char *const identifier) {
    const size_t identifier_length = strlen(identifier);
    const char *cursor = source;

    while ((cursor = strstr(cursor, identifier)) != NULL) {
        const unsigned char previous = cursor == source ? 0U : (unsigned char)cursor[-1];
        const unsigned char following = (unsigned char)cursor[identifier_length];

        if (!identifier_character(previous) && !identifier_character(following)) {
            return true;
        }
        cursor += identifier_length;
    }
    return false;
}

static bool source_has_call(const char *const source, const char *const name) {
    const size_t name_length = strlen(name);
    const char *cursor = source;

    while ((cursor = strstr(cursor, name)) != NULL) {
        const unsigned char previous = cursor == source ? 0U : (unsigned char)cursor[-1];
        const char *following = cursor + name_length;

        if (!identifier_character(previous) && !identifier_character((unsigned char)*following)) {
            while (*following == ' ' || *following == '\t' || *following == '\n' || *following == '\r') {
                ++following;
            }
            if (*following == '(') {
                return true;
            }
        }
        cursor += name_length;
    }
    return false;
}

static size_t source_count_literal(const char *const source, const char *const literal) {
    const size_t literal_length = strlen(literal);
    const char *cursor = source;
    size_t count = 0;

    while ((cursor = strstr(cursor, literal)) != NULL) {
        ++count;
        cursor += literal_length;
    }
    return count;
}

static void test_source_has_no_path_or_execution_surface(void) {
    const char *const forbidden_calls[] = {
        "execve", "execveat", "fexecve", "fork", "vfork", "clone", "clone3", "posix_spawn",
        "system", "popen", "socket", "socketpair", "connect", "flock", "open", "openat", "creat",
        "mkdir", "unlink", "rename", "chmod", "chown", "setuid", "setgid", "mount", "umount2",
        "ioctl", "kill", "tgkill", "dlopen", "dlsym",
    };
    const char *const forbidden_identifiers[] = {
        "O_WRONLY", "O_CREAT", "O_TRUNC",
    };
    const char *const source_path = "gate_e_sealed_leaf_snapshot.c";
    FILE *const source = fopen(source_path, "rb");
    char *buffer;
    long length;

    assert(source != NULL);
    assert(fseek(source, 0, SEEK_END) == 0);
    length = ftell(source);
    assert(length > 0);
    assert(fseek(source, 0, SEEK_SET) == 0);
    buffer = calloc((size_t)length + 1U, sizeof(*buffer));
    assert(buffer != NULL);
    assert(fread(buffer, 1, (size_t)length, source) == (size_t)length);
    assert(fclose(source) == 0);
    assert(source_count_literal(buffer, "SYS_") == 2U);
    assert(source_count_literal(buffer, "SYS_memfd_create") == 2U);
    assert(source_count_literal(buffer, "syscall(SYS_memfd_create,") == 1U);
    for (size_t index = 0; index < sizeof(forbidden_calls) / sizeof(forbidden_calls[0]); ++index) {
        assert(!source_has_call(buffer, forbidden_calls[index]));
    }
    for (size_t index = 0; index < sizeof(forbidden_identifiers) / sizeof(forbidden_identifiers[0]); ++index) {
        assert(!source_has_identifier(buffer, forbidden_identifiers[index]));
    }
    free(buffer);
}

int main(void) {
    test_sha256_matches_standard_vectors();
    test_success_seals_anonymous_copy_and_preserves_source_offset();
    test_pins_source_before_the_caller_fd_number_is_reused();
    test_alias_digest_and_initialization_contract();
    test_invalid_arguments_clear_a_valid_output_object();
    test_rejects_untrusted_descriptor_and_input_drift();
    test_rejects_low_rlimit_fsize_before_any_memfd_write();
    test_output_never_uses_standard_descriptors();
    test_close_never_interprets_a_corrupt_standard_descriptor_as_output();
    test_failure_hooks_do_not_publish_or_leak_memfds();
    test_source_has_no_path_or_execution_surface();
    (void)puts("gate-e sealed leaf snapshot tests passed");
    return 0;
}
