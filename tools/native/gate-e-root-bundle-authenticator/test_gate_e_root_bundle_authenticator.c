#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GATE_E_ROOT_BUNDLE_AUTHENTICATOR_LIBRARY
#define GATE_E_ROOT_BUNDLE_AUTHENTICATOR_TESTING
#include "gate_e_root_bundle_authenticator.c"

#include <assert.h>
#include <limits.h>

struct fixture {
    char root[PATH_MAX];
    char opt[PATH_MAX];
    char riley[PATH_MAX];
    char bundle[PATH_MAX];
    char manifest[PATH_MAX];
    char bootstrap[PATH_MAX];
    char core[PATH_MAX];
};

static const char BOOTSTRAP_CONTENTS[] = "guardian bootstrap fixture\n";
static const char CORE_CONTENTS[] = "guardian core fixture\n";

static int forced_enosys_calls = 0;

static int approved_filesystem(
    const int descriptor,
    struct statfs *const metadata
) {
    if (fstatfs(descriptor, metadata) != 0) {
        return -1;
    }
    metadata->f_type = EXT4_SUPER_MAGIC;
    return 0;
}

static int unapproved_filesystem(
    const int descriptor,
    struct statfs *const metadata
) {
    if (fstatfs(descriptor, metadata) != 0) {
        return -1;
    }
    metadata->f_type = OVERLAYFS_SUPER_MAGIC;
    return 0;
}

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

static ssize_t absent_xattr(
    const int descriptor,
    const char *const name,
    void *const value,
    const size_t size
) {
    (void)descriptor;
    (void)name;
    (void)value;
    (void)size;
    errno = ENODATA;
    return -1;
}

static ssize_t present_acl_xattr(
    const int descriptor,
    const char *const name,
    void *const value,
    const size_t size
) {
    (void)descriptor;
    (void)name;
    (void)value;
    (void)size;
    return 1;
}

static ssize_t present_bootstrap_capability(
    const int descriptor,
    const char *const name,
    void *const value,
    const size_t size
) {
    (void)descriptor;
    (void)value;
    (void)size;
    if (strcmp(name, "security.capability") == 0) {
        return 1;
    }
    errno = ENODATA;
    return -1;
}

static ssize_t unverifiable_xattr(
    const int descriptor,
    const char *const name,
    void *const value,
    const size_t size
) {
    (void)descriptor;
    (void)name;
    (void)value;
    (void)size;
    errno = EOPNOTSUPP;
    return -1;
}

static void write_exact_file(const char *const path, const char *const contents, const mode_t mode) {
    const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, mode);
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
    assert(chmod(path, mode) == 0);
}

static void sha256_text(const char *const text, char output[SHA256_HEX_BYTES + 1]) {
    struct sha256_context context;
    unsigned char digest[SHA256_DIGEST_BYTES];

    sha256_init(&context);
    sha256_update(&context, (const unsigned char *)text, strlen(text));
    sha256_final(&context, digest);
    digest_to_hex(digest, output);
}

static void write_canonical_manifest(const struct fixture *const fixture) {
    char bootstrap_digest[SHA256_HEX_BYTES + 1];
    char core_digest[SHA256_HEX_BYTES + 1];
    char manifest[MAX_MANIFEST_BYTES + 1];
    int rendered;

    sha256_text(BOOTSTRAP_CONTENTS, bootstrap_digest);
    sha256_text(CORE_CONTENTS, core_digest);
    rendered = snprintf(
        manifest,
        sizeof(manifest),
        "{\"bootstrap\":{\"byte_length\":%zu,\"filename\":\"%s\",\"sha256\":\"%s\"},"
        "\"core\":{\"byte_length\":%zu,\"filename\":\"%s\",\"sha256\":\"%s\"},"
        "\"schema_version\":\"%s\"}\n",
        strlen(BOOTSTRAP_CONTENTS),
        BOOTSTRAP_NAME,
        bootstrap_digest,
        strlen(CORE_CONTENTS),
        CORE_NAME,
        core_digest,
        MANIFEST_SCHEMA
    );
    assert(rendered > 0 && (size_t)rendered < sizeof(manifest));
    write_exact_file(fixture->manifest, manifest, 0644);
}

static void write_canonical_bundle(const struct fixture *const fixture) {
    write_exact_file(fixture->bootstrap, BOOTSTRAP_CONTENTS, 0755);
    write_exact_file(fixture->core, CORE_CONTENTS, 0644);
    write_canonical_manifest(fixture);
}

static struct fixture make_fixture(void) {
    struct fixture fixture = {0};
    char template[] = "/tmp/riley-root-bundle-authenticator.XXXXXX";
    char *const root = mkdtemp(template);

    assert(root != NULL);
    assert(strlen(root) < sizeof(fixture.root));
    memcpy(fixture.root, root, strlen(root) + 1U);
    assert(chmod(fixture.root, 0755) == 0);
    assert(snprintf(fixture.opt, sizeof(fixture.opt), "%s/opt", fixture.root) > 0);
    assert(snprintf(fixture.riley, sizeof(fixture.riley), "%s/riley", fixture.opt) > 0);
    assert(snprintf(fixture.bundle, sizeof(fixture.bundle), "%s/rc3-gate-e-v1", fixture.riley) > 0);
    assert(mkdir(fixture.opt, 0755) == 0);
    assert(mkdir(fixture.riley, 0755) == 0);
    assert(mkdir(fixture.bundle, 0755) == 0);
    assert(chmod(fixture.opt, 0755) == 0);
    assert(chmod(fixture.riley, 0755) == 0);
    assert(chmod(fixture.bundle, 0755) == 0);
    assert(snprintf(fixture.manifest, sizeof(fixture.manifest), "%s/%s", fixture.bundle, MANIFEST_NAME) > 0);
    assert(snprintf(fixture.bootstrap, sizeof(fixture.bootstrap), "%s/%s", fixture.bundle, BOOTSTRAP_NAME) > 0);
    assert(snprintf(fixture.core, sizeof(fixture.core), "%s/%s", fixture.bundle, CORE_NAME) > 0);
    write_canonical_bundle(&fixture);
    return fixture;
}

static void destroy_fixture(const struct fixture *const fixture) {
    (void)unlink(fixture->manifest);
    (void)unlink(fixture->bootstrap);
    (void)unlink(fixture->core);
    (void)rmdir(fixture->bundle);
    (void)rmdir(fixture->riley);
    (void)rmdir(fixture->opt);
    (void)rmdir(fixture->root);
}

static enum anchor_reason acquire_fixture(
    const struct fixture *const fixture,
    struct gate_e_root_bundle_held_v1 *const held
) {
    const int descriptor = open(fixture->root, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    enum anchor_reason reason;

    assert(descriptor >= 0);
    reason = acquire_bundle_from_held_prefix_fd(descriptor, getuid(), getgid(), held);
    if (close(descriptor) != 0 && reason == ANCHOR_OK) {
        reason = ANCHOR_CLOSE_FAILED;
    }
    return reason;
}

static enum anchor_reason authenticate_fixture(const struct fixture *const fixture) {
    struct gate_e_root_bundle_held_v1 held;
    enum anchor_reason reason;

    gate_e_root_bundle_held_v1_init(&held);
    reason = acquire_fixture(fixture, &held);
    if (reason == ANCHOR_OK) {
        reason = gate_e_root_bundle_held_v1_close(&held);
    }
    return reason;
}

static void assert_held_is_cleared(const struct gate_e_root_bundle_held_v1 *const held) {
    static const unsigned char zero_digest[GATE_E_ROOT_BUNDLE_SHA256_DIGEST_BYTES_V1] = {0};

    for (size_t index = 0; index < GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1; ++index) {
        assert(held->directories[index].descriptor == -1);
        assert(held->directories[index].identity.device == 0);
        assert(held->directories[index].identity.inode == 0);
        assert(held->directories[index].identity.mode == 0);
        assert(held->directories[index].identity.links == 0);
    }
    assert(held->manifest.descriptor == -1);
    assert(held->bootstrap.descriptor == -1);
    assert(held->core.descriptor == -1);
    assert(held->manifest.byte_length == 0);
    assert(held->bootstrap.byte_length == 0);
    assert(held->core.byte_length == 0);
    assert(memcmp(held->manifest.digest, zero_digest, sizeof(zero_digest)) == 0);
    assert(memcmp(held->bootstrap.digest, zero_digest, sizeof(zero_digest)) == 0);
    assert(memcmp(held->core.digest, zero_digest, sizeof(zero_digest)) == 0);
}

static void test_sha256_and_filesystem_policy(void) {
    char digest[SHA256_HEX_BYTES + 1];
    const struct open_how root_how = directory_open_how(true);
    const struct open_how child_how = directory_open_how(false);
    const struct open_how leaf_how = file_open_how();

    sha256_text("abc", digest);
    assert(strcmp(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);
    assert(allowed_filesystem_type(EXT4_SUPER_MAGIC));
    assert(allowed_filesystem_type(XFS_SUPER_MAGIC));
    assert(allowed_filesystem_type(BTRFS_SUPER_MAGIC));
    assert(!allowed_filesystem_type(OVERLAYFS_SUPER_MAGIC));
    assert((root_how.flags & O_NOATIME) != 0U);
    assert((child_how.flags & O_NOATIME) != 0U);
    assert((leaf_how.flags & O_NOATIME) != 0U);
    assert((child_how.resolve & (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV)) ==
           (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV));
    assert((leaf_how.resolve & (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV)) ==
           (RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV));
}

static void test_report_and_cli_are_non_authoritative(void) {
    const char *const valid_argv[] = {"gate_e_root_bundle_authenticator", "--authenticate-root-bundle-v1"};
    const char *const invalid_argv[] = {"gate_e_root_bundle_authenticator", "--any-path"};
    const char *const empty_name_argv[] = {"", "--authenticate-root-bundle-v1"};
    FILE *const report = tmpfile();
    char rendered[2048] = {0};
    size_t count;

    assert(accepted_cli(2, valid_argv));
    assert(!accepted_cli(2, invalid_argv));
    assert(!accepted_cli(1, valid_argv));
    assert(!accepted_cli(2, NULL));
    assert(strcmp(program_name_for_usage(2, valid_argv), "gate_e_root_bundle_authenticator") == 0);
    assert(strcmp(program_name_for_usage(0, valid_argv), "gate_e_root_bundle_authenticator") == 0);
    assert(strcmp(program_name_for_usage(2, empty_name_argv), "gate_e_root_bundle_authenticator") == 0);
    assert(strcmp(program_name_for_usage(0, NULL), "gate_e_root_bundle_authenticator") == 0);
    assert(report != NULL);
    assert(print_report(report, ANCHOR_OK));
    assert(fseek(report, 0, SEEK_SET) == 0);
    count = fread(rendered, 1, sizeof(rendered) - 1U, report);
    assert(count > 0 && count < sizeof(rendered));
    assert(fclose(report) == 0);
    assert(strstr(rendered, "\"status\":\"not-established\"") != NULL);
    assert(strstr(rendered, "\"object_observation_status\":\"checked\"") != NULL);
    assert(strstr(rendered, "\"host_initial_namespace\":\"not-established\"") != NULL);
    assert(strstr(rendered, "\"execution_authority\":\"not-established\"") != NULL);
    assert(strstr(rendered, "\"actual_gate_e_producer\":\"not-established\"") != NULL);
    assert(strstr(rendered, "\"qualification_status\":\"not-run\"") != NULL);
    assert(strstr(rendered, "\"reason_code\":null") != NULL);
}

static void test_valid_fixture_and_manifest_contract(void) {
    const struct fixture fixture = make_fixture();
    const statfs_invoker saved_statfs = invoke_fstatfs;

    invoke_fstatfs = approved_filesystem;
    invoke_fgetxattr = absent_xattr;
    assert(authenticate_fixture(&fixture) == ANCHOR_OK);
    invoke_fgetxattr = fgetxattr;
    invoke_fstatfs = saved_statfs;
    destroy_fixture(&fixture);
}

static void test_held_bundle_api_retains_rechecks_and_closes(void) {
    const struct fixture fixture = make_fixture();
    struct gate_e_root_bundle_held_v1 held;
    const statfs_invoker saved_statfs = invoke_fstatfs;
    const xattr_invoker saved_xattr = invoke_fgetxattr;

    gate_e_root_bundle_held_v1_init(&held);
    assert_held_is_cleared(&held);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_INVALID_ARGUMENT);
    invoke_fstatfs = approved_filesystem;
    invoke_fgetxattr = absent_xattr;
    assert(acquire_fixture(&fixture, &held) == ANCHOR_OK);
    for (size_t index = 0; index < GATE_E_ROOT_BUNDLE_HELD_DIRECTORY_COUNT_V1; ++index) {
        const int descriptor_flags = fcntl(held.directories[index].descriptor, F_GETFD);

        assert(held.directories[index].descriptor >= 3);
        assert(descriptor_flags >= 0 && (descriptor_flags & FD_CLOEXEC) != 0);
    }
    assert(held.manifest.descriptor >= 3 && held.bootstrap.descriptor >= 3 && held.core.descriptor >= 3);
    assert(held.manifest.byte_length > 0 && held.bootstrap.byte_length == strlen(BOOTSTRAP_CONTENTS));
    assert(held.core.byte_length == strlen(CORE_CONTENTS));
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_OK);
    assert(acquire_fixture(&fixture, &held) == ANCHOR_INVALID_ARGUMENT);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_OK);
    assert(gate_e_root_bundle_acquire_fixed_v1(&held) == ANCHOR_INVALID_ARGUMENT);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_OK);

    {
        const int descriptor_flags = fcntl(held.bootstrap.descriptor, F_GETFD);

        assert(descriptor_flags >= 0);
        assert(fcntl(held.bootstrap.descriptor, F_SETFD, descriptor_flags & ~FD_CLOEXEC) == 0);
        assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_INVALID_ARGUMENT);
        assert(fcntl(held.bootstrap.descriptor, F_SETFD, descriptor_flags) == 0);
    }
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_OK);
    assert(chmod(fixture.core, 0600) == 0);
    assert(gate_e_root_bundle_held_v1_recheck(&held) == ANCHOR_OBJECT_RACED);
    assert(gate_e_root_bundle_held_v1_close(&held) == ANCHOR_OK);
    assert_held_is_cleared(&held);
    assert(gate_e_root_bundle_held_v1_close(&held) == ANCHOR_OK);
    assert_held_is_cleared(&held);
    invoke_fgetxattr = saved_xattr;
    invoke_fstatfs = saved_statfs;
    destroy_fixture(&fixture);
}

static void test_held_bundle_api_failure_clears_output(void) {
    const struct fixture fixture = make_fixture();
    struct gate_e_root_bundle_held_v1 held;
    const openat2_invoker saved_openat2 = invoke_openat2;
    const statfs_invoker saved_statfs = invoke_fstatfs;
    const xattr_invoker saved_xattr = invoke_fgetxattr;

    gate_e_root_bundle_held_v1_init(&held);
    invoke_fstatfs = approved_filesystem;
    invoke_fgetxattr = absent_xattr;
    forced_enosys_calls = 0;
    invoke_openat2 = forced_enosys_openat2;
    assert(acquire_fixture(&fixture, &held) == ANCHOR_OPENAT2_UNAVAILABLE);
    assert(forced_enosys_calls == 1);
    assert_held_is_cleared(&held);
    invoke_openat2 = saved_openat2;
    invoke_fgetxattr = saved_xattr;
    invoke_fstatfs = saved_statfs;
    destroy_fixture(&fixture);
}

static void test_rejects_openat2_fallback_acl_and_capability(void) {
    const struct fixture fixture = make_fixture();
    const openat2_invoker saved_openat2 = invoke_openat2;
    const statfs_invoker saved_statfs = invoke_fstatfs;
    const xattr_invoker saved_xattr = invoke_fgetxattr;

    invoke_fstatfs = approved_filesystem;
    forced_enosys_calls = 0;
    invoke_openat2 = forced_enosys_openat2;
    assert(authenticate_fixture(&fixture) == ANCHOR_OPENAT2_UNAVAILABLE);
    assert(forced_enosys_calls == 1);
    invoke_openat2 = saved_openat2;

    invoke_fgetxattr = present_acl_xattr;
    assert(authenticate_fixture(&fixture) == ANCHOR_ACL_PRESENT);
    invoke_fgetxattr = present_bootstrap_capability;
    assert(authenticate_fixture(&fixture) == ANCHOR_CAPABILITY_PRESENT);
    invoke_fgetxattr = unverifiable_xattr;
    assert(authenticate_fixture(&fixture) == ANCHOR_ACL_UNVERIFIABLE);
    invoke_fgetxattr = saved_xattr;
    invoke_fstatfs = unapproved_filesystem;
    assert(authenticate_fixture(&fixture) == ANCHOR_UNSAFE_FILESYSTEM);
    invoke_fstatfs = saved_statfs;
    destroy_fixture(&fixture);
}

static void test_rejects_mode_link_manifest_and_digest_drift(void) {
    struct fixture fixture = make_fixture();
    const char replacement[] = "changed guardian core\n";

    invoke_fstatfs = approved_filesystem;
    invoke_fgetxattr = absent_xattr;
    assert(chmod(fixture.bundle, 0750) == 0);
    assert(authenticate_fixture(&fixture) == ANCHOR_UNSAFE_DIRECTORY);
    assert(chmod(fixture.bundle, 0755) == 0);

    {
        char extra[PATH_MAX] = {0};
        assert(snprintf(extra, sizeof(extra), "%s/core-link", fixture.bundle) > 0);
        assert(link(fixture.core, extra) == 0);
        assert(authenticate_fixture(&fixture) == ANCHOR_UNSAFE_FILE);
        assert(unlink(extra) == 0);
    }

    assert(unlink(fixture.manifest) == 0);
    write_exact_file(fixture.manifest, "{\"bootstrap\":{}}\n", 0644);
    assert(authenticate_fixture(&fixture) == ANCHOR_MANIFEST_INVALID);
    assert(unlink(fixture.manifest) == 0);
    write_canonical_manifest(&fixture);

    assert(unlink(fixture.core) == 0);
    write_exact_file(fixture.core, replacement, 0644);
    assert(authenticate_fixture(&fixture) == ANCHOR_DIGEST_MISMATCH);
    invoke_fgetxattr = fgetxattr;
    invoke_fstatfs = fstatfs;
    destroy_fixture(&fixture);
}

static void test_rejects_symlink_and_exact_parser_drift(void) {
    struct fixture fixture = make_fixture();
    char malformed[MAX_MANIFEST_BYTES + 1];
    int rendered;

    invoke_fstatfs = approved_filesystem;
    invoke_fgetxattr = absent_xattr;
    assert(unlink(fixture.core) == 0);
    assert(symlink(BOOTSTRAP_NAME, fixture.core) == 0);
    assert(authenticate_fixture(&fixture) == ANCHOR_UNSAFE_FILE);
    assert(unlink(fixture.core) == 0);
    write_exact_file(fixture.core, CORE_CONTENTS, 0644);

    assert(unlink(fixture.manifest) == 0);
    rendered = snprintf(
        malformed,
        sizeof(malformed),
        "{\"core\":{},\"bootstrap\":{},\"schema_version\":\"%s\"}\n",
        MANIFEST_SCHEMA
    );
    assert(rendered > 0 && (size_t)rendered < sizeof(malformed));
    write_exact_file(fixture.manifest, malformed, 0644);
    assert(authenticate_fixture(&fixture) == ANCHOR_MANIFEST_INVALID);
    invoke_fgetxattr = fgetxattr;
    invoke_fstatfs = fstatfs;
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

static void test_source_has_no_operational_surface(void) {
    const char *const forbidden_calls[] = {
        "execve", "execveat", "fexecve", "fork", "vfork", "clone", "clone3", "posix_spawn",
        "system", "popen", "socket", "socketpair", "connect", "flock", "open", "openat", "creat",
        "mkdir", "unlink", "rename", "chmod", "chown", "setuid", "setgid", "mount", "umount2",
        "ioctl", "kill", "tgkill", "write", "pwrite", "dlopen", "dlsym",
    };
    const char *const forbidden_identifiers[] = {
        "O_WRONLY", "O_RDWR",
    };
    const char *const source_path = "gate_e_root_bundle_authenticator.c";
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
    assert(source_count_literal(buffer, "SYS_openat2") == 2U);
    assert(source_count_literal(buffer, "syscall(") == 1U);
    assert(source_count_literal(buffer, "syscall(SYS_openat2,") == 1U);
    for (size_t index = 0; index < sizeof(forbidden_calls) / sizeof(forbidden_calls[0]); ++index) {
        assert(!source_has_call(buffer, forbidden_calls[index]));
    }
    for (size_t index = 0; index < sizeof(forbidden_identifiers) / sizeof(forbidden_identifiers[0]); ++index) {
        assert(!source_has_identifier(buffer, forbidden_identifiers[index]));
    }
    free(buffer);
}

int main(void) {
    test_sha256_and_filesystem_policy();
    test_report_and_cli_are_non_authoritative();
    test_valid_fixture_and_manifest_contract();
    test_held_bundle_api_retains_rechecks_and_closes();
    test_held_bundle_api_failure_clears_output();
    test_rejects_openat2_fallback_acl_and_capability();
    test_rejects_mode_link_manifest_and_digest_drift();
    test_rejects_symlink_and_exact_parser_drift();
    test_source_has_no_operational_surface();
    (void)puts("gate-e root bundle authenticator tests passed");
    return 0;
}
