CC ?= cc
NM ?= nm
CPPFLAGS ?= -D_GNU_SOURCE
CFLAGS ?= -O2
LDFLAGS ?=

REQUIRED_CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic \
	-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
REQUIRED_LDFLAGS = -pie -Wl,-z,relro,-z,now
TOOL_DIRECTORY := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: test analyze

test:
	@set -eu; \
	build_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/riley-gate-e-root-bundle-authenticator.XXXXXX")"; \
	test_binary="$$build_dir/test_gate_e_root_bundle_authenticator"; \
	tool_binary="$$build_dir/gate_e_root_bundle_authenticator"; \
	invalid_stdout="$$build_dir/invalid.stdout"; \
	invalid_stderr="$$build_dir/invalid.stderr"; \
	dynamic_symbols="$$build_dir/dynamic-symbols.txt"; \
	dynamic_symbol_names="$$build_dir/dynamic-symbol-names.txt"; \
	cleanup() { rm -f "$$test_binary" "$$tool_binary" "$$invalid_stdout" "$$invalid_stderr" "$$dynamic_symbols" "$$dynamic_symbol_names"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_root_bundle_authenticator.c" -o "$$test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	(cd "$(TOOL_DIRECTORY)" && "$$test_binary"); \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)gate_e_root_bundle_authenticator.c" -o "$$tool_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	$(NM) -D --undefined-only "$$tool_binary" >"$$dynamic_symbols"; \
	awk '{sub(/@.*/, "", $$NF); print $$NF}' "$$dynamic_symbols" >"$$dynamic_symbol_names"; \
	for forbidden_symbol in execve execveat fexecve fork vfork clone clone3 posix_spawn system popen socket socketpair connect flock open openat creat mkdir unlink rename chmod chown setuid setgid mount umount2 ioctl kill tgkill write pwrite dlopen dlsym; do \
		if grep -Fx "$$forbidden_symbol" "$$dynamic_symbol_names" >/dev/null; then \
			printf '%s\n' "forbidden dynamic symbol: $$forbidden_symbol" >&2; exit 1; \
		fi; \
	done; \
	if "$$tool_binary" --unexpected-argument >"$$invalid_stdout" 2>"$$invalid_stderr"; then \
		printf '%s\n' "invalid public invocation unexpectedly succeeded" >&2; exit 1; \
	else \
		invalid_status=$$?; \
	fi; \
	test "$$invalid_status" -eq 64; \
	grep -F -- "--authenticate-root-bundle-v1" "$$invalid_stderr" >/dev/null

analyze:
	@set -eu; \
	build_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/riley-gate-e-root-bundle-analyze.XXXXXX")"; \
	object_file="$$build_dir/gate_e_root_bundle_authenticator.o"; \
	cleanup() { rm -f "$$object_file"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) -fanalyzer -c "$(TOOL_DIRECTORY)gate_e_root_bundle_authenticator.c" -o "$$object_file"
