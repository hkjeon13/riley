CC ?= cc
NM ?= nm
CPPFLAGS ?= -D_GNU_SOURCE
CFLAGS ?= -O2
LDFLAGS ?=

REQUIRED_CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wshadow -Wformat=2 \
	-Wstrict-prototypes -Wmissing-prototypes -Werror=vla \
	-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
REQUIRED_LDFLAGS = -pie -Wl,-z,relro,-z,now
TOOL_DIRECTORY := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PARSER_DIRECTORY := $(abspath $(TOOL_DIRECTORY)../gate-e-execution-closure-manifest-parser)
DEPENDENCY_INCLUDES := -I$(PARSER_DIRECTORY)

.PHONY: test analyze

test:
	@set -eu; \
	build_dir="$$(mktemp -d /tmp/riley-gate-e-execution-closure-held-fds.XXXXXX)"; \
	parser_object="$$build_dir/gate_e_execution_closure_manifest_v1.o"; \
	binder_object="$$build_dir/gate_e_execution_closure_held_fds_v1.o"; \
	test_binary="$$build_dir/test_gate_e_execution_closure_held_fds_v1"; \
	api_test_binary="$$build_dir/test_gate_e_execution_closure_held_fds_api"; \
	binder_symbols="$$build_dir/binder-symbols.txt"; \
	binder_symbol_names="$$build_dir/binder-symbol-names.txt"; \
	binder_undefined="$$build_dir/binder-undefined.txt"; \
	binder_undefined_names="$$build_dir/binder-undefined-names.txt"; \
	cleanup() { rm -f "$$parser_object" "$$binder_object" "$$test_binary" "$$api_test_binary" "$$binder_symbols" "$$binder_symbol_names" "$$binder_undefined" "$$binder_undefined_names"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -c "$(PARSER_DIRECTORY)/gate_e_execution_closure_manifest_v1.c" -o "$$parser_object"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -c "$(TOOL_DIRECTORY)gate_e_execution_closure_held_fds_v1.c" -o "$$binder_object"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_execution_closure_held_fds_v1.c" "$$parser_object" "$$binder_object" -o "$$test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS) -Wl,--wrap=pread -Wl,--wrap=pread64 -Wl,--wrap=__pread_chk -Wl,--wrap=__pread64_chk; \
	"$$test_binary"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_execution_closure_held_fds_api.c" "$$parser_object" "$$binder_object" -o "$$api_test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	"$$api_test_binary"; \
	$(NM) -g --defined-only "$$binder_object" >"$$binder_symbols"; \
	awk '{print $$NF}' "$$binder_symbols" >"$$binder_symbol_names"; \
	for expected_symbol in gate_e_execution_closure_held_fds_v1_init gate_e_bind_execution_closure_held_fds_v1 gate_e_execution_closure_held_fds_v1_recheck gate_e_execution_closure_held_fds_v1_close; do \
		grep -Fx "$$expected_symbol" "$$binder_symbol_names" >/dev/null; \
	done; \
	for observed_symbol in $$(cat "$$binder_symbol_names"); do \
		case "$$observed_symbol" in gate_e_execution_closure_held_fds_v1_init|gate_e_bind_execution_closure_held_fds_v1|gate_e_execution_closure_held_fds_v1_recheck|gate_e_execution_closure_held_fds_v1_close) ;; *) printf '%s\n' "unexpected held-fds library symbol: $$observed_symbol" >&2; exit 1;; esac; \
	done; \
	$(NM) --undefined-only "$$binder_object" >"$$binder_undefined"; \
	awk '{sub(/@.*/, "", $$NF); print $$NF}' "$$binder_undefined" >"$$binder_undefined_names"; \
	for forbidden_symbol in open openat openat2 creat mkdir unlink rename chmod chown setuid setgid mount umount2 execve execveat fexecve fork vfork clone clone3 posix_spawn system popen socket socketpair connect flock read write pwrite lseek ioctl kill tgkill dlopen dlsym malloc calloc realloc free; do \
		if grep -Fx "$$forbidden_symbol" "$$binder_undefined_names" >/dev/null; then \
			printf '%s\n' "forbidden held-fds dependency: $$forbidden_symbol" >&2; exit 1; \
		fi; \
	done

analyze:
	@set -eu; \
	build_dir="$$(mktemp -d /tmp/riley-gate-e-execution-closure-held-fds-analyze.XXXXXX)"; \
	object_file="$$build_dir/gate_e_execution_closure_held_fds_v1.o"; \
	cleanup() { rm -f "$$object_file"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -fanalyzer -c "$(TOOL_DIRECTORY)gate_e_execution_closure_held_fds_v1.c" -o "$$object_file"
