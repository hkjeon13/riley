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
ROOT_BUNDLE_DIRECTORY := $(abspath $(TOOL_DIRECTORY)../gate-e-root-bundle-authenticator)
SNAPSHOT_DIRECTORY := $(abspath $(TOOL_DIRECTORY)../gate-e-sealed-leaf-snapshot)
DEPENDENCY_INCLUDES := -I$(ROOT_BUNDLE_DIRECTORY) -I$(SNAPSHOT_DIRECTORY)

.PHONY: test analyze

test:
	@set -eu; \
	build_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/riley-gate-e-root-bundle-sealed-leaves.XXXXXX")"; \
	root_object="$$build_dir/gate_e_root_bundle_held_v1.o"; \
	snapshot_object="$$build_dir/gate_e_sealed_leaf_snapshot.o"; \
	bridge_object="$$build_dir/gate_e_root_bundle_sealed_leaves_v1.o"; \
	test_binary="$$build_dir/test_gate_e_root_bundle_sealed_leaves_v1"; \
	api_test_binary="$$build_dir/test_gate_e_root_bundle_sealed_leaves_api"; \
	root_symbols="$$build_dir/root-symbols.txt"; \
	root_symbol_names="$$build_dir/root-symbol-names.txt"; \
	bridge_symbols="$$build_dir/bridge-symbols.txt"; \
	bridge_symbol_names="$$build_dir/bridge-symbol-names.txt"; \
	bridge_undefined="$$build_dir/bridge-undefined.txt"; \
	bridge_undefined_names="$$build_dir/bridge-undefined-names.txt"; \
	cleanup() { rm -f "$$root_object" "$$snapshot_object" "$$bridge_object" "$$test_binary" "$$api_test_binary" "$$root_symbols" "$$root_symbol_names" "$$bridge_symbols" "$$bridge_symbol_names" "$$bridge_undefined" "$$bridge_undefined_names"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -DGATE_E_ROOT_BUNDLE_AUTHENTICATOR_LIBRARY -c "$(ROOT_BUNDLE_DIRECTORY)/gate_e_root_bundle_authenticator.c" -o "$$root_object"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -c "$(SNAPSHOT_DIRECTORY)/gate_e_sealed_leaf_snapshot.c" -o "$$snapshot_object"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -c "$(TOOL_DIRECTORY)gate_e_root_bundle_sealed_leaves_v1.c" -o "$$bridge_object"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_root_bundle_sealed_leaves_v1.c" "$$root_object" "$$snapshot_object" "$$bridge_object" -o "$$test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	"$$test_binary"; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_root_bundle_sealed_leaves_api.c" "$$root_object" "$$snapshot_object" "$$bridge_object" -o "$$api_test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	"$$api_test_binary"; \
	$(NM) -g --defined-only "$$root_object" >"$$root_symbols"; \
	awk '{print $$NF}' "$$root_symbols" >"$$root_symbol_names"; \
	for expected_symbol in gate_e_root_bundle_held_v1_init gate_e_root_bundle_acquire_fixed_v1 gate_e_root_bundle_held_v1_recheck gate_e_root_bundle_held_v1_close; do \
		grep -Fx "$$expected_symbol" "$$root_symbol_names" >/dev/null; \
	done; \
	for observed_symbol in $$(cat "$$root_symbol_names"); do \
		case "$$observed_symbol" in gate_e_root_bundle_held_v1_init|gate_e_root_bundle_acquire_fixed_v1|gate_e_root_bundle_held_v1_recheck|gate_e_root_bundle_held_v1_close) ;; *) printf '%s\n' "unexpected root-bundle library symbol: $$observed_symbol" >&2; exit 1;; esac; \
	done; \
	$(NM) -g --defined-only "$$bridge_object" >"$$bridge_symbols"; \
	awk '{print $$NF}' "$$bridge_symbols" >"$$bridge_symbol_names"; \
	for expected_symbol in gate_e_root_bundle_sealed_leaves_v1_init gate_e_snapshot_held_root_bundle_leaves_v1 gate_e_root_bundle_sealed_leaves_v1_close; do \
		grep -Fx "$$expected_symbol" "$$bridge_symbol_names" >/dev/null; \
	done; \
	for observed_symbol in $$(cat "$$bridge_symbol_names"); do \
		case "$$observed_symbol" in gate_e_root_bundle_sealed_leaves_v1_init|gate_e_snapshot_held_root_bundle_leaves_v1|gate_e_root_bundle_sealed_leaves_v1_close) ;; *) printf '%s\n' "unexpected sealed-leaves library symbol: $$observed_symbol" >&2; exit 1;; esac; \
	done; \
	$(NM) --undefined-only "$$bridge_object" >"$$bridge_undefined"; \
	awk '{sub(/@.*/, "", $$NF); print $$NF}' "$$bridge_undefined" >"$$bridge_undefined_names"; \
	for forbidden_symbol in gate_e_root_bundle_acquire_fixed_v1 execve execveat fexecve fork vfork clone clone3 posix_spawn system popen socket socketpair connect flock open openat creat mkdir unlink rename chmod chown setuid setgid mount umount2 ioctl kill tgkill write pwrite dlopen dlsym; do \
		if grep -Fx "$$forbidden_symbol" "$$bridge_undefined_names" >/dev/null; then \
			printf '%s\n' "forbidden sealed-leaves dependency: $$forbidden_symbol" >&2; exit 1; \
		fi; \
	done

analyze:
	@set -eu; \
	build_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/riley-gate-e-root-bundle-sealed-leaves-analyze.XXXXXX")"; \
	object_file="$$build_dir/gate_e_root_bundle_sealed_leaves_v1.o"; \
	cleanup() { rm -f "$$object_file"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(DEPENDENCY_INCLUDES) $(CFLAGS) $(REQUIRED_CFLAGS) -fanalyzer -c "$(TOOL_DIRECTORY)gate_e_root_bundle_sealed_leaves_v1.c" -o "$$object_file"
