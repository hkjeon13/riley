CC ?= cc
NM ?= nm
CPPFLAGS ?=
CFLAGS ?= -O2
LDFLAGS ?=

REQUIRED_CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic -Wconversion -Wshadow -Wformat=2 \
	-Wstrict-prototypes -Wmissing-prototypes -Werror=vla \
	-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
REQUIRED_LDFLAGS = -pie -Wl,-z,relro,-z,now
TOOL_DIRECTORY := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: test analyze

test:
	@set -eu; \
	build_dir="$$(mktemp -d /tmp/riley-gate-e-controller-restart-witness.XXXXXX)"; \
	test_binary="$$build_dir/test_gate_e_guardian_controller_restart_witness_v1"; \
	tool_object="$$build_dir/gate_e_guardian_controller_restart_witness_v1.o"; \
	undefined_symbols="$$build_dir/undefined-symbols.txt"; \
	undefined_symbol_names="$$build_dir/undefined-symbol-names.txt"; \
	cleanup() { rm -f "$$test_binary" "$$tool_object" "$$undefined_symbols" "$$undefined_symbol_names"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) "$(TOOL_DIRECTORY)test_gate_e_guardian_controller_restart_witness_v1.c" -o "$$test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	(cd "$(TOOL_DIRECTORY)" && "$$test_binary"); \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) -c "$(TOOL_DIRECTORY)gate_e_guardian_controller_restart_witness_v1.c" -o "$$tool_object"; \
	$(NM) --undefined-only "$$tool_object" >"$$undefined_symbols"; \
	awk '{sub(/@.*/, "", $$NF); print $$NF}' "$$undefined_symbols" >"$$undefined_symbol_names"; \
	while IFS= read -r undefined_symbol; do \
		case "$$undefined_symbol" in \
			__stack_chk_fail|memcmp|memcpy|memmove|memset|__asan_*|__ubsan_*) ;; \
			*) printf '%s\n' "unexpected undefined symbol: $$undefined_symbol" >&2; exit 1 ;; \
		esac; \
	done < "$$undefined_symbol_names"

analyze:
	@set -eu; \
	build_dir="$$(mktemp -d /tmp/riley-gate-e-controller-restart-witness-analyze.XXXXXX)"; \
	object_file="$$build_dir/gate_e_guardian_controller_restart_witness_v1.o"; \
	cleanup() { rm -f "$$object_file"; rmdir "$$build_dir"; }; \
	trap cleanup EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) -fanalyzer -c "$(TOOL_DIRECTORY)gate_e_guardian_controller_restart_witness_v1.c" -o "$$object_file"
