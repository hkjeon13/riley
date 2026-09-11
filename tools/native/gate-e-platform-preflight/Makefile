CC ?= cc
CPPFLAGS ?= -D_GNU_SOURCE
CFLAGS ?= -O2
LDFLAGS ?=

REQUIRED_CFLAGS = -std=c11 -Wall -Wextra -Werror -Wpedantic \
	-fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
REQUIRED_LDFLAGS = -pie -Wl,-z,relro,-z,now

.PHONY: test

test:
	@set -eu; \
	build_dir="$$(mktemp -d "$${TMPDIR:-/tmp}/riley-gate-e-platform-preflight.XXXXXX")"; \
	test_binary="$$build_dir/test_gate_e_platform_preflight"; \
		$(CC) $(CPPFLAGS) $(CFLAGS) $(REQUIRED_CFLAGS) test_gate_e_platform_preflight.c -o "$$test_binary" $(LDFLAGS) $(REQUIRED_LDFLAGS); \
	"$$test_binary"; \
	rm -f "$$test_binary"; \
	rmdir "$$build_dir"
