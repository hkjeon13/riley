FROM ubuntu:22.04@sha256:45e9ddc6895604674618a409ca9f73d301da8fdb75fe330e8e9a27b3b219e4ad

# Use the signed Ubuntu archive as it existed when the reviewed binary was
# available. The runner trust store is mounted only for the package install.
RUN --mount=type=bind,source=ca-certificates.crt,target=/etc/ssl/certs/ca-certificates.crt \
    set -eu; \
    snapshot=https://snapshot.ubuntu.com/ubuntu/20260826T090000Z; \
    for suite in jammy jammy-updates jammy-security; do \
        printf 'deb [check-valid-until=no signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] %s %s main\n' \
            "$snapshot" "$suite"; \
    done > /tmp/ubuntu-snapshot.list; \
    apt_snapshot() { \
        apt-get -o Dir::Etc::sourcelist=/tmp/ubuntu-snapshot.list -o Dir::Etc::sourceparts=- "$@"; \
    }; \
    apt_snapshot update; \
    version='3.10.12-1~22.04.16'; \
    DEBIAN_FRONTEND=noninteractive apt_snapshot install -y --no-install-recommends \
        "python3.10=$version" "python3.10-minimal=$version" \
        "libpython3.10-minimal=$version" "libpython3.10-stdlib=$version"; \
    printf '%s  %s\n' 7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86 /usr/bin/python3.10 \
        | sha256sum --check; \
    rm -f /tmp/ubuntu-snapshot.list; \
    rm -rf /var/lib/apt/lists/*
