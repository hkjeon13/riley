FROM ubuntu:22.04@sha256:45e9ddc6895604674618a409ca9f73d301da8fdb75fe330e8e9a27b3b219e4ad

# The build context contains only the historical Python packages. APT
# resolves their ordinary system dependencies from the default Jammy archive.
RUN --mount=type=bind,target=/tmp/legacy-python \
    set -eu; \
    (cd /tmp/legacy-python && sha256sum --check packages.sha256); \
    apt-get -o Acquire::Retries=3 update; \
    DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        /tmp/legacy-python/*.deb; \
    printf '%s  %s\n' 7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86 /usr/bin/python3.10 \
        | sha256sum --check; \
    rm -rf /var/lib/apt/lists/*
